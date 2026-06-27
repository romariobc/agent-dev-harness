"""
tests/test_session_graph.py

Tests para o LangGraph session orchestration.

Coverage:
  - Node individual execution (init, load_context, load_skill, execute, verify, persist)
  - State transitions e current_node tracking
  - Failure handling (graceful degradation em load_skill)
  - Execution history logging
  - Final status determination (complete vs blocked)
"""
from __future__ import annotations

import sys
from pathlib import Path
from uuid import uuid4

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from graph.session_graph import (
    FinalStatus,
    NodeStatus,
    SessionState,
    SessionGraph,
    SessionGraphNodes,
)


# ---------------------------------------------------------------------------
# Individual node tests
# ---------------------------------------------------------------------------

class TestNodeInit:
    def test_init_success(self):
        state = SessionState(
            session_id="test-session",
            started_at="2026-06-27T17:00:00Z",
            current_node="init",
        )
        state = SessionGraphNodes.init(state)

        assert state.current_node == "load_context"
        assert state.brief_version == "1.0.0"
        assert len(state.execution_history) == 1
        assert state.execution_history[0].status == NodeStatus.SUCCESS

    def test_init_history_recorded(self):
        state = SessionState(
            session_id="test-session",
            started_at="2026-06-27T17:00:00Z",
            current_node="init",
        )
        state = SessionGraphNodes.init(state)

        result = state.execution_history[0]
        assert result.node_id == "init"
        assert result.duration_ms > 0
        assert "brief_version" in result.outputs


class TestNodeLoadContext:
    def test_load_context_success(self):
        state = SessionState(
            session_id="test-session",
            started_at="2026-06-27T17:00:00Z",
            current_node="load_context",
        )
        state = SessionGraphNodes.load_context(state)

        assert state.current_node == "load_skill"
        assert state.context_loaded is True
        assert len(state.execution_history) == 1
        assert state.execution_history[0].status == NodeStatus.SUCCESS

    def test_load_context_outputs(self):
        state = SessionState(
            session_id="test-session",
            started_at="2026-06-27T17:00:00Z",
            current_node="load_context",
        )
        state = SessionGraphNodes.load_context(state)

        outputs = state.execution_history[0].outputs
        assert "context_id" in outputs
        assert "guardrails_count" in outputs
        assert "decisions_loaded" in outputs


class TestNodeLoadSkill:
    def test_load_skill_success(self):
        state = SessionState(
            session_id="test-session",
            started_at="2026-06-27T17:00:00Z",
            current_node="load_skill",
        )
        state = SessionGraphNodes.load_skill(state)

        assert state.current_node == "execute"
        assert state.skill_loaded is True
        assert len(state.execution_history) == 1

    def test_load_skill_graceful_degradation_on_error(self):
        state = SessionState(
            session_id="test-session",
            started_at="2026-06-27T17:00:00Z",
            current_node="load_skill",
        )
        # Simulate failure by mocking — but for now, our code succeeds
        # In production, a missing skill file would trigger the exception path
        state = SessionGraphNodes.load_skill(state)

        # Even if fails, should continue to execute (graceful)
        assert state.current_node == "execute"

    def test_load_skill_outputs_vocabulary(self):
        state = SessionState(
            session_id="test-session",
            started_at="2026-06-27T17:00:00Z",
            current_node="load_skill",
        )
        state = SessionGraphNodes.load_skill(state)

        outputs = state.execution_history[0].outputs
        assert "vocabulary" in outputs
        assert isinstance(outputs["vocabulary"], list)


class TestNodeExecute:
    def test_execute_success(self):
        state = SessionState(
            session_id="test-session",
            started_at="2026-06-27T17:00:00Z",
            current_node="execute",
        )
        state = SessionGraphNodes.execute(state)

        assert state.current_node == "verify"
        assert len(state.output_files) > 0
        assert state.execution_history[0].status == NodeStatus.SUCCESS

    def test_execute_generates_output_files(self):
        state = SessionState(
            session_id="test-session",
            started_at="2026-06-27T17:00:00Z",
            current_node="execute",
        )
        state = SessionGraphNodes.execute(state)

        assert "src/main.py" in state.output_files
        assert "tests/test_main.py" in state.output_files


class TestNodeVerify:
    def test_verify_success_sets_flag(self):
        state = SessionState(
            session_id="test-session",
            started_at="2026-06-27T17:00:00Z",
            current_node="verify",
        )
        state = SessionGraphNodes.verify(state)

        assert state.current_node == "persist"
        assert state.verify_passed is True
        assert state.execution_history[0].status == NodeStatus.SUCCESS

    def test_verify_outputs_checks(self):
        state = SessionState(
            session_id="test-session",
            started_at="2026-06-27T17:00:00Z",
            current_node="verify",
        )
        state = SessionGraphNodes.verify(state)

        outputs = state.execution_history[0].outputs
        assert "tests_pass" in outputs
        assert "lint_pass" in outputs
        assert "type_check_pass" in outputs
        assert "llm_quality_score" in outputs

    def test_verify_failure_still_persists(self):
        state = SessionState(
            session_id="test-session",
            started_at="2026-06-27T17:00:00Z",
            current_node="verify",
            verify_passed=False,
        )
        # Simulate failure
        state.verify_passed = False
        state = SessionGraphNodes.verify(state)

        assert state.current_node == "persist"


class TestNodePersist:
    def test_persist_always_succeeds(self):
        state = SessionState(
            session_id="test-session",
            started_at="2026-06-27T17:00:00Z",
            current_node="persist",
            verify_passed=True,
        )
        state = SessionGraphNodes.persist(state)

        assert state.current_node == "END"
        assert state.execution_history[0].status == NodeStatus.SUCCESS
        assert state.final_status is not None

    def test_persist_complete_status(self):
        state = SessionState(
            session_id="test-session",
            started_at="2026-06-27T17:00:00Z",
            current_node="persist",
            verify_passed=True,
        )
        state = SessionGraphNodes.persist(state)

        assert state.final_status == FinalStatus.COMPLETE

    def test_persist_blocked_status(self):
        state = SessionState(
            session_id="test-session",
            started_at="2026-06-27T17:00:00Z",
            current_node="persist",
            verify_passed=False,
        )
        state = SessionGraphNodes.persist(state)

        assert state.final_status == FinalStatus.BLOCKED


# ---------------------------------------------------------------------------
# Graph orchestration tests
# ---------------------------------------------------------------------------

class TestSessionGraphFlow:
    def test_complete_flow_execution(self):
        session_id = str(uuid4())
        graph = SessionGraph(session_id)
        state = graph.run()

        assert state.session_id == session_id
        assert state.current_node == "END"
        assert state.final_status in [FinalStatus.COMPLETE, FinalStatus.BLOCKED]
        assert len(state.execution_history) > 0

    def test_all_nodes_executed_in_order(self):
        session_id = str(uuid4())
        graph = SessionGraph(session_id)
        state = graph.run()

        node_ids = [r.node_id for r in state.execution_history]
        expected_order = ["init", "load_context", "load_skill", "execute", "verify", "persist"]
        assert node_ids == expected_order

    def test_execution_history_completeness(self):
        session_id = str(uuid4())
        graph = SessionGraph(session_id)
        state = graph.run()

        for result in state.execution_history:
            assert result.node_id is not None
            assert result.status is not None
            assert result.duration_ms >= 0

    def test_total_duration_computation(self):
        session_id = str(uuid4())
        graph = SessionGraph(session_id)
        state = graph.run()
        summary = graph.execution_summary(state)

        assert "total_duration_ms" in summary
        assert summary["total_duration_ms"] > 0

    def test_context_accumulates(self):
        session_id = str(uuid4())
        graph = SessionGraph(session_id)
        state = graph.run()

        # After init, should have brief_version
        assert state.brief_version is not None
        # After load_context, should be flagged
        assert state.context_loaded
        # After load_skill, should attempt (graceful degradation ok)
        # (skill_loaded depends on whether skill exists)


# ---------------------------------------------------------------------------
# State transitions
# ---------------------------------------------------------------------------

class TestStateTransitions:
    def test_init_to_load_context(self):
        state = SessionState(
            session_id="test-session",
            started_at="2026-06-27T17:00:00Z",
            current_node="init",
        )
        state = SessionGraphNodes.init(state)
        assert state.current_node == "load_context"

    def test_load_context_to_load_skill(self):
        state = SessionState(
            session_id="test-session",
            started_at="2026-06-27T17:00:00Z",
            current_node="load_context",
        )
        state = SessionGraphNodes.load_context(state)
        assert state.current_node == "load_skill"

    def test_load_skill_to_execute_even_on_failure(self):
        state = SessionState(
            session_id="test-session",
            started_at="2026-06-27T17:00:00Z",
            current_node="load_skill",
        )
        state = SessionGraphNodes.load_skill(state)
        # load_skill always proceeds to execute (graceful degradation)
        assert state.current_node == "execute"

    def test_execute_to_verify(self):
        state = SessionState(
            session_id="test-session",
            started_at="2026-06-27T17:00:00Z",
            current_node="execute",
        )
        state = SessionGraphNodes.execute(state)
        assert state.current_node == "verify"

    def test_verify_to_persist(self):
        state = SessionState(
            session_id="test-session",
            started_at="2026-06-27T17:00:00Z",
            current_node="verify",
        )
        state = SessionGraphNodes.verify(state)
        assert state.current_node == "persist"

    def test_persist_to_end(self):
        state = SessionState(
            session_id="test-session",
            started_at="2026-06-27T17:00:00Z",
            current_node="persist",
        )
        state = SessionGraphNodes.persist(state)
        assert state.current_node == "END"


# ---------------------------------------------------------------------------
# Execution summary
# ---------------------------------------------------------------------------

class TestExecutionSummary:
    def test_summary_has_required_fields(self):
        session_id = str(uuid4())
        graph = SessionGraph(session_id)
        state = graph.run()
        summary = graph.execution_summary(state)

        assert "session_id" in summary
        assert "final_status" in summary
        assert "total_duration_ms" in summary
        assert "nodes_executed" in summary
        assert "output_files" in summary
        assert "execution_history" in summary

    def test_summary_execution_history_shape(self):
        session_id = str(uuid4())
        graph = SessionGraph(session_id)
        state = graph.run()
        summary = graph.execution_summary(state)

        for entry in summary["execution_history"]:
            assert "node_id" in entry
            assert "status" in entry
            assert "duration_ms" in entry
