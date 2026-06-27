"""
graph/session_graph.py

LangGraph session orchestration — implementa o grafo de estados definido
em session_graph.yaml para rastreamento previsível no LangSmith e integração
com o Nerve Layer event bus.

Estados:
  1. init — Carrega Brief, verifica versão
  2. load_context — Carrega AGENTS.md, GUARDRAILS.md, etc.
  3. load_skill — Carrega skill do domínio (opcional)
  4. execute — Executa feature (LLM ou template)
  5. verify — Validação computacional + inferencial
  6. persist — Persiste resultado, emite evento final

Interrupt signals monitoram eventos do Nerve Layer durante execução.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Optional


# ---------------------------------------------------------------------------
# State & Enums
# ---------------------------------------------------------------------------

class NodeStatus(str, Enum):
    """Status de um nó após execução."""
    SUCCESS = "success"
    FAILURE = "failure"
    TIMEOUT = "timeout"
    SKIPPED = "skipped"


class FinalStatus(str, Enum):
    """Status final da sessão."""
    COMPLETE = "complete"
    BLOCKED = "blocked"
    ABORTED = "aborted"


@dataclass
class NodeExecutionResult:
    """Resultado de executar um nó."""
    node_id: str
    status: NodeStatus
    duration_ms: int
    outputs: dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None


@dataclass
class SessionState:
    """Estado da sessão durante execução."""
    session_id: str
    started_at: str
    current_node: str
    execution_history: list[NodeExecutionResult] = field(default_factory=list)

    # Contexto acumulado
    brief_version: Optional[str] = None
    context_loaded: bool = False
    skill_loaded: bool = False
    verify_passed: bool = True

    # Flags de interrupt
    context_stale: bool = False
    skill_stale: bool = False

    # Resultados
    final_status: Optional[FinalStatus] = None
    output_files: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Node implementations
# ---------------------------------------------------------------------------

class SessionGraphNodes:
    """Implementações dos nós do grafo."""

    @staticmethod
    def init(state: SessionState) -> SessionState:
        """
        Nó 1: Initialize Session
        Carrega Brief, verifica versão, baseline tests.
        """
        start = datetime.now(timezone.utc)

        try:
            # Carregar Brief (simulado — determinístico)
            brief_version = "1.0.0"
            state.brief_version = brief_version

            # Verificar versão
            assert brief_version, "Brief version required"

            # Baseline test (simulado)
            baseline_pass = True

            duration = int((datetime.now(timezone.utc) - start).total_seconds() * 1000)
            result = NodeExecutionResult(
                node_id="init",
                status=NodeStatus.SUCCESS,
                duration_ms=duration,
                outputs={
                    "brief_version": brief_version,
                    "baseline_pass": baseline_pass,
                },
            )
            state.execution_history.append(result)
            state.current_node = "load_context"
            return state

        except Exception as e:
            duration = int((datetime.now(timezone.utc) - start).total_seconds() * 1000)
            result = NodeExecutionResult(
                node_id="init",
                status=NodeStatus.FAILURE,
                duration_ms=duration,
                error=str(e),
            )
            state.execution_history.append(result)
            state.final_status = FinalStatus.ABORTED
            return state

    @staticmethod
    def load_context(state: SessionState) -> SessionState:
        """
        Nó 2: Load Project Context
        Carrega AGENTS.md, GUARDRAILS.md, PLAYBOOK.md, decisions.
        """
        start = datetime.now(timezone.utc)

        try:
            # Simular carregamento de contexto
            context_id = "ctx-" + state.session_id[:8]
            guardrails_count = 12
            decisions_loaded = 5

            state.context_loaded = True

            duration = int((datetime.now(timezone.utc) - start).total_seconds() * 1000)
            result = NodeExecutionResult(
                node_id="load_context",
                status=NodeStatus.SUCCESS,
                duration_ms=duration,
                outputs={
                    "context_id": context_id,
                    "guardrails_count": guardrails_count,
                    "decisions_loaded": decisions_loaded,
                },
            )
            state.execution_history.append(result)
            state.current_node = "load_skill"
            return state

        except Exception as e:
            duration = int((datetime.now(timezone.utc) - start).total_seconds() * 1000)
            result = NodeExecutionResult(
                node_id="load_context",
                status=NodeStatus.FAILURE,
                duration_ms=duration,
                error=str(e),
            )
            state.execution_history.append(result)
            state.final_status = FinalStatus.ABORTED
            return state

    @staticmethod
    def load_skill(state: SessionState) -> SessionState:
        """
        Nó 3: Load Domain Skill (Optional)
        Falha aqui NÃO bloqueia — graceful degradation.
        """
        start = datetime.now(timezone.utc)

        try:
            # Simular carregamento de skill (pode falhar)
            skill_id = "agendamento"
            skill_version = "1.0.0"
            vocabulary = ["reserva", "profissional", "cliente"]

            state.skill_loaded = True

            duration = int((datetime.now(timezone.utc) - start).total_seconds() * 1000)
            result = NodeExecutionResult(
                node_id="load_skill",
                status=NodeStatus.SUCCESS,
                duration_ms=duration,
                outputs={
                    "skill_id": skill_id,
                    "skill_version": skill_version,
                    "vocabulary": vocabulary,
                    "skill_loaded": True,
                },
            )
            state.execution_history.append(result)
            state.current_node = "execute"
            return state

        except Exception as e:
            # Graceful degradation — continue sem skill
            duration = int((datetime.now(timezone.utc) - start).total_seconds() * 1000)
            result = NodeExecutionResult(
                node_id="load_skill",
                status=NodeStatus.FAILURE,
                duration_ms=duration,
                error=str(e),
                outputs={"skill_loaded": False},
            )
            state.execution_history.append(result)
            state.current_node = "execute"  # Continue anyway
            return state

    @staticmethod
    def execute(state: SessionState) -> SessionState:
        """
        Nó 4: Execute Feature
        Executa LLM ou template. Pode falhar, mas persiste o resultado.
        """
        start = datetime.now(timezone.utc)

        try:
            # Simular execução (LLM ou template)
            output_files = [
                "src/main.py",
                "tests/test_main.py",
                "README.md",
            ]
            state.output_files = output_files

            duration = int((datetime.now(timezone.utc) - start).total_seconds() * 1000)
            result = NodeExecutionResult(
                node_id="execute",
                status=NodeStatus.SUCCESS,
                duration_ms=duration,
                outputs={
                    "output_files": output_files,
                    "execution_time_ms": duration,
                    "validation_pass": True,
                },
            )
            state.execution_history.append(result)
            state.current_node = "verify"
            return state

        except Exception as e:
            duration = int((datetime.now(timezone.utc) - start).total_seconds() * 1000)
            result = NodeExecutionResult(
                node_id="execute",
                status=NodeStatus.FAILURE,
                duration_ms=duration,
                error=str(e),
            )
            state.execution_history.append(result)
            state.current_node = "persist"  # Go to persist anyway
            return state

    @staticmethod
    def verify(state: SessionState) -> SessionState:
        """
        Nó 5: Verify & Validate
        Verificação computacional (pytest, lint) + inferencial (LLM judge).
        Não bloqueia se falhar.
        """
        start = datetime.now(timezone.utc)

        try:
            # Simular verificação
            tests_pass = True
            lint_pass = True
            type_check_pass = True
            llm_quality_score = 8.5  # 0-10

            state.verify_passed = all([tests_pass, lint_pass, type_check_pass])

            duration = int((datetime.now(timezone.utc) - start).total_seconds() * 1000)
            result = NodeExecutionResult(
                node_id="verify",
                status=NodeStatus.SUCCESS,
                duration_ms=duration,
                outputs={
                    "tests_pass": tests_pass,
                    "lint_pass": lint_pass,
                    "type_check_pass": type_check_pass,
                    "llm_quality_score": llm_quality_score,
                },
            )
            state.execution_history.append(result)
            state.current_node = "persist"
            return state

        except Exception as e:
            # Mesmo se falhar, vai para persist
            duration = int((datetime.now(timezone.utc) - start).total_seconds() * 1000)
            result = NodeExecutionResult(
                node_id="verify",
                status=NodeStatus.FAILURE,
                duration_ms=duration,
                error=str(e),
            )
            state.execution_history.append(result)
            state.verify_passed = False
            state.current_node = "persist"
            return state

    @staticmethod
    def persist(state: SessionState) -> SessionState:
        """
        Nó 6: Persist & Emit Event
        Atualiza logs, emite harness.task.complete ou harness.task.blocked.
        Sempre bem-sucedido (final node).
        """
        start = datetime.now(timezone.utc)

        try:
            # Decidir status final
            if state.verify_passed and not state.context_stale:
                final_status = FinalStatus.COMPLETE
            else:
                final_status = FinalStatus.BLOCKED

            state.final_status = final_status

            # Simular append ao sessions.log
            log_entry = {
                "session_id": state.session_id,
                "started_at": state.started_at,
                "completed_at": datetime.now(timezone.utc).isoformat(),
                "final_status": final_status.value,
                "output_files": state.output_files,
            }

            duration = int((datetime.now(timezone.utc) - start).total_seconds() * 1000)
            result = NodeExecutionResult(
                node_id="persist",
                status=NodeStatus.SUCCESS,
                duration_ms=duration,
                outputs={
                    "final_status": final_status.value,
                    "session_id": state.session_id,
                },
            )
            state.execution_history.append(result)
            state.current_node = "END"
            return state

        except Exception as e:
            # Mesmo se falhar, marca como END
            duration = int((datetime.now(timezone.utc) - start).total_seconds() * 1000)
            result = NodeExecutionResult(
                node_id="persist",
                status=NodeStatus.FAILURE,
                duration_ms=duration,
                error=str(e),
            )
            state.execution_history.append(result)
            state.final_status = FinalStatus.BLOCKED
            state.current_node = "END"
            return state


# ---------------------------------------------------------------------------
# Graph orchestration
# ---------------------------------------------------------------------------

class SessionGraph:
    """Orquestra a execução do grafo de estados."""

    def __init__(self, session_id: str):
        self.session_id = session_id
        self.nodes = {
            "init": SessionGraphNodes.init,
            "load_context": SessionGraphNodes.load_context,
            "load_skill": SessionGraphNodes.load_skill,
            "execute": SessionGraphNodes.execute,
            "verify": SessionGraphNodes.verify,
            "persist": SessionGraphNodes.persist,
        }

    def run(self) -> SessionState:
        """Executa o grafo completo."""
        state = SessionState(
            session_id=self.session_id,
            started_at=datetime.now(timezone.utc).isoformat(),
            current_node="init",
        )

        while state.current_node != "END":
            node_fn = self.nodes.get(state.current_node)
            if not node_fn:
                break
            state = node_fn(state)

        return state

    @staticmethod
    def execution_summary(state: SessionState) -> dict[str, Any]:
        """Retorna resumo da execução para LangSmith trace."""
        total_duration = sum(r.duration_ms for r in state.execution_history)
        return {
            "session_id": state.session_id,
            "final_status": state.final_status.value if state.final_status else None,
            "total_duration_ms": total_duration,
            "nodes_executed": len(state.execution_history),
            "output_files": state.output_files,
            "execution_history": [
                {
                    "node_id": r.node_id,
                    "status": r.status.value,
                    "duration_ms": r.duration_ms,
                    "error": r.error,
                }
                for r in state.execution_history
            ],
        }
