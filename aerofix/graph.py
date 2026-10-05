"""LangGraph workflow with a persisted, mandatory human review interrupt."""

from __future__ import annotations

import re
import sqlite3
from typing import Any

from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt
from pydantic import BaseModel, Field

from aerofix.audit import AuditLog
from aerofix.config import Settings, get_settings
from aerofix.llm import DraftingLLM, get_llm
from aerofix.models import ChunkRecord, DraftResponse, DraftStep, HumanDecision, ReviewPayload, Status
from aerofix.retriever import AeroFixRetriever, retrieval_confidence

OUT_OF_SCOPE_PATTERN = re.compile(
    r"\b(boeing|airbus|turbofan|turbojet|helicopter|rotorcraft|main rotor|"
    r"chest pain|medical|weather|ifr alternate|air traffic control)\b",
    re.IGNORECASE,
)


class WorkflowState(BaseModel):
    """Persisted graph state. Every state field is typed and serializable."""

    thread_id: str = ""
    fault_input: str = ""
    retrieved_chunks: list[ChunkRecord] = Field(default_factory=list)
    draft: DraftResponse | None = None
    confidence: float = 0.0
    status: Status = Status.DRAFTING
    reviewer_id: str | None = None
    human_decision: HumanDecision | None = None
    edited_steps: list[DraftStep] | None = None
    reject_reason: str | None = None
    message: str | None = None


def validate_citations(
    draft: DraftResponse, retrieved_chunks: list[ChunkRecord]
) -> tuple[DraftResponse, list[str]]:
    """Drop any draft step that lacks citations from the retrieved result set."""
    available = {chunk.chunk_id for chunk in retrieved_chunks}
    valid_steps: list[DraftStep] = []
    flags: list[str] = []
    for step in draft.steps:
        unresolved = [citation for citation in step.citation_ids if citation not in available]
        if not step.citation_ids or unresolved:
            flags.append(f"Step {step.n} was removed: unresolved citations {unresolved or ['none']}")
            continue
        valid_steps.append(step)
    return draft.model_copy(update={"steps": valid_steps}), flags


class AeroFixGraph:
    """Application service for starting and safely resuming maintenance workflows."""

    def __init__(
        self,
        settings: Settings | None = None,
        *,
        retriever: AeroFixRetriever | Any | None = None,
        llm: DraftingLLM | None = None,
        audit_log: AuditLog | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.settings.ensure_runtime_directories()
        self.retriever = retriever or AeroFixRetriever(self.settings)
        self.llm = llm or get_llm(self.settings)
        self.audit_log = audit_log or AuditLog(self.settings.sqlite_path)
        self._checkpoint_connection = sqlite3.connect(
            self.settings.checkpointer_path, check_same_thread=False
        )
        self.checkpointer = SqliteSaver(self._checkpoint_connection)
        self.app = self._build()

    def start(self, fault_input: str, thread_id: str) -> dict[str, Any]:
        """Start a workflow; execution stops at the LangGraph review interrupt."""
        return self.app.invoke(
            {"fault_input": fault_input, "thread_id": thread_id},
            config={"configurable": {"thread_id": thread_id}},
        )

    def resume(self, thread_id: str, review: ReviewPayload) -> dict[str, Any]:
        """Resume only the existing interrupted thread with a validated human decision."""
        return self.app.invoke(
            Command(resume=review.model_dump(mode="json")),
            config={"configurable": {"thread_id": thread_id}},
        )

    def state(self, thread_id: str) -> WorkflowState | None:
        snapshot = self.app.get_state({"configurable": {"thread_id": thread_id}})
        return WorkflowState.model_validate(snapshot.values) if snapshot.values else None

    def close(self) -> None:
        self._checkpoint_connection.close()
        self.audit_log.close()

    def _build(self):
        graph = StateGraph(WorkflowState)
        graph.add_node("intake", self._intake)
        graph.add_node("retrieve", self._retrieve)
        graph.add_node("draft", self._draft)
        graph.add_node("validate", self._validate)
        graph.add_node("review_gate", self._review_gate)
        graph.add_node("commit", self._commit)
        graph.add_node("insufficient", self._insufficient)
        graph.add_edge(START, "intake")
        graph.add_edge("intake", "retrieve")
        graph.add_conditional_edges(
            "retrieve", self._route_after_retrieval, {"draft": "draft", "insufficient": "insufficient"}
        )
        graph.add_edge("draft", "validate")
        graph.add_conditional_edges(
            "validate", self._route_after_validation, {"review": "review_gate", "insufficient": "insufficient"}
        )
        graph.add_edge("review_gate", "commit")
        graph.add_edge("commit", END)
        graph.add_edge("insufficient", END)
        return graph.compile(checkpointer=self.checkpointer)

    @staticmethod
    def _typed(state: WorkflowState | dict[str, Any]) -> WorkflowState:
        return state if isinstance(state, WorkflowState) else WorkflowState.model_validate(state)

    def _intake(self, state: WorkflowState | dict[str, Any]) -> dict[str, Any]:
        current = self._typed(state)
        normalized = re.sub(r"\s+", " ", current.fault_input).strip()
        if not normalized:
            return {"status": Status.INSUFFICIENT_DATA, "message": "Enter a fault symptom or code."}
        if OUT_OF_SCOPE_PATTERN.search(normalized):
            return {
                "fault_input": normalized,
                "status": Status.INSUFFICIENT_DATA,
                "message": "This request is outside the Cessna 172 / Lycoming O-320 maintenance scope.",
            }
        return {"fault_input": normalized, "status": Status.DRAFTING}

    def _retrieve(self, state: WorkflowState | dict[str, Any]) -> dict[str, Any]:
        current = self._typed(state)
        if current.status == Status.INSUFFICIENT_DATA:
            return {"retrieved_chunks": [], "confidence": 0.0}
        chunks = self.retriever.retrieve(current.fault_input, self.settings.retrieval_top_k)
        confidence = retrieval_confidence(chunks, current.fault_input)
        return {"retrieved_chunks": chunks, "confidence": confidence}

    def _route_after_retrieval(self, state: WorkflowState | dict[str, Any]) -> str:
        current = self._typed(state)
        if current.status == Status.INSUFFICIENT_DATA or current.confidence < self.settings.confidence_threshold:
            return "insufficient"
        return "draft"

    def _draft(self, state: WorkflowState | dict[str, Any]) -> dict[str, Any]:
        current = self._typed(state)
        return {"draft": self.llm.draft(current.fault_input, current.retrieved_chunks)}

    def _validate(self, state: WorkflowState | dict[str, Any]) -> dict[str, Any]:
        current = self._typed(state)
        if current.draft is None:
            return {"status": Status.INSUFFICIENT_DATA, "message": "The drafting service returned no draft."}
        draft, flags = validate_citations(current.draft, current.retrieved_chunks)
        if not draft.steps:
            return {
                "draft": draft,
                "status": Status.INSUFFICIENT_DATA,
                "message": "No draft steps had valid citations. " + " ".join(flags),
            }
        return {
            "draft": draft,
            "status": Status.PENDING_REVIEW,
            "message": " ".join(flags) if flags else None,
        }

    def _route_after_validation(self, state: WorkflowState | dict[str, Any]) -> str:
        return "insufficient" if self._typed(state).status == Status.INSUFFICIENT_DATA else "review"

    def _review_gate(self, state: WorkflowState | dict[str, Any]) -> dict[str, Any]:
        current = self._typed(state)
        review_payload = interrupt(
            {
                "status": current.status.value,
                "fault_input": current.fault_input,
                "draft": current.draft.model_dump(mode="json") if current.draft else None,
                "message": "Human approval, edit, or rejection is required before audit commit.",
            }
        )
        review = ReviewPayload.model_validate(review_payload)
        return review.model_dump(mode="json")

    def _commit(self, state: WorkflowState | dict[str, Any]) -> dict[str, Any]:
        current = self._typed(state)
        if current.status != Status.PENDING_REVIEW:
            raise PermissionError("Commit is only allowed after the pending review gate.")
        if not current.reviewer_id or not current.human_decision:
            raise PermissionError("A reviewer identity and human decision are required to commit.")
        if current.human_decision == HumanDecision.EDIT and not current.edited_steps:
            raise ValueError("Edited approval requires at least one edited step.")
        final_steps = (
            current.edited_steps
            if current.human_decision == HumanDecision.EDIT
            else (current.draft.steps if current.human_decision == HumanDecision.APPROVE and current.draft else [])
        )
        status = Status.REJECTED if current.human_decision == HumanDecision.REJECT else Status.APPROVED
        self.audit_log.append(
            thread_id=current.thread_id,
            fault_input=current.fault_input,
            retrieved_chunk_ids=[chunk.chunk_id for chunk in current.retrieved_chunks],
            draft=current.draft.model_dump(mode="json") if current.draft else {},
            final_steps=[step.model_dump(mode="json") for step in final_steps],
            decision=current.human_decision.value,
            reviewer_id=current.reviewer_id,
            reject_reason=current.reject_reason,
        )
        return {"status": status, "message": "Human decision was committed to the audit log."}

    @staticmethod
    def _insufficient(state: WorkflowState | dict[str, Any]) -> dict[str, Any]:
        current = state if isinstance(state, WorkflowState) else WorkflowState.model_validate(state)
        return {
            "status": Status.INSUFFICIENT_DATA,
            "message": current.message
            or "Retrieved evidence is below the configured confidence threshold; no draft was created.",
        }
