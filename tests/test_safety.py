from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from aerofix.audit import AuditLog
from aerofix.config import Settings
from aerofix.graph import AeroFixGraph, WorkflowState, validate_citations
from aerofix.llm import MockLLM
from aerofix.models import ChunkRecord, DraftResponse, DraftStep, Status


class FakeRetriever:
    def __init__(self) -> None:
        self.calls = 0

    def retrieve(self, query: str, top_k: int) -> list[ChunkRecord]:
        self.calls += 1
        return [
            ChunkRecord(
                chunk_id="source-1",
                text="1. Record the symptom.\n\nWARNING: Test warning.",
                document_title="Test Manual",
                source="test.pdf",
                page=3,
                section="Troubleshooting",
                score=0.95,
            )
        ]


def test_commit_requires_review_and_reviewer_id(tmp_path: Path) -> None:
    graph = build_graph(tmp_path)
    with pytest.raises(PermissionError):
        graph._commit(WorkflowState(fault_input="rough idle", status=Status.DRAFTING))
    with pytest.raises(PermissionError):
        graph._commit(WorkflowState(fault_input="rough idle", status=Status.PENDING_REVIEW))
    graph.close()


def test_fabricated_citation_is_removed() -> None:
    chunks = FakeRetriever().retrieve("rough idle", 3)
    draft = DraftResponse(
        steps=[DraftStep(n=1, action="Invented action", citation_ids=["fabricated-id"])]
    )
    validated, flags = validate_citations(draft, chunks)
    assert validated.steps == []
    assert "fabricated-id" in flags[0]


def test_tampering_breaks_audit_chain(tmp_path: Path) -> None:
    path = tmp_path / "audit.sqlite"
    audit = AuditLog(path)
    audit.append(
        thread_id="thread-1",
        fault_input="rough idle",
        retrieved_chunk_ids=["source-1"],
        draft={},
        final_steps=[],
        decision="reject",
        reviewer_id="TECH-1",
        reject_reason="Test",
    )
    assert audit.verify_chain()[0] is True
    audit.connection.execute("UPDATE audit_log SET fault_input = 'tampered' WHERE id = 1")
    audit.connection.commit()
    assert audit.verify_chain()[0] is False
    audit.close()


def test_interrupted_state_survives_restart(tmp_path: Path) -> None:
    graph = build_graph(tmp_path)
    graph.start("rough idle", "persisted-thread")
    assert graph.state("persisted-thread").status == Status.PENDING_REVIEW
    graph.close()

    restarted = build_graph(tmp_path)
    from aerofix.models import ReviewPayload

    result = restarted.resume(
        "persisted-thread", ReviewPayload(human_decision="approve", reviewer_id="TECH-9")
    )
    assert result["status"] == Status.APPROVED
    restarted.close()


def test_explicitly_out_of_scope_input_does_not_retrieve(tmp_path: Path) -> None:
    retriever = FakeRetriever()
    graph = build_graph(tmp_path, retriever=retriever)
    graph.start("turbofan EGT exceeds redline on a Boeing 737", "outside-scope")
    state = graph.state("outside-scope")
    assert state.status == Status.INSUFFICIENT_DATA
    assert retriever.calls == 0
    graph.close()


def build_graph(tmp_path: Path, retriever: FakeRetriever | None = None) -> AeroFixGraph:
    settings = Settings(
        chroma_dir=tmp_path / "chroma",
        sqlite_path=tmp_path / "audit.sqlite",
        checkpointer_path=tmp_path / "checkpoints.sqlite",
        confidence_threshold=0.1,
        llm_provider="mock",
        embedding_provider="mock",
    )
    return AeroFixGraph(settings, retriever=retriever or FakeRetriever(), llm=MockLLM())
