"""RAG correctness tests (offline): grounded answers, exact refusal, calibration.

These tests never touch the network or the live index: they exercise the
query-aware confidence gate, the MockLLM extractive drafting path, and the
exact insufficient-source contract.
"""

from __future__ import annotations

from pathlib import Path

from aerofix.config import Settings
from aerofix.graph import AeroFixGraph
from aerofix.llm import MockLLM
from aerofix.models import ChunkRecord, Status
from aerofix.retriever import retrieval_confidence
from ui import copy as C

INSUFFICIENT = "Insufficient verified source information."

GROUNDING_TEXT = (
    "Inspect the torque wrench scale before use. "
    "The wrench must be clean and free of dirt.\n\n"
    "WARNING: Do not use a damaged torque wrench."
)


def chunk(score: float, text: str = GROUNDING_TEXT, chunk_id: str = "src-1") -> ChunkRecord:
    return ChunkRecord(
        chunk_id=chunk_id,
        text=text,
        document_title="Test Manual",
        source="test.pdf",
        page=7,
        section="Inspection",
        score=score,
        citation="TEST-1, p. 7",
        dataset="unit-test",
    )


class FixedRetriever:
    """Returns a canned result set for any query."""

    def __init__(self, chunks: list[ChunkRecord]) -> None:
        self.chunks = chunks
        self.calls = 0

    def retrieve(self, query: str, top_k: int) -> list[ChunkRecord]:
        self.calls += 1
        return self.chunks


def build(tmp_path: Path, retriever: FixedRetriever) -> AeroFixGraph:
    settings = Settings(
        chroma_dir=tmp_path / "chroma",
        sqlite_path=tmp_path / "audit.sqlite",
        checkpointer_path=tmp_path / "checkpoints.sqlite",
        confidence_threshold=0.45,
        llm_provider="mock",
        embedding_provider="mock",
    )
    return AeroFixGraph(settings, retriever=retriever, llm=MockLLM())


# -- confidence calibration ----------------------------------------------------

def test_grounded_query_clears_threshold() -> None:
    chunks = [chunk(0.70)]
    confidence = retrieval_confidence(
        chunks, "What does the handbook say about torque wrench inspection?"
    )
    assert confidence >= 0.45, confidence


def test_out_of_corpus_query_falls_below_threshold() -> None:
    # Top chunk is about tires, but the question asks for a Cessna part number
    # and price the corpus does not contain: only the word "tire" matches.
    chunks = [chunk(0.68, text="Each wheel must be fitted with a suitable tire assembly.")]
    confidence = retrieval_confidence(
        chunks, "What is the part number and price of the Cessna 172 nosewheel tire?"
    )
    assert confidence < 0.45, confidence


def test_ambiguous_query_falls_below_threshold() -> None:
    # One content term, fully covered: still underspecified for a procedure.
    chunks = [chunk(0.73, text="The limit engine torque may not be less than the following.")]
    confidence = retrieval_confidence(chunks, "How much torque, exactly?")
    assert confidence < 0.45, confidence


def test_no_query_keeps_similarity_only_blend() -> None:
    scores = [0.80, 0.60]
    expected = round((0.7 * 0.80) + (0.3 * 0.70), 4)  # 0.7*max + 0.3*mean
    assert retrieval_confidence([chunk(0.80), chunk(0.60)]) == expected


def test_empty_chunks_score_zero() -> None:
    assert retrieval_confidence([]) == 0.0
    assert retrieval_confidence([], "any query") == 0.0


# -- exact refusal contract ----------------------------------------------------

def test_insufficient_message_is_exact() -> None:
    assert C.INSUFFICIENT_MSG == INSUFFICIENT


def test_low_confidence_run_refuses_with_exact_status(tmp_path: Path) -> None:
    # Query terms absent from the top chunk + weak scores -> refusal, no draft.
    retriever = FixedRetriever(
        [chunk(0.30, text="General information about aircraft cleaning schedules.")]
    )
    graph = build(tmp_path, retriever)
    graph.start("What is the GPS coordinates of the aircraft right now?", "refuse-thread")
    state = graph.state("refuse-thread")
    assert state.status == Status.INSUFFICIENT_DATA
    assert state.draft is None
    graph.close()


# -- grounded drafting ---------------------------------------------------------

def test_grounded_draft_cites_a_retrieved_chunk_verbatim(tmp_path: Path) -> None:
    retriever = FixedRetriever([chunk(0.72)])
    graph = build(tmp_path, retriever)
    graph.start("How do you inspect the torque wrench?", "grounded-thread")
    state = graph.state("grounded-thread")
    assert state.status == Status.PENDING_REVIEW
    assert state.draft is not None and state.draft.steps, state.draft
    step = state.draft.steps[0]
    # Every citation resolves to a chunk that was actually retrieved.
    retrieved_ids = {c.chunk_id for c in state.retrieved_chunks}
    assert set(step.citation_ids) <= retrieved_ids
    # The action is verbatim source text, never an invention.
    source = next(c for c in state.retrieved_chunks if c.chunk_id == step.citation_ids[0])
    assert step.action in source.text
    assert source.dataset == "unit-test"
    graph.close()


def test_insufficient_when_no_chunks_retrieve(tmp_path: Path) -> None:
    graph = build(tmp_path, FixedRetriever([]))
    graph.start("rough idle", "empty-thread")
    assert graph.state("empty-thread").status == Status.INSUFFICIENT_DATA
    graph.close()
