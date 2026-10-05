"""Grounded RAG question answering over the AeroFix knowledge index.

    python scripts/ask.py "What does the handbook say about ...?"

Rules enforced here (source-grounded generation):
  * the answer is extracted verbatim from retrieved source text - never written
    by a model - so it cannot invent values, limits, part numbers or citations;
  * if confidence is below the configured threshold the script returns exactly
    "Insufficient verified source information.";
  * every answer carries its source document and citation/page.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from aerofix.config import get_settings
from aerofix.models import ChunkRecord
from aerofix.retriever import AeroFixRetriever, retrieval_confidence

INSUFFICIENT = "Insufficient verified source information."

_STOP = {
    "the", "a", "an", "of", "for", "and", "or", "to", "in", "on", "is", "are",
    "what", "how", "does", "do", "when", "should", "can", "with", "from", "by",
    "be", "this", "that", "it", "at", "as", "if", "into", "their", "there",
}

_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+")


def _answer_terms(text: str) -> set[str]:
    return {
        token
        for token in re.findall(r"[a-z0-9]+", text.lower())
        if len(token) > 1 and token not in _STOP
    }


def extractive_answer(query: str, chunk: ChunkRecord, max_chars: int = 700) -> str:
    """Pick the source sentences that actually address the query (verbatim)."""
    terms = _answer_terms(query)
    sentences = [s.strip() for s in _SENTENCE_RE.split(chunk.text.strip()) if s.strip()]
    if not sentences:
        return ""
    scored = sorted(
        enumerate(sentences),
        key=lambda pair: -len(terms & _answer_terms(pair[1])),
    )
    chosen_idx = [idx for idx, sentence in scored if terms & _answer_terms(sentence)] or [0]
    chosen_idx = sorted(chosen_idx[:3])
    answer = " ".join(sentences[idx] for idx in chosen_idx)
    if not answer:
        answer = sentences[0]
    if len(answer) > max_chars:
        answer = answer[: max_chars - 3].rsplit(" ", 1)[0] + "..."
    return answer


def _citation_line(chunk: ChunkRecord) -> str:
    page = f"page {chunk.page}" if chunk.page else "page not available"
    citation = chunk.citation or "no formal citation in source record"
    return f"{citation} ({page})"


def answer_query(
    retriever: AeroFixRetriever, query: str, top_k: int
) -> dict[str, object]:
    chunks = retriever.retrieve(query, top_k=top_k)
    confidence = retrieval_confidence(chunks, query)
    settings = get_settings()
    if not chunks or confidence < settings.confidence_threshold:
        return {
            "question": query,
            "answer": INSUFFICIENT,
            "confidence": confidence,
            "threshold": settings.confidence_threshold,
            "retrieved": [],
        }
    best = chunks[0]
    return {
        "question": query,
        "answer": extractive_answer(query, best) or INSUFFICIENT,
        "source": best.document_title,
        "citation": _citation_line(best),
        "section": best.section,
        "chunk_id": best.chunk_id,
        "confidence": confidence,
        "threshold": settings.confidence_threshold,
        "retrieved": [
            {
                "rank": rank,
                "score": chunk.score,
                "source": chunk.document_title,
                "citation": chunk.citation,
                "page": chunk.page,
                "chunk_id": chunk.chunk_id,
            }
            for rank, chunk in enumerate(chunks, start=1)
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Ask the AeroFix knowledge base.")
    parser.add_argument("question", help="The question to answer.")
    parser.add_argument("--top-k", type=int, default=None, help="Chunks to retrieve.")
    parser.add_argument("--json", action="store_true", help="JSON output.")
    args = parser.parse_args()

    settings = get_settings()
    retriever = AeroFixRetriever(settings)
    result = answer_query(retriever, args.question, args.top_k or settings.retrieval_top_k)

    if args.json:
        print(json.dumps(result, indent=2))
        return

    print(f"QUESTION: {result['question']}")
    print(f"CONFIDENCE: {result['confidence']} (threshold {result['threshold']})")
    print(f"ANSWER: {result['answer']}")
    if result["answer"] != INSUFFICIENT:
        print(f"SOURCE: {result['source']}")
        print(f"CITATION: {result['citation']}")
        print(f"SECTION: {result['section']}")
        print(f"CHUNK: {result['chunk_id']}")
    if result["retrieved"]:
        print("\nRETRIEVED SOURCES:")
        for hit in result["retrieved"]:
            print(
                f"  {hit['rank']}. score={hit['score']}  {hit['source']}"
                f"  [{hit['citation'] or 'no citation'} p.{hit['page']}]"
                f"  {hit['chunk_id']}"
            )


if __name__ == "__main__":
    main()
