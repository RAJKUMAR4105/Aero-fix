"""Hybrid (vector + keyword) retrieval adapter returning citation-ready chunks.

Vector search comes from Chroma through LlamaIndex; an in-memory BM25 pass adds
exact-term matches (part numbers, ATA/section references) and the two lists are
fused with Reciprocal Rank Fusion. Every failure on the keyword side falls back
to pure vector retrieval, so the existing LangGraph path never breaks.
"""

from __future__ import annotations

import re
from collections import Counter
from math import log

from llama_index.core import VectorStoreIndex

from aerofix.config import Settings, get_settings
from aerofix.ingest import ChromaBackend, get_embedding_model
from aerofix.models import ChunkRecord

_TOKEN_RE = re.compile(r"[a-z0-9]+")
_RRF_K = 60

# Question/function words that carry no retrieval meaning. Content terms left
# after this filter decide whether the top chunk can actually answer the query.
_STOPWORDS = frozenset(
    """
    a about after all also an and any are as at be been before being both but by
    can could did do does down during each exact exactly few for from further get
    give had has have handbook how i if in into is it its just may me might more
    most much must my no nor not of on one only or other out over own please same
    say should so some such tell than that the their them then there these they
    this those through to too under up us very via was we were what when where
    whether which while who whom why will with would you your
    """.split()
)


def _tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


class BM25Index:
    """Compact in-memory BM25 over the persisted corpus (no external service)."""

    def __init__(self, ids: list[str], documents: list[str]) -> None:
        self.ids = ids
        self.n = len(ids)
        self.k1 = 1.5
        self.b = 0.75
        self.doc_len: list[int] = []
        self.postings: dict[str, list[tuple[int, int]]] = {}
        total = 0
        for idx, doc in enumerate(documents or []):
            terms = _tokenize(doc or "")
            self.doc_len.append(len(terms))
            total += len(terms)
            for term, tf in Counter(terms).items():
                self.postings.setdefault(term, []).append((idx, tf))
        self.avg_len = (total / len(documents)) if documents else 1.0

    def search(self, query: str, top_k: int) -> list[tuple[str, float]]:
        if not self.n or not self.avg_len:
            return []
        scores: dict[int, float] = {}
        for term in set(_tokenize(query)):
            postings = self.postings.get(term)
            if not postings:
                continue
            df = len(postings)
            idf = log(1.0 + (self.n - df + 0.5) / (df + 0.5))
            for idx, tf in postings:
                dl = self.doc_len[idx] or 1
                norm = self.k1 * (1.0 - self.b + self.b * dl / self.avg_len)
                scores[idx] = scores.get(idx, 0.0) + idf * ((tf * (self.k1 + 1.0)) / (tf + norm))
        ranked = sorted(scores.items(), key=lambda item: -item[1])[:top_k]
        return [(self.ids[idx], float(score)) for idx, score in ranked]


class AeroFixRetriever:
    """Retrieve maintenance chunks with immutable source metadata."""

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self.embedding_model = get_embedding_model(self.settings)
        self.backend = ChromaBackend(self.settings, self.embedding_model)
        self.index = VectorStoreIndex.from_vector_store(
            self.backend.storage_context.vector_store,
            embed_model=self.embedding_model,
        )
        self._bm25: BM25Index | None = None

    # -- keyword side --------------------------------------------------------
    def _keyword_index(self) -> BM25Index | None:
        if self._bm25 is None:
            data = self.backend.collection.get(include=["documents"])
            if not data["ids"]:
                return None
            self._bm25 = BM25Index(data["ids"], data["documents"])
        return self._bm25

    # -- scoring -------------------------------------------------------------
    @staticmethod
    def _as_similarity(raw_score: float) -> float:
        """LlamaIndex's Chroma adapter returns exp(-cosine_distance); invert it
        back to a raw cosine similarity so thresholds keep a real meaning."""
        if raw_score <= 0.0:
            return 0.0
        if raw_score >= 1.0:
            return 1.0
        return max(-1.0, min(1.0, 1.0 + log(raw_score)))

    def _record_from_node(self, result: object) -> ChunkRecord:
        node = getattr(result, "node", None)
        metadata = node.metadata
        return ChunkRecord(
            chunk_id=str(metadata.get("chunk_id") or node.node_id),
            text=node.get_content(),
            document_title=str(metadata.get("document_title", "Unknown document")),
            source=str(metadata.get("source", "Unknown source")),
            page=int(metadata.get("page", 0)),
            section=str(metadata.get("section", "Unsectioned source text")),
            score=round(self._as_similarity(float(getattr(result, "score", 0.0) or 0.0)), 4),
            citation=str(metadata.get("citation", "")),
            dataset=str(metadata.get("dataset", "")),
        )

    # -- main retrieval ------------------------------------------------------
    def retrieve(self, query: str, top_k: int | None = None) -> list[ChunkRecord]:
        """Return the best matching chunks, vector and keyword results fused."""
        k = top_k or self.settings.retrieval_top_k
        candidate_k = max(k, getattr(self.settings, "retrieval_candidate_k", k))

        retriever = self.index.as_retriever(similarity_top_k=candidate_k)
        results = retriever.retrieve(query)
        chunks = [self._record_from_node(result) for result in results]
        if not getattr(self.settings, "hybrid", False):
            return chunks[:k]

        try:
            keyword_hits = self._keyword_fusion(query, chunks, candidate_k)
        except Exception:  # pragma: no cover - keyword side is best effort
            keyword_hits = None
        if keyword_hits is None:
            return chunks[:k]

        fused = self._rrf(chunks, keyword_hits)
        return fused[:k]

    def _keyword_fusion(
        self, query: str, vector_chunks: list[ChunkRecord], candidate_k: int
    ) -> list[ChunkRecord] | None:
        """BM25 over the full corpus; missing chunks are fetched and scored."""
        bm25 = self._keyword_index()
        if bm25 is None:
            return None
        hits = bm25.search(query, top_k=candidate_k)
        if not hits:
            return []
        known = {chunk.chunk_id: chunk for chunk in vector_chunks}
        missing = [chunk_id for chunk_id, _ in hits if chunk_id not in known]
        extra: dict[str, ChunkRecord] = {}
        if missing:
            data = self.backend.collection.get(
                ids=missing, include=["documents", "metadatas", "embeddings"]
            )
            query_vector = self.embedding_model.get_query_embedding(query)
            for cid, document, metadata, embedding in zip(
                data["ids"], data["documents"], data["metadatas"], data["embeddings"]
            ):
                if embedding is None:
                    continue
                extra[str(cid)] = ChunkRecord(
                    chunk_id=str((metadata or {}).get("chunk_id") or cid),
                    text=document or "",
                    document_title=str((metadata or {}).get("document_title", "Unknown document")),
                    source=str((metadata or {}).get("source", "Unknown source")),
                    page=int((metadata or {}).get("page", 0)),
                    section=str((metadata or {}).get("section", "Unsectioned source text")),
                    score=round(self._cosine(query_vector, [float(v) for v in embedding]), 4),
                    citation=str((metadata or {}).get("citation", "")),
                    dataset=str((metadata or {}).get("dataset", "")),
                )
        merged: list[ChunkRecord] = []
        for cid, _ in hits:
            chunk = known.get(cid) or extra.get(cid)
            if chunk is not None:
                merged.append(chunk)
        return merged

    @staticmethod
    def _cosine(left: list[float], right: list[float]) -> float:
        dot = sum(a * b for a, b in zip(left, right))
        left_norm = sum(a * a for a in left) ** 0.5
        right_norm = sum(b * b for b in right) ** 0.5
        if not left_norm or not right_norm:
            return 0.0
        return max(-1.0, min(1.0, dot / (left_norm * right_norm)))

    @staticmethod
    def _rrf(vector_chunks: list[ChunkRecord], keyword_chunks: list[ChunkRecord]) -> list[ChunkRecord]:
        """Reciprocal Rank Fusion over the two ranked lists; scores stay cosine."""
        order: list[str] = []
        fused: dict[str, float] = {}
        by_id: dict[str, ChunkRecord] = {}
        for rank, chunk in enumerate(vector_chunks):
            if chunk.chunk_id not in fused:
                order.append(chunk.chunk_id)
            fused[chunk.chunk_id] = fused.get(chunk.chunk_id, 0.0) + 1.0 / (_RRF_K + rank + 1)
            by_id.setdefault(chunk.chunk_id, chunk)
        for rank, chunk in enumerate(keyword_chunks):
            if chunk.chunk_id not in fused:
                order.append(chunk.chunk_id)
            fused[chunk.chunk_id] = fused.get(chunk.chunk_id, 0.0) + 1.0 / (_RRF_K + rank + 1)
            by_id.setdefault(chunk.chunk_id, chunk)
        order.sort(key=lambda cid: -fused[cid])
        return [by_id[cid] for cid in order]


def _content_terms(query: str) -> list[str]:
    """Meaningful query terms (stems kept; stopwords and 1-char tokens dropped)."""
    return [token for token in _tokenize(query) if len(token) >= 2 and token not in _STOPWORDS]


def retrieval_confidence(chunks: list[ChunkRecord], query: str | None = None) -> float:
    """Blend similarity with how well the top chunk covers the query.

    Cosine alone measures topicality, not answerability: an off-corpus question
    ("...part number and price of a Cessna 172 nosewheel tire") still scores
    ~0.65 against a vaguely related manual page. So when a query is provided the
    score is additionally scaled by (a) the lexical coverage of the query's
    content terms in the chunk that will answer it, and (b) query specificity,
    so underspecified prompts ("How much torque, exactly?") fall below the
    threshold instead of producing a fabricated answer. Without a query the
    original similarity-only blend is returned unchanged.
    """
    scores = [chunk.score or 0.0 for chunk in chunks]
    if not scores:
        return 0.0
    base = (0.7 * max(scores)) + (0.3 * (sum(scores) / len(scores)))
    if not query or not chunks:
        return round(base, 4)
    terms = _content_terms(query)
    if not terms:
        return round(base, 4)
    top_tokens = set(_tokenize(chunks[0].text))
    hits = sum(1 for term in terms if any(tok.startswith(term) for tok in top_tokens))
    coverage = hits / len(terms)
    specificity = min(1.0, len(terms) / 2.0)
    floor = 0.2  # similarity retained even with zero lexical overlap (paraphrases)
    return round(base * (floor + (1.0 - floor) * coverage) * specificity, 4)
