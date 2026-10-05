"""PDF/text ingestion into persistent Chroma through LlamaIndex."""

from __future__ import annotations

import argparse
import hashlib
import logging
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import chromadb
import fitz
from llama_index.core import StorageContext, VectorStoreIndex
from llama_index.core.embeddings import BaseEmbedding, MockEmbedding
from llama_index.core.schema import TextNode
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore
from pydantic import PrivateAttr

from aerofix.config import Settings, get_settings

LOGGER = logging.getLogger(__name__)
TARGET_TOKENS = 650
OVERLAP_TOKENS = 120


@dataclass(frozen=True)
class SourcePage:
    """A page extracted from a source document."""

    source: Path
    document_title: str
    page: int
    text: str


class VectorBackend(Protocol):
    """Port for a persistent vector backend such as Chroma or Pinecone."""

    def existing_ids(self) -> set[str]: ...

    def add(self, nodes: list[TextNode]) -> None: ...


class ChromaBackend:
    """Local persistent Chroma implementation of the vector backend port."""

    def __init__(self, settings: Settings, embed_model: BaseEmbedding) -> None:
        settings.ensure_runtime_directories()
        client = chromadb.PersistentClient(path=str(settings.chroma_dir))
        collection = client.get_or_create_collection(
            name=settings.collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        self.collection = collection
        self.storage_context = StorageContext.from_defaults(
            vector_store=ChromaVectorStore(chroma_collection=collection)
        )
        self.embed_model = embed_model

    def existing_ids(self) -> set[str]:
        result = self.collection.get(include=[])
        return set(result["ids"])

    def add(self, nodes: list[TextNode]) -> None:
        if nodes:
            VectorStoreIndex(
                nodes=nodes,
                storage_context=self.storage_context,
                embed_model=self.embed_model,
                show_progress=False,
            )


class LocalEmbedding(BaseEmbedding):
    """Local sentence embedding (BAAI/bge-small-en-v1.5 via sentence-transformers).

    Runs entirely on CPU - no OpenAI key is required to index or query.
    """

    _model: Any = PrivateAttr(default=None)
    _workers: int = PrivateAttr(default=1)
    _cache: Any = PrivateAttr(default=None)

    def __init__(
        self,
        model_name: str = "BAAI/bge-small-en-v1.5",
        workers: int = 1,
        **kwargs: Any,
    ) -> None:
        kwargs.setdefault("embed_batch_size", 64)
        super().__init__(model_name=model_name, **kwargs)
        import torch
        from sentence_transformers import SentenceTransformer

        torch.set_num_threads(max(1, (os.cpu_count() or 4) // 2))
        self._model = SentenceTransformer(model_name)
        self._workers = int(workers)
        self._cache = {}

    def _get_query_embedding(self, query: str) -> list[float]:
        return self._model.encode(query, show_progress_bar=False).tolist()

    async def _aget_query_embedding(self, query: str) -> list[float]:
        return self._get_query_embedding(query)

    def _get_text_embedding(self, text: str) -> list[float]:
        return self._get_text_embeddings([text])[0]

    def _get_text_embeddings(self, texts: list[str]) -> list[list[float]]:
        """Cached batch embedding; large miss sets go to the worker pool."""
        vectors: list[list[float] | None] = [None] * len(texts)
        missing: list[int] = []
        for position, text in enumerate(texts):
            cached = (self._cache or {}).get(text)
            if cached is None:
                missing.append(position)
            else:
                vectors[position] = cached
        if missing:
            missing_texts = [texts[position] for position in missing]
            if self._workers > 1 and len(missing_texts) >= 64:
                from aerofix.embed_pool import embed_batch

                computed = embed_batch(missing_texts, self.model_name, self._workers)
            else:
                computed = [
                    row.tolist()
                    for row in self._model.encode(
                        missing_texts,
                        batch_size=self.embed_batch_size,
                        show_progress_bar=False,
                    )
                ]
            for position, vector in zip(missing, computed):
                self._cache[texts[position]] = vector
                vectors[position] = vector
        return [vector for vector in vectors if vector is not None]


def get_embedding_model(settings: Settings) -> BaseEmbedding:
    """Create the configured LlamaIndex embedding adapter."""
    provider = settings.embedding_provider.lower()
    if provider == "mock":
        return MockEmbedding(embed_dim=384)
    if provider in {"huggingface", "local", "bge"}:
        return LocalEmbedding(
            model_name=settings.embedding_model,
            workers=getattr(settings, "embedding_workers", 1),
        )
    if provider == "openai":
        return OpenAIEmbedding(model=settings.embedding_model)
    raise ValueError(
        "Unsupported embedding provider. Use 'huggingface' (local), 'openai', or 'mock'."
    )


def extract_pages(path: Path) -> list[SourcePage]:
    """Extract text and document metadata from PDFs or clearly labelled text fixtures."""
    if path.suffix.lower() == ".pdf":
        document = fitz.open(path)
        title = str(document.metadata.get("title") or path.stem).strip()
        return [
            SourcePage(path, title, index + 1, page.get_text("text").strip())
            for index, page in enumerate(document)
            if page.get_text("text").strip()
        ]
    if path.suffix.lower() == ".txt":
        return [SourcePage(path, path.stem, 1, path.read_text(encoding="utf-8").strip())]
    return []


def section_heading(text: str) -> str:
    """Use the nearest apparent heading; fall back to an explicit unknown marker."""
    for line in text.splitlines()[:12]:
        compact = " ".join(line.split())
        if re.match(r"^(\d+(?:\.\d+)*\s+)?[A-Z][A-Z0-9 /,.-]{4,}$", compact):
            return compact
        if re.match(r"^\d+(?:\.\d+)+\s+", compact):
            return compact
    return "Unsectioned source text"


def _blocks(text: str) -> list[str]:
    """Keep procedures and warning/caution blocks indivisible where possible."""
    raw = [" ".join(block.split()) for block in re.split(r"\n\s*\n", text) if block.strip()]
    blocks: list[str] = []
    current = ""
    for block in raw:
        is_guardrail = bool(re.match(r"^(WARNING|CAUTION)\b", block, re.IGNORECASE))
        is_numbered = bool(re.match(r"^\d+[.)]\s+", block))
        if (is_guardrail or is_numbered) and current:
            blocks.append(current)
            current = block
        elif current:
            current = f"{current}\n\n{block}"
        else:
            current = block
        if is_guardrail:
            continue
        if len(current.split()) >= TARGET_TOKENS:
            blocks.append(current)
            current = ""
    if current:
        blocks.append(current)
    return blocks


def chunk_page(page: SourcePage) -> list[TextNode]:
    """Chunk a source page at paragraph boundaries with bounded overlap."""
    nodes: list[TextNode] = []
    recent: list[str] = []
    current: list[str] = []
    current_words = 0
    for block in _blocks(page.text):
        words = len(block.split())
        if current and current_words + words > TARGET_TOKENS:
            nodes.append(_node_from_blocks(page, current))
            overlap: list[str] = []
            overlap_words = 0
            for prior in reversed(current):
                overlap.insert(0, prior)
                overlap_words += len(prior.split())
                if overlap_words >= OVERLAP_TOKENS:
                    break
            recent = overlap
            current = [*recent, block]
            current_words = sum(len(item.split()) for item in current)
        else:
            current.append(block)
            current_words += words
    if current:
        nodes.append(_node_from_blocks(page, current))
    return nodes


def _node_from_blocks(page: SourcePage, blocks: list[str]) -> TextNode:
    text = "\n\n".join(blocks)
    section = section_heading(text)
    digest_input = f"{page.source.resolve()}:{page.page}:{text}".encode("utf-8")
    chunk_id = hashlib.sha256(digest_input).hexdigest()[:24]
    return TextNode(
        id_=chunk_id,
        text=text,
        metadata={
            "chunk_id": chunk_id,
            "document_title": page.document_title,
            "source": page.source.name,
            "page": page.page,
            "section": section,
        },
    )


def ingest_directory(
    source_dir: Path, backend: VectorBackend, *, include_text_fixtures: bool = False
) -> dict[str, int]:
    """Ingest new source chunks only, returning per-document chunk counts."""
    allowed = {".pdf"}
    if include_text_fixtures:
        allowed.add(".txt")
    existing_ids = backend.existing_ids()
    counts: dict[str, int] = {}
    for path in sorted(source_dir.iterdir()):
        if not path.is_file() or path.suffix.lower() not in allowed:
            continue
        nodes = [node for page in extract_pages(path) for node in chunk_page(page)]
        new_nodes = [node for node in nodes if node.node_id not in existing_ids]
        backend.add(new_nodes)
        counts[path.name] = len(new_nodes)
        LOGGER.info("Ingested %s: %s new chunks", path.name, len(new_nodes))
    return counts


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest AeroFix maintenance sources.")
    parser.add_argument("--source-dir", type=Path, default=None)
    parser.add_argument("--include-sample", action="store_true")
    args = parser.parse_args()
    settings = get_settings()
    source_dir = args.source_dir or settings.data_dir / "raw"
    counts = ingest_directory(
        source_dir,
        ChromaBackend(settings, get_embedding_model(settings)),
        include_text_fixtures=args.include_sample,
    )
    if not counts:
        print(f"No eligible source files found in {source_dir}")
        return
    for document, count in counts.items():
        print(f"{document}: {count} new chunks")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    main()

