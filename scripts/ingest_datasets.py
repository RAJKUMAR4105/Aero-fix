"""Ingest the aviation knowledge datasets into the existing AeroFix Chroma index.

    python scripts/ingest_datasets.py            # incremental (skips known chunk IDs)
    python scripts/ingest_datasets.py --rebuild  # clear the collection first

Pipeline: load dataset -> normalize -> structure-aware chunk -> local embeddings
(BAAI/bge-small-en-v1.5) -> Chroma (existing backend) -> manifest.json.

Datasets used
  - tpattanayak/aeroground, config retrieval_corpus   (knowledge)
  - Remixonwin/amt-airframe-handbook-dataset         (knowledge)
The AeroGround `abstention` config is EVALUATION ONLY and is never indexed.
The AeroGround `dpo` config is never loaded.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import re
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from llama_index.core.schema import MetadataMode, TextNode
from llama_index.core.vector_stores.utils import node_to_metadata_dict

from aerofix.config import get_settings
from aerofix.ingest import ChromaBackend, get_embedding_model

TARGET_WORDS = 260          # ~350 tokens per chunk
OVERLAP_WORDS = 40
AMT_TITLE = "AMT Airframe Handbook (Remixonwin extract)"


# --------------------------------------------------------------------------
# normalization helpers (only fields the datasets actually carry)
# --------------------------------------------------------------------------
def _clean(value: Any) -> str | None:
    if value is None:
        return None
    text = " ".join(str(value).split())
    return text or None


def parse_page(*candidates: Any) -> int:
    """Real page number from the dataset text itself, else 0 = 'not available'."""
    for candidate in candidates:
        if candidate is None:
            continue
        if isinstance(candidate, int):
            return candidate
        match = re.search(r"\bpp?\.\s*(\d+)", str(candidate))
        if match:
            return int(match.group(1))
    return 0


def document_title_for(source: str) -> str:
    """Book-level title so the UI's 'manuals indexed' count stays meaningful."""
    if ", 14 CFR" in source:
        return source.split(", 14 CFR")[0].strip()
    return source


def document_type_for(source: str) -> str:
    lowered = source.lower()
    if "handbook" in lowered:
        return "handbook"
    if "cfr" in lowered or "ecfr" in lowered:
        return "regulation"
    return "reference"


def split_long_text(text: str, target: int = TARGET_WORDS, overlap: int = OVERLAP_WORDS) -> list[str]:
    """Deterministic word-based split; keeps long passages inside budget."""
    words = text.split()
    if len(words) <= target:
        return [text]
    pieces: list[str] = []
    start = 0
    while start < len(words):
        end = min(start + target, len(words))
        pieces.append(" ".join(words[start:end]))
        if end >= len(words):
            break
        start = max(end - overlap, start + 1)
    return pieces


# --------------------------------------------------------------------------
# dataset -> normalized records
# --------------------------------------------------------------------------
def normalize_aeroground(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for row in rows:
        text = _clean(row.get("text"))
        if not text:
            continue
        source = _clean(row.get("source")) or "Unknown source"
        citation = _clean(row.get("citation"))
        title = _clean(row.get("title"))
        records.append(
            {
                "text": text,
                "dataset": "aeroground",
                "document_title": document_title_for(source),
                "source": source,
                "section": title,                 # dataset field `title`
                "page": parse_page(citation, title),
                "citation": citation,
                "domain": _clean(row.get("domain")),
                "document_type": document_type_for(source),
                "external_id": _clean(row.get("id")),
                # aircraft / engine / ata_chapter do not exist in this dataset
                # and are therefore never written.
            }
        )
    return records


def normalize_amt(row: dict[str, Any]) -> list[dict[str, Any]]:
    """One dataset row holds the whole document: parallel paragraph arrays."""
    paragraphs = row.get("paragraphs") or {}
    if not isinstance(paragraphs, dict):
        return []
    texts = paragraphs.get("text") or []
    pages = paragraphs.get("page") or [0] * len(texts)
    para_ids = paragraphs.get("para_id") or [None] * len(texts)

    records: list[dict[str, Any]] = []
    bucket: list[str] = []
    bucket_words = 0
    bucket_page = int(pages[0]) if pages else 0
    bucket_first_para = para_ids[0] if para_ids else None
    index_on_page: dict[int, int] = {}

    def flush() -> None:
        if not bucket:
            return
        index_on_page[bucket_page] = index_on_page.get(bucket_page, 0) + 1
        records.append(
            {
                "text": " ".join(bucket),
                "dataset": "amt-airframe-handbook",
                "document_title": AMT_TITLE,
                "source": "Remixonwin/amt-airframe-handbook-dataset",
                "section": None,        # the dataset exposes a single degenerate section
                "page": bucket_page,
                "citation": f"Remixonwin/amt-airframe-handbook-dataset, p.{bucket_page}",
                "domain": None,
                "document_type": "question_bank",
                "external_id": bucket_first_para,
                "chunk_index_on_page": index_on_page[bucket_page],
            }
        )

    for text, page, para_id in zip(texts, pages, para_ids):
        text = _clean(text)
        if not text:
            continue
        page = int(page or 0)
        words = len(text.split())
        if page != bucket_page and bucket:
            flush()
            bucket, bucket_words = [], 0
            bucket_page = page
            bucket_first_para = para_id
        if bucket_words and bucket_words + words > TARGET_WORDS:
            flush()
            bucket, bucket_words = [], 0
            bucket_page = page
            bucket_first_para = para_id
        bucket.append(text)
        bucket_words += words
    flush()
    return records


# --------------------------------------------------------------------------
# chunking: deterministic chunk IDs
# --------------------------------------------------------------------------
def chunk_id_for(record: dict[str, Any], part: int, parts: int) -> str:
    slug = re.sub(r"[^A-Za-z0-9]+", "-", str(record.get("external_id") or "doc")).strip("-")
    dataset = record["dataset"]
    if dataset == "aeroground":
        base = f"aeroground_document_{slug}_page_{record['page']:04d}"
    else:
        base = f"amt_airframe_document_{slug}_page_{record['page']:04d}"
    if parts > 1:
        return f"{base}_chunk_{part + 1:02d}"
    return base


def to_nodes(records: list[dict[str, Any]]) -> list[TextNode]:
    nodes: list[TextNode] = []
    for record in records:
        pieces = split_long_text(record["text"])
        for part, text in enumerate(pieces):
            metadata: dict[str, Any] = {
                "chunk_id": chunk_id_for(record, part, len(pieces)),
                "document_title": record["document_title"],
                "source": record["source"],
                "page": int(record.get("page") or 0),
                "section": record.get("section") or "Unsectioned source text",
                "citation": record.get("citation") or "",
                "dataset": record["dataset"],
                "document_type": record.get("document_type") or "reference",
            }
            for key in ("domain", "external_id", "aircraft", "engine", "ata_chapter"):
                value = record.get(key)
                if value:  # never write unknown metadata
                    metadata[key] = str(value)
            nodes.append(TextNode(id_=metadata["chunk_id"], text=text, metadata=metadata))
    return nodes


# --------------------------------------------------------------------------
def load_sources() -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any]]:
    from datasets import load_dataset

    stats: dict[str, Any] = {}
    normalized: dict[str, list[dict[str, Any]]] = {}

    print("Loading AeroGround retrieval_corpus...")
    started = time.perf_counter()
    aeroground = load_dataset("tpattanayak/aeroground", "retrieval_corpus")
    rows = [dict(r) for r in (aeroground["train"] if "train" in aeroground else aeroground)]
    print(f"  Records: {len(rows)}  ({time.perf_counter() - started:.1f}s)")
    normalized["aeroground"] = normalize_aeroground(rows)
    stats["aeroground"] = {
        "name": "tpattanayak/aeroground",
        "config": "retrieval_corpus",
        "rows": len(rows),
        "chunks": len(normalized["aeroground"]),
    }

    print("Loading AMT Airframe Handbook...")
    started = time.perf_counter()
    amt = load_dataset("Remixonwin/amt-airframe-handbook-dataset")
    amt_rows = [dict(r) for r in (amt["train"] if "train" in amt else amt)]
    print(f"  Records: {len(amt_rows)}  ({time.perf_counter() - started:.1f}s)")
    amt_records: list[dict[str, Any]] = []
    for row in amt_rows:
        amt_records.extend(normalize_amt(row))
    normalized["amt-airframe-handbook"] = amt_records
    stats["amt-airframe-handbook"] = {
        "name": "Remixonwin/amt-airframe-handbook-dataset",
        "config": "default",
        "rows": len(amt_rows),
        "chunks": len(amt_records),
    }
    return normalized, stats


def write_manifest(path: Path, stats: dict[str, Any], settings: Any, indexed: int) -> None:
    manifest = {
        "ingested_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "index_version": 1,
        "collection": settings.collection_name,
        "embedding_provider": settings.embedding_provider,
        "embedding_model": settings.embedding_model,
        "passages_indexed": indexed,
        "datasets": list(stats.values()),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")


# --------------------------------------------------------------------------
# bulk write: identical records to LlamaIndex's reader, 30x faster
# --------------------------------------------------------------------------
def add_direct(backend: ChromaBackend, nodes: list[TextNode], embed_model: Any) -> None:
    """Write nodes straight to Chroma in the exact format LlamaIndex reads back.

    The VectorStoreIndex write path manages only ~6 passages/s (per-node
    wrapper overhead); embedding is ~17/s and raw Chroma add is ~230/s, so this
    serializes with node_to_metadata_dict(..., remove_text=True) exactly like
    llama_index.vector_stores.chroma.ChromaVectorStore.add does and writes the
    batch in one call.
    """
    texts = [node.get_content(metadata_mode=MetadataMode.NONE) for node in nodes]
    vectors = embed_model.get_text_embedding_batch(texts, show_progress=False)
    if len(vectors) != len(nodes):  # pragma: no cover - defensive
        raise RuntimeError(f"embedding count mismatch: {len(vectors)} != {len(nodes)}")
    metadatas: list[dict[str, Any]] = []
    for node in nodes:
        meta = node_to_metadata_dict(node, remove_text=True, flat_metadata=True)
        for key, value in meta.items():
            if value is None:  # Chroma only accepts scalar metadata values
                meta[key] = ""
        metadatas.append(meta)
    backend.collection.add(
        ids=[node.node_id for node in nodes],
        embeddings=[[float(x) for x in vector] for vector in vectors],
        metadatas=metadatas,
        documents=texts,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest HF aviation datasets into AeroFix.")
    parser.add_argument("--rebuild", action="store_true", help="Clear the collection first.")
    args = parser.parse_args()

    settings = get_settings()
    settings.ensure_runtime_directories()
    embed_model = get_embedding_model(settings)
    backend = ChromaBackend(settings, embed_model)

    if args.rebuild:
        old_ids = backend.collection.get(include=[])["ids"]
        print(f"Rebuilding collection '{settings.collection_name}' ({len(old_ids)} old passages)...")
        if old_ids:
            for offset in range(0, len(old_ids), 5000):
                backend.collection.delete(ids=old_ids[offset : offset + 5000])

    normalized, stats = load_sources()
    print("Creating normalized passages...")
    nodes = [node for records in normalized.values() for node in to_nodes(records)]
    print(f"Passages created: {len(nodes)}")

    existing = backend.existing_ids()
    fresh = [node for node in nodes if node.node_id not in existing]
    print(f"Already indexed: {len(nodes) - len(fresh)}   New: {len(fresh)}")

    started = time.perf_counter()
    print(f"Generating embeddings ({settings.embedding_model}) + updating index...")
    batch = 512
    for offset in range(0, len(fresh), batch):
        chunk = fresh[offset : offset + batch]
        add_direct(backend, chunk, embed_model)
        done = min(offset + len(chunk), len(fresh))
        elapsed = time.perf_counter() - started
        rate = done / elapsed if elapsed else 0.0
        print(f"  {done}/{len(fresh)} passages  ({elapsed:.0f}s, {rate:.1f}/s)", flush=True)

    indexed = backend.collection.count()
    write_manifest(Path("data/knowledge/manifest.json"), stats, settings, indexed)

    documents = len(
        {
            item.get("document_title")
            for item in backend.collection.get(include=["metadatas"])["metadatas"]
            if item and item.get("document_title")
        }
    )
    print()
    print("AeroFix Knowledge Base")
    print("----------------------")
    print(f"Documents: {documents}")
    print(f"Passages: {indexed}")
    print(f"Embeddings: {indexed} ({settings.embedding_model})")
    print(f"Ingestion wall time: {time.perf_counter() - started:.0f}s")


if __name__ == "__main__":
    main()
