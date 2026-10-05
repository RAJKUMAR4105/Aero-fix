"""Multiprocess local embedding pool (torch / sentence-transformers).

CPU-only inference for BAAI/bge-small-en-v1.5 is limited on this machine
(~8 texts/s single process), so batch embedding is split over worker
processes. Workers are reused across batches and keep their own model.
"""

from __future__ import annotations

import os
from concurrent.futures import ProcessPoolExecutor
from typing import Any

_EXECUTOR: ProcessPoolExecutor | None = None
_WORKER_MODELS: dict[str, Any] = {}
_WORKER_BATCH = 64


def _model(model_name: str) -> Any:
    model = _WORKER_MODELS.get(model_name)
    if model is None:
        import torch
        from sentence_transformers import SentenceTransformer

        torch.set_num_threads(2)
        model = SentenceTransformer(model_name)
        _WORKER_MODELS[model_name] = model
    return model


def _embed_chunk(payload: tuple[str, list[str]]) -> list[list[float]]:
    model_name, texts = payload
    if not texts:
        return []
    vectors = _model(model_name).encode(
        texts, batch_size=min(_WORKER_BATCH, len(texts)), show_progress_bar=False
    )
    return [vector.tolist() for vector in vectors]


def _executor(workers: int) -> ProcessPoolExecutor:
    global _EXECUTOR
    if _EXECUTOR is None:
        _EXECUTOR = ProcessPoolExecutor(max_workers=workers)
    return _EXECUTOR


def embed_batch(texts: list[str], model_name: str, workers: int) -> list[list[float]]:
    """Embed texts in worker processes, preserving input order."""
    if not texts:
        return []
    if workers <= 1 or len(texts) < _WORKER_BATCH:
        return _embed_chunk((model_name, texts))
    chunks = [
        texts[offset : offset + _WORKER_BATCH]
        for offset in range(0, len(texts), _WORKER_BATCH)
    ]
    executor = _executor(workers)
    results = list(executor.map(_embed_chunk, [(model_name, chunk) for chunk in chunks]))
    return [vector for group in results for vector in group]


def shutdown() -> None:
    """Stop worker processes (called after bulk ingestion)."""
    global _EXECUTOR
    if _EXECUTOR is not None:
        _EXECUTOR.shutdown(wait=False, cancel_futures=True)
        _EXECUTOR = None
