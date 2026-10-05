"""Typed contracts shared across ingestion, retrieval, graph, and audit layers."""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field, field_validator


class Status(str, Enum):
    DRAFTING = "DRAFTING"
    PENDING_REVIEW = "PENDING_REVIEW"
    APPROVED = "APPROVED"
    REJECTED = "REJECTED"
    INSUFFICIENT_DATA = "INSUFFICIENT_DATA"


class HumanDecision(str, Enum):
    APPROVE = "approve"
    EDIT = "edit"
    REJECT = "reject"


class ChunkRecord(BaseModel):
    chunk_id: str
    text: str
    document_title: str
    source: str
    page: int
    section: str
    score: float | None = None
    # Optional provenance from the knowledge datasets (absent = unknown).
    citation: str = ""
    dataset: str = ""


class DraftStep(BaseModel):
    n: int = Field(ge=1)
    action: str = Field(min_length=1)
    citation_ids: list[str] = Field(min_length=1)
    warning: str | None = None


class DraftResponse(BaseModel):
    steps: list[DraftStep] = Field(default_factory=list)
    unresolved_questions: list[str] = Field(default_factory=list)
    disclaimer: str = (
        "Draft only. Requires certified technician review and must be verified "
        "against current approved maintenance data."
    )


class ReviewPayload(BaseModel):
    human_decision: HumanDecision
    reviewer_id: str = Field(min_length=1)
    edited_steps: list[DraftStep] | None = None
    reject_reason: str | None = None

    @field_validator("reviewer_id")
    @classmethod
    def reviewer_id_must_not_be_blank(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("reviewer_id is required")
        return value


class AuditRecord(BaseModel):
    id: int | None = None
    timestamp: datetime
    thread_id: str
    fault_input: str
    retrieved_chunk_ids: list[str]
    draft: dict[str, Any]
    final_steps: list[dict[str, Any]]
    decision: str
    reviewer_id: str
    reject_reason: str | None = None
    prev_hash: str
    record_hash: str | None = None
