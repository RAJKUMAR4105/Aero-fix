"""Append-only, hash-chained SQLite audit log."""

from __future__ import annotations

import hashlib
import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from aerofix.models import AuditRecord

GENESIS_HASH = "0" * 64


class AuditLog:
    """Persist immutable review decisions and verify their hash chain."""

    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(path, check_same_thread=False)
        self.connection.row_factory = sqlite3.Row
        self.connection.execute(
            """
            CREATE TABLE IF NOT EXISTS audit_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                thread_id TEXT NOT NULL,
                fault_input TEXT NOT NULL,
                retrieved_chunk_ids TEXT NOT NULL,
                draft TEXT NOT NULL,
                final_steps TEXT NOT NULL,
                decision TEXT NOT NULL,
                reviewer_id TEXT NOT NULL,
                reject_reason TEXT,
                prev_hash TEXT NOT NULL,
                record_hash TEXT NOT NULL UNIQUE
            )
            """
        )
        self.connection.commit()

    def append(
        self,
        *,
        thread_id: str,
        fault_input: str,
        retrieved_chunk_ids: list[str],
        draft: dict[str, Any],
        final_steps: list[dict[str, Any]],
        decision: str,
        reviewer_id: str,
        reject_reason: str | None = None,
    ) -> AuditRecord:
        """Append one human decision, chaining it cryptographically to its predecessor."""
        previous = self.connection.execute(
            "SELECT record_hash FROM audit_log ORDER BY id DESC LIMIT 1"
        ).fetchone()
        prev_hash = str(previous["record_hash"]) if previous else GENESIS_HASH
        record = AuditRecord(
            timestamp=datetime.now(timezone.utc),
            thread_id=thread_id,
            fault_input=fault_input,
            retrieved_chunk_ids=retrieved_chunk_ids,
            draft=draft,
            final_steps=final_steps,
            decision=decision,
            reviewer_id=reviewer_id,
            reject_reason=reject_reason,
            prev_hash=prev_hash,
        )
        record_hash = self._record_hash(record)
        cursor = self.connection.execute(
            """
            INSERT INTO audit_log (
                timestamp, thread_id, fault_input, retrieved_chunk_ids, draft,
                final_steps, decision, reviewer_id, reject_reason, prev_hash, record_hash
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                record.timestamp.isoformat(),
                record.thread_id,
                record.fault_input,
                self._json(record.retrieved_chunk_ids),
                self._json(record.draft),
                self._json(record.final_steps),
                record.decision,
                record.reviewer_id,
                record.reject_reason,
                record.prev_hash,
                record_hash,
            ),
        )
        self.connection.commit()
        return record.model_copy(update={"id": cursor.lastrowid, "record_hash": record_hash})

    def records(self, limit: int = 100) -> list[AuditRecord]:
        """Return newest records first for the read-only UI audit view."""
        rows = self.connection.execute(
            "SELECT * FROM audit_log ORDER BY id DESC LIMIT ?", (limit,)
        ).fetchall()
        return [self._from_row(row) for row in rows]

    def verify_chain(self) -> tuple[bool, str]:
        """Recompute the full hash chain and report the first broken record."""
        expected_previous = GENESIS_HASH
        rows = self.connection.execute("SELECT * FROM audit_log ORDER BY id ASC").fetchall()
        for row in rows:
            record = self._from_row(row)
            if record.prev_hash != expected_previous:
                return False, f"Record {record.id} has an unexpected predecessor hash."
            if record.record_hash != self._record_hash(record):
                return False, f"Record {record.id} has an invalid record hash."
            expected_previous = str(record.record_hash)
        return True, f"Verified {len(rows)} audit record(s)."

    def close(self) -> None:
        self.connection.close()

    def _record_hash(self, record: AuditRecord) -> str:
        payload = record.model_dump(mode="json", exclude={"id", "record_hash"})
        return hashlib.sha256(
            f"{record.prev_hash}:{self._json(payload)}".encode("utf-8")
        ).hexdigest()

    @staticmethod
    def _json(value: Any) -> str:
        return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)

    @staticmethod
    def _from_row(row: sqlite3.Row) -> AuditRecord:
        return AuditRecord(
            id=int(row["id"]),
            timestamp=datetime.fromisoformat(str(row["timestamp"])),
            thread_id=str(row["thread_id"]),
            fault_input=str(row["fault_input"]),
            retrieved_chunk_ids=json.loads(str(row["retrieved_chunk_ids"])),
            draft=json.loads(str(row["draft"])),
            final_steps=json.loads(str(row["final_steps"])),
            decision=str(row["decision"]),
            reviewer_id=str(row["reviewer_id"]),
            reject_reason=row["reject_reason"],
            prev_hash=str(row["prev_hash"]),
            record_hash=str(row["record_hash"]),
        )

