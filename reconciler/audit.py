"""Append-only audit log of every state transition."""
from __future__ import annotations
from datetime import datetime, timezone

from .store import Store


class Audit:
    def __init__(self, store: Store):
        self.store = store

    def record(self, entity_type: str, entity_id: str, frm: str, to: str, reason: str = "") -> None:
        self.store.append_audit({
            "ts": datetime.now(timezone.utc).isoformat(),
            "entity_type": entity_type,   # "job" | "node"
            "entity_id": entity_id,
            "from": frm,
            "to": to,
            "reason": reason,
        })
