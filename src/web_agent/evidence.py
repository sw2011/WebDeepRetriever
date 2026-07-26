from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Any

from .contracts import EvidenceRef, EvidenceSource


class EvidenceStore:
    """Append-only evidence store with deterministic task-local identifiers."""

    def __init__(self) -> None:
        self._items: dict[str, EvidenceRef] = {}
        self._lock = threading.Lock()
        self._counter = 0

    def add(
        self,
        source: EvidenceSource,
        url: str,
        summary: str,
        payload: dict[str, Any],
    ) -> EvidenceRef:
        with self._lock:
            self._counter += 1
            evidence_id = f"ev-{self._counter:05d}"
            ref = EvidenceRef(evidence_id, source, url, summary[:500], payload)
            self._items[evidence_id] = ref
            return ref

    def get(self, evidence_id: str) -> EvidenceRef | None:
        return self._items.get(evidence_id)

    def has(self, evidence_id: str) -> bool:
        return evidence_id in self._items

    def values(self) -> list[EvidenceRef]:
        return list(self._items.values())

    def to_list(self) -> list[dict[str, Any]]:
        return [item.to_dict() for item in self.values()]

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(json.dumps(self.to_list(), ensure_ascii=False, indent=2), encoding="utf-8")
        temporary.replace(path)

