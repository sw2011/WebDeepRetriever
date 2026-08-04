from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Any, Callable

from .contracts import EvidenceRef, EvidenceSource
from .sanitization import redact_text, redact_value, sanitize_url


class EvidenceStore:
    """Append-only evidence store with deterministic task-local identifiers."""

    def __init__(self) -> None:
        self._items: dict[str, EvidenceRef] = {}
        self._lock = threading.Lock()
        self._counter = 0
        self._task_generation: int | None = None
        self._attempt_provider: Callable[[], int] | None = None
        self._publisher: Callable[[Callable[[], EvidenceRef]], EvidenceRef] | None = None

    def bind(
        self,
        task_generation: int,
        attempt_provider: Callable[[], int],
        publisher: Callable[[Callable[[], EvidenceRef]], EvidenceRef] | None = None,
    ) -> None:
        with self._lock:
            self._task_generation = task_generation
            self._attempt_provider = attempt_provider
            self._publisher = publisher

    def add(
        self,
        source: EvidenceSource,
        url: str,
        summary: str,
        payload: dict[str, Any],
    ) -> EvidenceRef:
        def commit() -> EvidenceRef:
            with self._lock:
                bound_payload = dict(payload)
                if self._task_generation is not None:
                    bound_payload["task_generation"] = self._task_generation
                    bound_payload["attempt"] = self._attempt_provider() if self._attempt_provider else 0
                self._counter += 1
                evidence_id = f"ev-{self._counter:05d}"
                ref = EvidenceRef(
                    evidence_id,
                    source,
                    sanitize_url(url),
                    redact_text(summary[:500]),
                    redact_value(bound_payload),
                )
                self._items[evidence_id] = ref
                return ref

        with self._lock:
            publisher = self._publisher
        if publisher is not None:
            return publisher(commit)
        return commit()

    def get(self, evidence_id: str) -> EvidenceRef | None:
        with self._lock:
            return self._items.get(evidence_id)

    def has(self, evidence_id: str) -> bool:
        with self._lock:
            return evidence_id in self._items

    def values(self) -> list[EvidenceRef]:
        with self._lock:
            return list(self._items.values())

    def discard_attempt(self, task_generation: int, attempt: int) -> None:
        with self._lock:
            self._items = {
                evidence_id: item
                for evidence_id, item in self._items.items()
                if not (
                    item.payload.get("task_generation") == task_generation
                    and item.payload.get("attempt") == attempt
                )
            }

    def to_list(self) -> list[dict[str, Any]]:
        return [item.to_dict() for item in self.values()]

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(json.dumps(self.to_list(), ensure_ascii=False, indent=2), encoding="utf-8")
        temporary.replace(path)
