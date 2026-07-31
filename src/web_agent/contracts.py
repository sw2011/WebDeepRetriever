from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Literal

from .sanitization import redact_text, redact_value, sanitize_error_text, sanitize_url


EvidenceSource = Literal["dom", "accessibility", "network", "document", "visual", "receipt"]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass(frozen=True)
class EvidenceRef:
    evidence_id: str
    source: EvidenceSource
    url: str
    summary: str
    payload: dict[str, Any]
    created_at: str = field(default_factory=utc_now)

    def to_dict(self) -> dict[str, Any]:
        return {
            "evidence_id": self.evidence_id,
            "source": self.source,
            "url": sanitize_url(self.url),
            "summary": redact_text(self.summary),
            "payload": redact_value(self.payload),
            "created_at": self.created_at,
        }


@dataclass(frozen=True)
class ActionReceipt:
    action_id: str
    action: str
    success: bool
    before_url: str
    after_url: str
    before_dom_hash: str
    after_dom_hash: str
    postconditions: dict[str, Any]
    evidence_ids: tuple[str, ...] = ()
    error: str | None = None
    stale_bid: bool = False
    created_at: str = field(default_factory=utc_now)

    @property
    def changed(self) -> bool:
        return self.before_url != self.after_url or self.before_dom_hash != self.after_dom_hash

    def to_dict(self) -> dict[str, Any]:
        return {
            "action_id": self.action_id,
            "action": self.action,
            "success": self.success,
            "before_url": sanitize_url(self.before_url),
            "after_url": sanitize_url(self.after_url),
            "before_dom_hash": self.before_dom_hash,
            "after_dom_hash": self.after_dom_hash,
            "changed": self.changed,
            "postconditions": redact_value(self.postconditions),
            "evidence_ids": list(self.evidence_ids),
            "error": sanitize_error_text(self.error) if self.error else None,
            "stale_bid": self.stale_bid,
            "created_at": self.created_at,
        }


@dataclass(frozen=True)
class CoverageCertificate:
    strategy: Literal["pagination", "cursor", "virtual_list", "declared_total", "not_required"]
    unique_item_count: int
    duplicate_item_count: int = 0
    pages_visited: int = 1
    expected_total: int | None = None
    terminal_reason: Literal[
        "next_disabled",
        "next_absent",
        "cursor_exhausted",
        "no_new_items",
        "total_matched",
        "not_required",
    ] = "not_required"
    terminal_evidence_id: str | None = None
    item_fingerprint: str = ""

    @classmethod
    def from_dict(cls, value: dict[str, Any] | None) -> CoverageCertificate | None:
        if value is None:
            return None
        allowed = {
            "strategy",
            "unique_item_count",
            "duplicate_item_count",
            "pages_visited",
            "expected_total",
            "terminal_reason",
            "terminal_evidence_id",
            "item_fingerprint",
        }
        unknown = set(value) - allowed
        if unknown:
            raise ValueError(f"CoverageCertificate 包含未知字段: {sorted(unknown)}")
        return cls(**value)

    def to_dict(self) -> dict[str, Any]:
        return {
            "strategy": self.strategy,
            "unique_item_count": self.unique_item_count,
            "duplicate_item_count": self.duplicate_item_count,
            "pages_visited": self.pages_visited,
            "expected_total": self.expected_total,
            "terminal_reason": self.terminal_reason,
            "terminal_evidence_id": self.terminal_evidence_id,
            "item_fingerprint": self.item_fingerprint,
        }


_COVERAGE_PATTERNS = re.compile(
    r"(?:全部|所有|完整|列出|分别|每(?:个|项)|全[部量]|top\s*(?!\d+\s*%)\d+|前\s*(?:\d+|[一二三四五六七八九十]+)(?:个|项)?|哪\s*(?:\d+|[一二三四五六七八九十]+)\s*个|排名|共有多少|总数|哪些|\bhow\s+many\b|\bwhich\b[^?.]{0,80}\b(?:are|were|have|contain|include)\b|\blist\s+all\b|\btotal\s+number\b|\bevery\b)",
    re.IGNORECASE,
)
_FORM_PATTERNS = re.compile(
    r"(?:提交|申请|预约|注册|登录|下单|购买|发送|发布|保存|创建|上传|确认表单)",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class TaskContract:
    task_idx: int | str
    task_id: str
    website: str
    task: str
    requires_coverage: bool
    requires_form_confirmation: bool
    max_steps: int = 100

    @classmethod
    def from_item(cls, item: dict[str, Any], max_steps: int = 100) -> TaskContract:
        task = str(item.get("task", "")).strip()
        website = str(item.get("website", "")).strip()
        if website and not website.startswith(("http://", "https://")):
            website = "https://" + website
        return cls(
            task_idx=item.get("task_idx", item.get("task_index", "")),
            task_id=str(item.get("task_id", item.get("task_idx", item.get("task_index", "")))),
            website=website,
            task=task,
            requires_coverage=bool(_COVERAGE_PATTERNS.search(task)),
            requires_form_confirmation=bool(_FORM_PATTERNS.search(task)),
            max_steps=min(max(int(max_steps), 1), 100),
        )
