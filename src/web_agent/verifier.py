from __future__ import annotations

from dataclasses import dataclass
from datetime import date
import json
import re
from typing import Any, Iterable

from .contracts import ActionReceipt, CoverageCertificate, TaskContract
from .evidence import EvidenceStore


@dataclass(frozen=True)
class VerificationResult:
    accepted: bool
    reasons: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {"accepted": self.accepted, "reasons": list(self.reasons)}


def _leaf_paths(value: Any, prefix: str = "$") -> list[str]:
    if isinstance(value, dict):
        if not value:
            return [prefix]
        result: list[str] = []
        for key, child in value.items():
            result.extend(_leaf_paths(child, f"{prefix}.{key}"))
        return result
    if isinstance(value, list):
        if not value:
            return [prefix]
        result = []
        for index, child in enumerate(value):
            result.extend(_leaf_paths(child, f"{prefix}[{index}]"))
        return result
    return [prefix]


def _leaf_values(value: Any, prefix: str = "$") -> list[tuple[str, Any]]:
    if isinstance(value, dict):
        if not value:
            return [(prefix, value)]
        return [
            leaf
            for key, child in value.items()
            for leaf in _leaf_values(child, f"{prefix}.{key}")
        ]
    if isinstance(value, list):
        if not value:
            return [(prefix, value)]
        return [
            leaf
            for index, child in enumerate(value)
            for leaf in _leaf_values(child, f"{prefix}[{index}]")
        ]
    return [(prefix, value)]


_DATE_TASK_PATTERN = re.compile(
    r"(?:日期|哪一天|几月几日)(?:\s*(?:是|为)?\s*(?:多少|什么))?\s*[?？]?$"
    r"|\b(?:what|which)\s+(?:is|was|were)\s+(?:the\s+)?date\b"
    r"|\bdate\s+of\b",
    re.IGNORECASE,
)
_NUMERIC_DATE_PATTERN = re.compile(
    r"(?<!\d)(\d{4})\s*[-/.年]\s*(\d{1,2})\s*[-/.月]\s*(\d{1,2})\s*日?(?!\d)"
)
_REVERSED_NUMERIC_DATE_PATTERN = re.compile(
    r"(?<!\d)(\d{1,2})\s*[-/.]\s*(\d{1,2})\s*[-/.]\s*(\d{4})(?!\d)"
)
_COMPACT_DATE_PATTERN = re.compile(r"(?<!\d)(\d{4})(\d{2})(\d{2})(?!\d)")
_DATETIME_PATTERN = re.compile(
    r"\d{4}\s*[-/.]\s*\d{1,2}\s*[-/.]\s*\d{1,2}"
    r"[T\s](?:[01]\d|2[0-3]):[0-5]\d(?::[0-5]\d(?:\.\d+)?)?"
    r"(?:Z|[+-]\d{2}:?\d{2})?",
    re.IGNORECASE,
)
_MONTHS = {
    "jan": 1,
    "january": 1,
    "feb": 2,
    "february": 2,
    "mar": 3,
    "march": 3,
    "apr": 4,
    "april": 4,
    "may": 5,
    "jun": 6,
    "june": 6,
    "jul": 7,
    "july": 7,
    "aug": 8,
    "august": 8,
    "sep": 9,
    "sept": 9,
    "september": 9,
    "oct": 10,
    "october": 10,
    "nov": 11,
    "november": 11,
    "dec": 12,
    "december": 12,
}
_MONTH_NAME = "|".join(sorted(_MONTHS, key=len, reverse=True))
_MONTH_FIRST_DATE_PATTERN = re.compile(
    rf"\b({_MONTH_NAME})\.?\s+(\d{{1,2}})(?:st|nd|rd|th)?\s*,?\s*(\d{{4}})\b",
    re.IGNORECASE,
)
_DAY_FIRST_DATE_PATTERN = re.compile(
    rf"\b(\d{{1,2}})(?:st|nd|rd|th)?\s+({_MONTH_NAME})\.?\s*,?\s*(\d{{4}})\b",
    re.IGNORECASE,
)
_ACCESS_BARRIER_EVIDENCE_PATTERN = re.compile(
    r"undeclared automated tool|request rate threshold exceeded"
    r"|access (?:was )?(?:blocked|denied|restricted)|request (?:was )?blocked"
    r"|too many requests|verify (?:that )?you are human|captcha|bot detection"
    r"|访问(?:被)?(?:阻止|拦截|拒绝|限制)|请求(?:被)?(?:阻止|拦截|拒绝)",
    re.IGNORECASE,
)
_ACCESS_BARRIER_TASK_PATTERN = re.compile(
    r"(?:是否|能否|可否)[^。；?？]{0,24}(?:访问|获取|打开|可用)"
    r"|(?:错误|报错|提示|状态|拦截|阻止|拒绝)(?:信息|内容|原因|是什么)?"
    r"|\b(?:is|was|whether)\b[^?.]{0,40}\b(?:accessible|available|reachable)\b"
    r"|\bcan\b[^?.]{0,20}\baccess\b"
    r"|\b(?:error|message|status|blocked|denied|restriction)\b",
    re.IGNORECASE,
)


def _valid_date(year: int, month: int, day: int) -> str | None:
    try:
        return date(year, month, day).isoformat()
    except ValueError:
        return None


def _normalized_dates(value: Any) -> set[str]:
    text = str(value)
    normalized: set[str] = set()
    for year, month, day in _NUMERIC_DATE_PATTERN.findall(text):
        if parsed := _valid_date(int(year), int(month), int(day)):
            normalized.add(parsed)
    for first, second, year in _REVERSED_NUMERIC_DATE_PATTERN.findall(text):
        left, right = int(first), int(second)
        month, day = (left, right) if left <= 12 else (right, left)
        if parsed := _valid_date(int(year), month, day):
            normalized.add(parsed)
    for year, month, day in _COMPACT_DATE_PATTERN.findall(text):
        if parsed := _valid_date(int(year), int(month), int(day)):
            normalized.add(parsed)
    for month, day, year in _MONTH_FIRST_DATE_PATTERN.findall(text):
        if parsed := _valid_date(int(year), _MONTHS[month.casefold()], int(day)):
            normalized.add(parsed)
    for day, month, year in _DAY_FIRST_DATE_PATTERN.findall(text):
        if parsed := _valid_date(int(year), _MONTHS[month.casefold()], int(day)):
            normalized.add(parsed)
    return normalized


def _is_empty_leaf(value: Any) -> bool:
    return value is None or value == [] or value == {} or (
        isinstance(value, str) and not value.strip()
    )


def _is_direct_date(value: Any) -> bool:
    if not isinstance(value, str):
        return False
    stripped = value.strip().rstrip("。")
    return any(
        pattern.fullmatch(stripped) is not None
        for pattern in (
            _NUMERIC_DATE_PATTERN,
            _REVERSED_NUMERIC_DATE_PATTERN,
            _COMPACT_DATE_PATTERN,
            _DATETIME_PATTERN,
            _MONTH_FIRST_DATE_PATTERN,
            _DAY_FIRST_DATE_PATTERN,
        )
    )


def _is_access_barrier_evidence(ref: Any) -> bool:
    text = json.dumps(
        {"summary": ref.summary, "payload": ref.payload},
        ensure_ascii=False,
        default=str,
    )
    return bool(_ACCESS_BARRIER_EVIDENCE_PATTERN.search(text))


def _task_fulfillment_reasons(
    contract: TaskContract,
    answer: Any,
    evidence_bindings: dict[str, list[str]],
    evidence_store: EvidenceStore,
) -> list[str]:
    reasons = [
        f"答案字段 {path} 为空"
        for path, leaf in _leaf_values(answer)
        if _is_empty_leaf(leaf)
    ]
    answer_text = json.dumps(answer, ensure_ascii=False, default=str)
    requires_date = bool(_DATE_TASK_PATTERN.search(contract.task))
    answer_dates = _normalized_dates(answer_text)
    if requires_date and not answer_dates:
        reasons.append("任务要求日期值，但 agent_answer 未包含可识别的完整日期")
    if requires_date:
        for path, leaf in _leaf_values(answer):
            if _normalized_dates(leaf) and not _is_direct_date(leaf):
                reasons.append(f"答案字段 {path} 必须是任务所求的直接日期值，不能夹带说明中的日期")

    if requires_date:
        for path, leaf in _leaf_values(answer):
            leaf_dates = _normalized_dates(leaf)
            if not leaf_dates:
                continue
            supported_dates: set[str] = set()
            for evidence_id in evidence_bindings.get(path, []):
                ref = evidence_store.get(evidence_id)
                if ref is None:
                    continue
                supported_dates.update(
                    _normalized_dates(
                        json.dumps(
                            {"summary": ref.summary, "payload": ref.payload},
                            ensure_ascii=False,
                            default=str,
                        )
                    )
                )
            missing = sorted(leaf_dates - supported_dates)
            if missing:
                reasons.append(f"答案字段 {path} 的日期 {missing} 未被对应绑定证据支持")

    if not _ACCESS_BARRIER_TASK_PATTERN.search(contract.task):
        for path, _ in _leaf_values(answer):
            refs = [
                ref
                for evidence_id in evidence_bindings.get(path, [])
                if (ref := evidence_store.get(evidence_id)) is not None
            ]
            if refs and all(_is_access_barrier_evidence(ref) for ref in refs):
                reasons.append(f"答案字段 {path} 的绑定证据仅证明访问受阻，不能证明任务所求值")
    return reasons


class CompletionVerifier:
    def verify(
        self,
        contract: TaskContract,
        answer: Any,
        evidence_ids: Iterable[str],
        evidence_bindings: dict[str, list[str]],
        coverage: CoverageCertificate | None,
        evidence_store: EvidenceStore,
        receipts: Iterable[ActionReceipt],
        visited_urls: Iterable[str],
    ) -> VerificationResult:
        reasons: list[str] = []
        evidence_ids = tuple(dict.fromkeys(evidence_ids))
        visited_urls = tuple(visited_urls)
        receipts = tuple(receipts)

        if answer is None or answer == "" or answer == [] or answer == {}:
            reasons.append("agent_answer 为空")
        reasons.extend(
            _task_fulfillment_reasons(contract, answer, evidence_bindings, evidence_store)
        )
        if not evidence_ids:
            reasons.append("未提供答案证据")

        unknown = [evidence_id for evidence_id in evidence_ids if not evidence_store.has(evidence_id)]
        if unknown:
            reasons.append(f"引用了不存在的证据: {unknown}")

        valid_urls = set(visited_urls)
        for evidence_id in evidence_ids:
            ref = evidence_store.get(evidence_id)
            if ref and ref.url not in valid_urls:
                reasons.append(f"证据 {evidence_id} 的 URL 未被访问")

        for path in _leaf_paths(answer):
            bound = evidence_bindings.get(path, [])
            if not bound:
                reasons.append(f"答案字段 {path} 未绑定证据")
                continue
            invalid = [evidence_id for evidence_id in bound if evidence_id not in evidence_ids]
            if invalid:
                reasons.append(f"答案字段 {path} 绑定了未提交的证据: {invalid}")

        if contract.requires_form_confirmation:
            confirmed = any(
                receipt.success and receipt.postconditions.get("confirmation") is True
                for receipt in receipts
            )
            if not confirmed:
                reasons.append("任务涉及提交型表单，但没有可验证的确认回执")

        if contract.requires_coverage:
            reasons.extend(self._verify_coverage(coverage, evidence_store))

        return VerificationResult(not reasons, tuple(reasons))

    @staticmethod
    def _verify_coverage(
        coverage: CoverageCertificate | None,
        evidence_store: EvidenceStore,
    ) -> list[str]:
        if coverage is None:
            return ["全量任务缺少 CoverageCertificate"]
        reasons: list[str] = []
        if coverage.strategy == "not_required" or coverage.terminal_reason == "not_required":
            reasons.append("全量任务的覆盖策略或终止原因无效")
        if coverage.unique_item_count < 0 or coverage.duplicate_item_count < 0:
            reasons.append("覆盖计数不能为负数")
        if coverage.pages_visited < 1:
            reasons.append("分页访问数必须至少为 1")
        if not re.fullmatch(r"[0-9a-f]{64}", coverage.item_fingerprint):
            reasons.append("去重条目指纹不是有效 SHA-256")
        if not coverage.terminal_evidence_id or not evidence_store.has(coverage.terminal_evidence_id):
            reasons.append("覆盖终止条件没有有效证据")
        if coverage.expected_total is not None and coverage.expected_total != coverage.unique_item_count:
            reasons.append(
                f"页面声明总数 {coverage.expected_total} 与去重数 {coverage.unique_item_count} 不一致"
            )
        return reasons
