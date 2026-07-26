from __future__ import annotations

from dataclasses import dataclass
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
