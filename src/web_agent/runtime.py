from __future__ import annotations

import asyncio
import base64
import hashlib
import json
import re
import secrets
import time
from decimal import Decimal, InvalidOperation
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Literal

from agents import (
    Agent,
    FunctionToolResult,
    MaxTurnsExceeded,
    ModelSettings,
    OpenAIChatCompletionsModel,
    RunConfig,
    RunContextWrapper,
    Runner,
    ToolExecutionConfig,
    ToolsToFinalOutputResult,
    function_tool,
)
from agents.lifecycle import RunHooksBase
from agents.run_config import CallModelData, ModelInputData
from openai import AsyncOpenAI
from pydantic import BaseModel, ConfigDict, Field

from .browser_actor import (
    ActorCallDeadlineExceeded,
    BrowserActor,
    BrowserActorPoisonedError,
    canonical_url,
)
from .contracts import ActionReceipt, CoverageCertificate, TaskContract
from .evidence import EvidenceStore
from .model_profiles import build_model_settings, vision_request_options
from .sanitization import redact_value, sanitize_exception, sanitize_url
from .token_control import (
    MAX_SERIALIZED_CONTEXT_BYTES,
    SharedTPMLimiter,
    TaskUsageStats,
    ThrottledModel,
    estimate_input_tokens,
    model_input_metrics,
)
from .verifier import CompletionVerifier


SYSTEM_INSTRUCTIONS = """你是 WebRetriever Protocol III 的网页任务执行 Agent。
你只能通过给定的 Playwright 工具观察和操作比赛方浏览器，禁止搜索引擎、AnySearch、主动 HTTP/API 请求和坐标点击。

执行规则：
1. 先 observe。所有可定位交互都使用最新观察中的 bid；陈旧 bid 失败后必须重新 observe。
2. 原生 input/select/checkbox 分别使用 fill/select/set_checked。自定义下拉先 click，再 observe 新出现的 listbox/option。
3. SPA、弹窗、多标签、iframe、Shadow DOM、分页和虚拟列表均通过工具处理。不要猜测页面状态。
4. 答案必须来自访问过 URL 的 DOM、页面触发的 XHR/Fetch、下载文档或获准的局部视觉证据。截图默认不会发送给你。
5. 只有结构化页面、ARIA、网络响应和文档文本均无法表达 Canvas、图片、图表或扫描 PDF 时，才调用 visual_inspect。
6. 对“全部、完整、列出、前 N、总数、排名”等任务，必须耗尽分页/游标/虚拟列表，先用 record_coverage 从真实证据签发 coverage_id，再在 finish 引用该短标识；不要复制计数或指纹。
7. finish 的 evidence_bindings 必须覆盖答案字段：标量答案可省略并安全绑定到 `$`；JSON 数组/对象必须按 `$[0]`、`$.items[0]` 等叶子路径逐项绑定。所有 evidence_ids 都必须真实存在。
8. 不得直接输出最终文本。每一轮必须调用且只调用一个工具，最终只能调用 finish。
9. 工具 Schema 中标记 required 但允许 null 的参数必须显式传 null，不得省略或添加未知字段。
10. observe 返回 unchanged=true 时不得继续重复 observe/tabs(list)/wait；应使用现有 bid 操作、按需 extract，或在证据不足时 finish 失败原因。
11. 历史摘要中的结构化证据如需尾部或完整分块，使用 recall_evidence 按 evidence_id 和 offset 回读，不要重复抓取页面。
12. 同页需要多种文本/列表/表格/链接时优先一次 extract_many（最多 8 项）；cache_hit=true 表示没有新信息，不要重复调用。
13. finish 前只回答被问值或列表；数值、单位、前 N/全部条目数必须与绑定证据和 coverage 摘要一致，不得用解释性长答案掩盖缺项。
14. 工具返回 terminal_uncertain=true 或 ACTOR_POISONED 时，动作终态不确定且浏览器不可继续；不得重试该动作。
"""


class ExtractRequestInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    kind: Literal["text", "links", "table", "list"]
    bid: str | None = None
    limit: int = Field(default=1000, ge=1, le=5000)


class EvidenceBindingInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    path: str
    evidence_ids: list[str]


class BoundedToolOutputFilter:
    """Replace tool-call history with a bounded, recallable working-memory checkpoint."""

    _CONTENT_TOOLS = {"extract", "network", "document", "recall_evidence"}

    def __init__(
        self,
        keep_recent_outputs: int = 4,
        old_output_chars: int = 800,
        max_current_output_chars: int = 24_000,
        max_total_output_chars: int = 60_000,
        max_context_bytes: int = 60_000,
    ) -> None:
        self.keep_recent_outputs = keep_recent_outputs
        self.old_output_chars = old_output_chars
        self.max_current_output_chars = max_current_output_chars
        self.max_total_output_chars = max_total_output_chars
        self.max_context_bytes = max_context_bytes

    @staticmethod
    def _text(output: Any) -> str:
        return output if isinstance(output, str) else json.dumps(output, ensure_ascii=False, default=str)

    @staticmethod
    def _summary(tool: str, text: str, preview_chars: int) -> str:
        try:
            payload = json.loads(text)
        except (json.JSONDecodeError, TypeError):
            payload = None
        summary: dict[str, Any] = {
            "compacted_tool_output": True,
            "tool": tool,
            "original_chars": len(text),
        }
        if isinstance(payload, dict):
            for key in (
                "ok",
                "accepted",
                "success",
                "action_id",
                "action",
                "changed",
                "url",
                "title",
                "dom_hash",
                "evidence_id",
                "evidence_ids",
                "coverage_evidence_id",
                "element_count",
                "total_element_count",
                "unchanged",
                "stale_bid",
                "error",
                "reasons",
                "coverage_id",
                "semantic_page_fingerprint",
                "progress_reason",
                "cache_hit",
            ):
                if key in payload:
                    summary[key] = payload[key]
            content = next(
                (
                    payload[key]
                    for key in ("data", "records", "text", "content", "preview")
                    if payload.get(key) is not None
                ),
                None,
            )
            if content is not None and preview_chars:
                encoded = (
                    content
                    if isinstance(content, str)
                    else json.dumps(content, ensure_ascii=False, separators=(",", ":"), default=str)
                )
                summary["content_chars"] = len(encoded)
                summary["content_preview"] = encoded[:preview_chars]
            if tool in BoundedToolOutputFilter._CONTENT_TOOLS and payload.get("evidence_id"):
                summary["recall_instruction"] = (
                    "需要此证据的其余内容时调用 recall_evidence，并从 offset=0 开始按 next_offset 续读"
                )
        elif preview_chars:
            summary["content_preview"] = text[:preview_chars]
        return _json(summary, limit=max(1_000, preview_chars + 800))

    @staticmethod
    def _runtime_context(data: CallModelData[Any]) -> TaskRuntimeContext | None:
        value = getattr(data, "context", None)
        if isinstance(value, TaskRuntimeContext):
            return value
        nested = getattr(value, "context", None)
        return nested if isinstance(nested, TaskRuntimeContext) else None

    def _history_checkpoint(self, items: list[Any]) -> dict[str, Any]:
        call_names = {
            str(item.get("call_id")): str(item.get("name", "unknown"))
            for item in items
            if isinstance(item, dict) and item.get("type") == "function_call"
        }
        results: list[dict[str, Any]] = []
        evidence: dict[str, dict[str, Any]] = {}
        latest_bids: list[dict[str, Any]] = []
        for item in items:
            if not isinstance(item, dict) or item.get("type") != "function_call_output":
                continue
            tool = call_names.get(str(item.get("call_id")), "unknown")
            text = self._text(item.get("output"))
            summary = json.loads(self._summary(tool, text, min(2_000, self.old_output_chars * 4)))
            results.append(summary)
            try:
                payload = json.loads(text)
            except (json.JSONDecodeError, TypeError):
                payload = {}
            for evidence_id in payload.get("evidence_ids", []) or []:
                evidence[str(evidence_id)] = {"evidence_id": str(evidence_id), "recallable": True}
            if payload.get("evidence_id"):
                evidence[str(payload["evidence_id"])] = {
                    "evidence_id": str(payload["evidence_id"]),
                    "summary": payload.get("summary"),
                    "recallable": tool in self._CONTENT_TOOLS,
                }
            if tool == "observe" and isinstance(payload.get("elements"), list):
                if payload.get("unchanged") is not True or not latest_bids:
                    latest_bids = payload["elements"][:120]
        return {
            "evidence_ledger": list(evidence.values())[-80:],
            "latest_bid_catalog": latest_bids,
            "recent_tool_results": results[-self.keep_recent_outputs :],
        }

    def _context_checkpoint(self, context: TaskRuntimeContext) -> dict[str, Any]:
        evidence_ledger = []
        relevant_ids = set(context.last_finish_evidence_ids)
        values = context.evidence_store.values()
        if context.finish_phase and relevant_ids:
            recent_ids = {value.evidence_id for value in values[-20:]}
            values = [value for value in values if value.evidence_id in relevant_ids | recent_ids]
        for evidence in values[-80:]:
            encoded = json.dumps(evidence.payload, sort_keys=True, ensure_ascii=False, separators=(",", ":"), default=str)
            evidence_ledger.append(
                {
                    "evidence_id": evidence.evidence_id,
                    "source": evidence.source,
                    "url": evidence.url,
                    "summary": evidence.summary,
                    "content_hash": hashlib.sha256(encoded.encode("utf-8")).hexdigest()[:16],
                    "recallable": evidence.source in {"dom", "network", "document"},
                }
            )
        finish_phase = context.finish_phase
        return {
            "task": context.contract.task,
            "website": context.contract.website,
            "current_url": context.current_url,
            "semantic_page_fingerprint": context.current_semantic_fingerprint,
            "latest_bid_catalog": (
                []
                if finish_phase
                else list(context.latest_model_elements)
            ),
            "evidence_ledger": evidence_ledger,
            "coverage_id": context.latest_coverage_id,
            "recent_tool_results": context.tool_outcomes[-self.keep_recent_outputs :],
            "last_verifier_reasons": context.last_finish_reasons,
            "finish_phase": finish_phase,
            "budget": {
                "used": context.tool_steps,
                "available": context.adaptive_step_budget,
                "hard_limit": context.hard_step_limit,
            },
        }

    def __call__(self, data: CallModelData[Any]) -> ModelInputData:
        model_data = data.model_data
        items = list(model_data.input)
        context = self._runtime_context(data)
        checkpoint = self._context_checkpoint(context) if context else self._history_checkpoint(items)
        original = next(
            (
                item
                for item in items
                if not (isinstance(item, dict) and item.get("type") in {"function_call", "function_call_output"})
            ),
            {"role": "user", "content": "继续执行原任务"},
        )

        def render(value: dict[str, Any]) -> list[Any]:
            return [
                original,
                {
                    "role": "user",
                    "content": "WORKING_MEMORY_CHECKPOINT\n"
                    + json.dumps(redact_value(value), ensure_ascii=False, separators=(",", ":")),
                },
            ]

        bounded = render(checkpoint)

        def size() -> int:
            return len(json.dumps(bounded, ensure_ascii=False, separators=(",", ":")).encode("utf-8"))

        while size() > self.max_context_bytes:
            recent = checkpoint.get("recent_tool_results", [])
            bids = checkpoint.get("latest_bid_catalog", [])
            evidence = checkpoint.get("evidence_ledger", [])
            if recent:
                checkpoint["recent_tool_results"] = recent[1:]
            elif any(isinstance(item, dict) and item.get("options") for item in bids):
                checkpoint["latest_bid_catalog"] = [
                    ({key: value for key, value in item.items() if key != "options"} | {"options_omitted": True})
                    if isinstance(item, dict) and item.get("options")
                    else item
                    for item in bids
                ]
            elif bids:
                checkpoint["latest_bid_catalog"] = bids[: len(bids) // 2]
            elif len(evidence) > 1:
                checkpoint["evidence_ledger"] = evidence[-max(1, len(evidence) // 2) :]
            elif evidence and any(
                isinstance(item, dict) and (item.get("summary") or item.get("url")) for item in evidence
            ):
                checkpoint["evidence_ledger"] = [
                    {
                        key: item[key]
                        for key in ("evidence_id", "source", "content_hash", "recallable")
                        if key in item
                    }
                    for item in evidence
                    if isinstance(item, dict)
                ]
            elif checkpoint.get("last_verifier_reasons"):
                checkpoint["last_verifier_reasons"] = [
                    str(value)[:500] for value in checkpoint["last_verifier_reasons"][-2:]
                ]
                if all(len(str(value)) <= 500 for value in checkpoint["last_verifier_reasons"]):
                    checkpoint["last_verifier_reasons"] = []
            elif len(str(checkpoint.get("task", ""))) > 8_000:
                checkpoint["task"] = str(checkpoint["task"])[:8_000]
            else:
                checkpoint["checkpoint_truncated"] = True
                break
            bounded = render(checkpoint)
        if size() > self.max_context_bytes:
            minimal = {
                "task": str(checkpoint.get("task", ""))[:8_000],
                "website": checkpoint.get("website"),
                "current_url": checkpoint.get("current_url"),
                "semantic_page_fingerprint": checkpoint.get("semantic_page_fingerprint"),
                "coverage_id": checkpoint.get("coverage_id"),
                "checkpoint_truncated": True,
            }
            bounded = render(minimal)
        return ModelInputData(input=bounded, instructions=model_data.instructions)


@dataclass
class TaskRuntimeContext:
    actor: BrowserActor
    contract: TaskContract
    evidence_store: EvidenceStore
    verifier: CompletionVerifier
    vision_client: AsyncOpenAI
    vision_model: str
    receipts: list[ActionReceipt] = field(default_factory=list)
    actions: list[dict[str, Any]] = field(default_factory=list)
    thoughts: list[str] = field(default_factory=list)
    visited_urls: list[str] = field(default_factory=list)
    latest_elements: dict[str, dict[str, Any]] = field(default_factory=dict)
    latest_model_elements: list[dict[str, Any]] = field(default_factory=list)
    latest_model_element_state: tuple[str, str] | None = None
    downloaded_paths: set[str] = field(default_factory=set)
    scanned_document_paths: set[str] = field(default_factory=set)
    final_answer: str | None = None
    final_evidence_ids: list[str] = field(default_factory=list)
    final_bindings: dict[str, list[str]] = field(default_factory=dict)
    final_coverage: CoverageCertificate | None = None
    finish_accepted: bool = False
    tool_steps: int = 0
    coverage_records: dict[str, CoverageCertificate] = field(default_factory=dict)
    coverage_evidence_ids: dict[str, str] = field(default_factory=dict)
    usage_stats: TaskUsageStats | None = None
    rate_limiter: SharedTPMLimiter | None = None
    last_observation_state: tuple[str, str] | None = None
    seen_page_states: set[tuple[str, str]] = field(default_factory=set)
    seen_urls: set[str] = field(default_factory=set)
    seen_content_hashes: set[str] = field(default_factory=set)
    recalled_chunks: set[tuple[str, int, int | None, str]] = field(default_factory=set)
    extract_cache: dict[str, dict[str, Any]] = field(default_factory=dict)
    tool_outcomes: list[dict[str, Any]] = field(default_factory=list)
    current_url: str | None = None
    current_semantic_fingerprint: str | None = None
    latest_coverage_id: str | None = None
    last_finish_reasons: list[str] = field(default_factory=list)
    last_finish_evidence_ids: list[str] = field(default_factory=list)
    finish_phase: bool = False
    progress_credit: int = 0
    base_step_budget: int = 30
    absolute_step_limit: int = 60
    no_progress_streak: int = 0
    no_progress_limit: int = 8
    loop_detected: bool = False
    loop_reason: str | None = None
    rejected_finish_counts: dict[str, int] = field(default_factory=dict)
    repeated_no_info: dict[str, int] = field(default_factory=dict)
    repeat_count: int = 0
    cycle_count: int = 0
    terminal_browser_error: str | None = None
    progress_callback: Callable[[str], None] | None = None

    @property
    def hard_step_limit(self) -> int:
        return min(self.contract.max_steps, self.absolute_step_limit)

    @property
    def adaptive_step_budget(self) -> int:
        return min(self.hard_step_limit, self.base_step_budget + self.progress_credit)

    @staticmethod
    def _signature(name: str, arguments: dict[str, Any]) -> str:
        normalized = dict(arguments)
        if name == "tabs" and normalized.get("url"):
            try:
                normalized["url"] = canonical_url(str(normalized["url"]))
            except ValueError:
                pass
        encoded = json.dumps(normalized, sort_keys=True, ensure_ascii=False, separators=(",", ":"), default=str)
        return f"{name}:{hashlib.sha256(encoded.encode('utf-8')).hexdigest()[:16]}"

    def record_call(self, name: str, arguments: dict[str, Any]) -> None:
        if self.tool_steps >= self.hard_step_limit:
            raise RuntimeError(f"STEP_LIMIT: 工具调用不得超过 {self.hard_step_limit} 步")
        self.tool_steps += 1
        self.actions.append(
            {
                "step": self.tool_steps,
                "task_generation": int(getattr(self.actor, "task_generation", 0)),
                "attempt": self.tool_steps,
                "tool": name,
                "arguments": redact_value(arguments),
                "signature": self._signature(name, arguments),
            }
        )

    def record_receipt(self, result: dict[str, Any], *, track_outcome: bool = True) -> None:
        receipt = ActionReceipt(
            action_id=result["action_id"],
            action=result["action"],
            success=result["success"],
            before_url=result["before_url"],
            after_url=result["after_url"],
            before_dom_hash=result["before_dom_hash"],
            after_dom_hash=result["after_dom_hash"],
            postconditions=result["postconditions"],
            evidence_ids=tuple(result.get("evidence_ids", [])),
            error=result.get("error"),
            stale_bid=result.get("stale_bid", False),
            created_at=result.get("created_at", ""),
            task_generation=int(result.get("task_generation", 0)),
            attempt=int(result.get("attempt", 0)),
        )
        self.receipts.append(receipt)
        if result.get("after_url"):
            self.visited_urls.append(result["after_url"])
        if track_outcome:
            self.record_tool_outcome(receipt.action, result.get("postconditions", {}), result)

    @staticmethod
    def _content_hash(result: dict[str, Any]) -> str | None:
        if result.get("content_hash"):
            return str(result["content_hash"])
        key = next((name for name in ("data", "records", "text", "content", "analysis") if name in result), None)
        if key is None:
            return None
        if result[key] in (None, "", [], {}):
            return None
        encoded = json.dumps(result[key], sort_keys=True, ensure_ascii=False, separators=(",", ":"), default=str)
        return hashlib.sha256(encoded.encode("utf-8")).hexdigest()[:20]

    def record_tool_outcome(
        self,
        name: str,
        arguments: dict[str, Any],
        result: dict[str, Any],
        *,
        cache_hit: bool = False,
        tool_latency_ms: float = 0.0,
        browser_latency_ms: float = 0.0,
    ) -> dict[str, Any]:
        result_generation = result.get("task_generation")
        result_attempt = result.get("attempt")
        task_generation = (
            int(result_generation)
            if result_generation is not None
            else int(getattr(self.actor, "task_generation", 0))
        )
        attempt = int(result_attempt) if result_attempt is not None else self.tool_steps
        if self.actions and self.actions[-1].get("step") == self.tool_steps:
            self.actions[-1]["task_generation"] = task_generation
            self.actions[-1]["attempt"] = attempt
        signature = self._signature(name, arguments)
        state_before = (self.current_url, self.current_semantic_fingerprint)
        reasons: list[str] = []
        success = bool(result.get("ok", result.get("success", result.get("accepted", False))))

        urls: list[str] = []
        current_urls: list[str] = []
        for key in ("url", "after_url"):
            if result.get(key):
                value = str(result[key])
                urls.append(value)
                current_urls.append(value)
        for tab in result.get("tabs", []):
            if not isinstance(tab, dict) or not tab.get("url"):
                continue
            value = str(tab["url"])
            urls.append(value)
            if tab.get("active") is True:
                current_urls.append(value)
        for url in urls:
            try:
                normalized = canonical_url(url)
            except ValueError:
                normalized = url
            if normalized not in self.seen_urls:
                self.seen_urls.add(normalized)
                reasons.append("new_url")
            self.visited_urls.append(url)
        if current_urls:
            self.current_url = current_urls[-1]

        postconditions = result.get("postconditions", {}) if isinstance(result.get("postconditions"), dict) else {}
        semantic = result.get("semantic_page_fingerprint") or postconditions.get(
            "after_semantic_page_fingerprint"
        )
        if semantic:
            state = (canonical_url(self.current_url or self.contract.website), str(semantic))
            if state not in self.seen_page_states:
                self.seen_page_states.add(state)
                reasons.append("new_semantic_state")
            self.last_observation_state = state
            self.current_semantic_fingerprint = str(semantic)
            if self.latest_model_element_state != state:
                self.latest_model_elements = []
                self.latest_model_element_state = None

        if success and name == "scroll" and postconditions.get("after") != postconditions.get("before"):
            reasons.append("scroll_position_changed")
        if success and name in {"fill", "select", "set_checked"} and postconditions.get("value_changed") is True:
            reasons.append("form_value_changed")
        if success and name == "download":
            path = postconditions.get("download", {}).get("path") if isinstance(postconditions.get("download"), dict) else None
            if path and path not in self.downloaded_paths:
                self.downloaded_paths.add(path)
                reasons.append("new_download")
        if success and postconditions.get("confirmation") is True:
            reasons.append("submission_confirmed")

        content_hash = self._content_hash(result)
        if name == "recall_evidence" and success and content_hash:
            chunk = (
                str(result.get("evidence_id", "")),
                int(result.get("offset", 0)),
                result.get("next_offset"),
                content_hash,
            )
            if chunk not in self.recalled_chunks:
                self.recalled_chunks.add(chunk)
                reasons.append("new_evidence_chunk")
        elif name in {"extract", "extract_many", "network", "document", "visual_inspect", "visual_document"}:
            hashes = [content_hash] if content_hash else []
            hashes.extend(
                str(item.get("content_hash"))
                for item in result.get("results", [])
                if isinstance(item, dict) and item.get("content_hash")
            )
            new_hashes = [value for value in hashes if value not in self.seen_content_hashes]
            if new_hashes:
                self.seen_content_hashes.update(new_hashes)
                reasons.append("new_content_hash")

        coverage_id = result.get("coverage_id")
        if success and coverage_id and coverage_id in self.coverage_records and coverage_id != self.latest_coverage_id:
            self.latest_coverage_id = str(coverage_id)
            self.finish_phase = True
            reasons.append("valid_coverage_handle")
        if name == "finish":
            self.finish_phase = True
            self.last_finish_reasons = [str(value) for value in result.get("reasons", [])]
            self.last_finish_evidence_ids = list(
                dict.fromkeys(str(value) for value in arguments.get("evidence_ids", []))
            )
            if result.get("accepted") is True:
                reasons.append("finish_accepted")

        progressed = bool(reasons) and not cache_hit
        self.note_progress(progressed, reasons[0] if reasons else f"{name}:no_new_information")
        content_key = next(
            (key for key in ("data", "records", "text", "content", "analysis") if key in result),
            None,
        )
        content_preview: str | None = None
        if content_key:
            raw_content = result[content_key]
            content_preview = (
                raw_content
                if isinstance(raw_content, str)
                else json.dumps(raw_content, ensure_ascii=False, separators=(",", ":"), default=str)
            )[:1_200]
        outcome = {
            "step": self.tool_steps,
            "task_generation": task_generation,
            "attempt": attempt,
            "tool": name,
            "signature": signature,
            "success": success,
            "progressed": progressed,
            "progress_reason": reasons or ["no_new_information"],
            "semantic_page_fingerprint": self.current_semantic_fingerprint,
            "cache_hit": cache_hit,
            "repeat_count": 0,
            "cycle_count": 0,
            "tool_latency_ms": round(tool_latency_ms, 3),
            "browser_latency_ms": round(browser_latency_ms, 3),
            "result": redact_value(
                {
                    key: result[key]
                    for key in (
                        "ok",
                        "success",
                        "accepted",
                        "error",
                        "reasons",
                        "evidence_id",
                        "evidence_ids",
                        "coverage_id",
                        "url",
                        "next_offset",
                    )
                    if key in result
                }
                | (
                    {"post_observation": postconditions["post_observation"]}
                    if isinstance(postconditions.get("post_observation"), dict)
                    else {}
                )
                | ({"content_preview": content_preview} if content_preview is not None else {})
            ),
            "state_before": state_before,
            "state_after": (self.current_url, self.current_semantic_fingerprint),
        }
        self.tool_outcomes.append(outcome)

        if name == "finish" and result.get("accepted") is not True:
            count = self.rejected_finish_counts.get(signature, 0) + 1
            self.rejected_finish_counts[signature] = count
            outcome["repeat_count"] = count
            self.repeat_count = max(self.repeat_count, count)
            if count >= 2:
                self._trip_loop("REPEATED_FINISH", "相同 finish 已连续被拒绝 2 次")
        guarded_repeat = name in {"extract", "recall_evidence"} or (
            name == "tabs" and arguments.get("action") == "new"
        )
        if guarded_repeat and not progressed:
            count = self.repeated_no_info.get(signature, 0) + 1
            self.repeated_no_info[signature] = count
            outcome["repeat_count"] = max(int(outcome["repeat_count"]), count)
            self.repeat_count = max(self.repeat_count, count)
            if count >= 2:
                self._trip_loop("REPEATED_TOOL", f"相同 {name} 调用未产生新信息")
        elif progressed:
            self.repeated_no_info.pop(signature, None)

        self._detect_cycle()
        if self.actions and self.actions[-1].get("step") == self.tool_steps:
            self.actions[-1].update(
                {
                    "progress_reason": outcome["progress_reason"],
                    "cache_hit": cache_hit,
                    "repeat_count": outcome["repeat_count"],
                    "cycle_count": outcome["cycle_count"],
                }
            )
        return outcome

    def _detect_cycle(self) -> None:
        for period in range(1, 5):
            if len(self.tool_outcomes) < period * 2:
                continue
            window = self.tool_outcomes[-period * 2 :]
            if any(item["progressed"] for item in window):
                continue
            first = [(item["signature"], item["state_after"]) for item in window[:period]]
            second = [(item["signature"], item["state_after"]) for item in window[period:]]
            if first == second:
                self.cycle_count += 1
                self.tool_outcomes[-1]["cycle_count"] = self.cycle_count
                self._trip_loop("STATE_ACTION_CYCLE", f"检测到周期 {period} 的状态/动作循环")
                return

    def _trip_loop(self, code: str, reason: str) -> None:
        if self.loop_detected:
            return
        self.loop_detected = True
        self.loop_reason = f"NO_PROGRESS_LOOP: {code}: {reason}"
        self.thoughts.append(f"loop_guard: {reason}")

    def note_page_state(self, url: str, semantic_fingerprint: str) -> bool:
        state = (canonical_url(url), semantic_fingerprint)
        unchanged = state == self.last_observation_state
        is_new = state not in self.seen_page_states
        self.seen_page_states.add(state)
        self.last_observation_state = state
        self.current_url = url
        self.current_semantic_fingerprint = semantic_fingerprint
        self.seen_urls.add(canonical_url(url))
        self.note_progress(is_new, "observe")
        return unchanged

    def note_progress(self, progressed: bool, source: str) -> None:
        if progressed:
            self.no_progress_streak = 0
            self.progress_credit = min(self.progress_credit + 1, self.absolute_step_limit - self.base_step_budget)
            return
        self.no_progress_streak += 1
        if self.no_progress_streak >= self.no_progress_limit and not self.loop_detected:
            self._trip_loop(
                "NO_NEW_INFORMATION",
                f"连续 {self.no_progress_streak} 次工具调用没有可证明的新信息（最后来源 {source}）",
            )

    def assert_model_budget(self) -> None:
        if self.tool_steps >= self.adaptive_step_budget and not self.finish_accepted:
            self._trip_loop(
                "ADAPTIVE_BUDGET",
                f"已使用 {self.tool_steps} 次模型请求，当前可证明进展额度为 {self.adaptive_step_budget}",
            )

    def telemetry_snapshot(self) -> dict[str, Any]:
        latest = self.tool_outcomes[-1] if self.tool_outcomes else {}
        return {
            "last_tool": latest.get("tool"),
            "semantic_state": self.current_semantic_fingerprint,
            "progress_reason": latest.get("progress_reason", []),
            "repeat_count": int(latest.get("repeat_count", 0)),
            "cycle_count": int(latest.get("cycle_count", 0)),
            "cache_hit": bool(latest.get("cache_hit", False)),
            "tool_latency_ms": float(latest.get("tool_latency_ms", 0.0)),
            "browser_latency_ms": float(latest.get("browser_latency_ms", 0.0)),
        }


class NoProgressLoopError(RuntimeError):
    pass


class TerminalBrowserError(RuntimeError):
    pass


def _actor_poisoned(context: TaskRuntimeContext) -> bool:
    return bool(getattr(context.actor, "poisoned", False))


def _actor_poisoned_reason(context: TaskRuntimeContext) -> str:
    return str(getattr(context.actor, "poisoned_reason", None) or "ACTOR_POISONED")


class ProtocolRunHooks(RunHooksBase[TaskRuntimeContext, Agent[TaskRuntimeContext]]):
    async def on_llm_start(
        self,
        context: RunContextWrapper[TaskRuntimeContext],
        agent: Agent[TaskRuntimeContext],
        system_prompt: str | None,
        input_items: list[Any],
    ) -> None:
        if context.context.usage_stats is not None:
            context.context.usage_stats.set_runtime_snapshot(context.context.telemetry_snapshot())
        if context.context.progress_callback is not None:
            context.context.progress_callback("model_start")
        if context.context.terminal_browser_error or _actor_poisoned(context.context):
            raise TerminalBrowserError(
                context.context.terminal_browser_error
                or _actor_poisoned_reason(context.context)
            )
        context.context.assert_model_budget()
        if context.context.loop_detected:
            raise NoProgressLoopError(context.context.loop_reason or "NO_PROGRESS_LOOP")


_INTERACTIVE_TAGS = {"a", "button", "input", "select", "textarea", "option", "summary", "details"}
_INTERACTIVE_ROLES = {
    "button",
    "checkbox",
    "combobox",
    "link",
    "listbox",
    "menuitem",
    "option",
    "radio",
    "searchbox",
    "slider",
    "spinbutton",
    "switch",
    "tab",
    "textbox",
}
_ACTION_LABEL = re.compile(
    r"(?:\b(?:next|previous|prev|continue|more|search|submit|download|view)\b|下一|上一|继续|更多|搜索|提交|下载|查看)",
    re.IGNORECASE,
)
_PRIMARY_ACTION_LABEL = re.compile(
    r"(?:\b(?:search|submit|download|view)\b|搜索|提交|下载|查看)",
    re.IGNORECASE,
)


def _task_terms(task: str) -> set[str]:
    normalized = task.casefold()
    terms = {token for token in re.findall(r"[a-z0-9][a-z0-9_.-]{1,}|[\u4e00-\u9fff]{2,}", normalized)}
    for sequence in re.findall(r"[\u4e00-\u9fff]{3,}", normalized):
        terms.update(sequence[index : index + 2] for index in range(len(sequence) - 1))
    return terms


def _is_interactive(element: dict[str, Any]) -> bool:
    return bool(
        element.get("tag") in _INTERACTIVE_TAGS
        or element.get("role") in _INTERACTIVE_ROLES
        or element.get("href")
        or element.get("type")
    )


def _compact_element(element: dict[str, Any]) -> dict[str, Any]:
    compact: dict[str, Any] = {"bid": element["bid"], "tag": element.get("tag", "")}
    limits = {"role": 80, "label": 180, "text": 240, "value": 160, "type": 60, "name": 100}
    for key, limit in limits.items():
        value = element.get(key)
        if value not in (None, ""):
            compact[key] = str(value)[:limit]
    href = element.get("href")
    if href:
        compact["href"] = sanitize_url(str(href))[:300]
    for key in ("checked", "selected", "expanded"):
        if element.get(key) is not None:
            compact[key] = bool(element[key])
    for key in ("new", "changed"):
        if element.get(key) is True:
            compact[key] = True
    if element.get("disabled"):
        compact["disabled"] = True
    if element.get("visible") is False:
        compact["visible"] = False
    if element.get("shadow"):
        compact["shadow"] = True
    if element.get("frame"):
        compact["frame"] = element["frame"]
        if element.get("frame_url"):
            compact["frame_url"] = sanitize_url(str(element["frame_url"]))[:300]
    rect = element.get("rect")
    if _is_interactive(element) and isinstance(rect, list) and len(rect) == 4:
        compact["rect"] = rect
    context = element.get("context")
    if isinstance(context, list):
        compact_context = [str(value)[:120] for value in context[:2] if value not in (None, "")]
        if compact_context:
            compact["context"] = compact_context
    options = element.get("options")
    if isinstance(options, list):
        compact["options"] = [
            {
                key: str(option[key])[:160] if key != "selected" else bool(option[key])
                for key in ("value", "label", "selected")
                if key in option
            }
            for option in options[:40]
            if isinstance(option, dict)
        ]
        if len(options) > 40:
            compact["options_truncated"] = True
    return compact


def _truncate_utf8(value: Any, max_bytes: int) -> str:
    encoded = str(value).encode("utf-8")
    if len(encoded) <= max_bytes:
        return str(value)
    return encoded[:max_bytes].decode("utf-8", errors="ignore")


def _compact_scroll_states(value: Any, limit: int = 24) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    ranked = []
    for index, state in enumerate(value):
        if not isinstance(state, dict) or state.get("kind") not in {"page", "container"}:
            continue
        frame = state.get("frame") if isinstance(state.get("frame"), int) else 1_000_000
        if state["kind"] == "page" and frame == 0:
            priority = 0
        elif state["kind"] == "container" and frame == 0 and state.get("visible") is True:
            priority = 1
        elif state["kind"] == "page":
            priority = 2
        elif state.get("visible") is True:
            priority = 3
        else:
            priority = 4
        ranked.append((priority, frame, index, state))
    ranked.sort(key=lambda item: item[:3])
    projected: list[dict[str, Any]] = []
    for _, _, _, state in ranked:
        if len(projected) >= limit:
            break
        compact: dict[str, Any] = {"kind": state["kind"]}
        for key in ("frame", "position", "remaining", "viewport", "extent"):
            if isinstance(state.get(key), (int, float)):
                compact[key] = max(0, round(state[key]))
        for key in ("top", "bottom"):
            if isinstance(state.get(key), bool):
                compact[key] = state[key]
        for key, size in (("bid", 80), ("tag", 40), ("role", 80), ("label", 96)):
            if state.get(key) not in (None, ""):
                compact[key] = _truncate_utf8(state[key], size)
        if state.get("frame_url"):
            compact["frame_url"] = _truncate_utf8(sanitize_url(str(state["frame_url"])), 240)
        projected.append(compact)
    return projected


def _serialized_bytes(value: Any) -> int:
    return len(json.dumps(value, ensure_ascii=False, separators=(",", ":")).encode("utf-8"))


def _project_observation(
    result: dict[str, Any],
    task: str,
    *,
    unchanged: bool,
    max_elements: int = 120,
    max_chars: int = 20_000,
) -> dict[str, Any]:
    terms = _task_terms(task)
    candidates: list[tuple[int, int, dict[str, Any], bool, bool, bool]] = []
    frame_errors: list[dict[str, Any]] = []
    for index, element in enumerate(result.get("elements", [])):
        if not isinstance(element, dict) or not element.get("bid"):
            if isinstance(element, dict) and element.get("frame_error"):
                frame_errors.append(element)
            continue
        context = element.get("context")
        context_text = " ".join(str(value) for value in context) if isinstance(context, list) else ""
        searchable = " ".join(
            [
                *(
                    str(element.get(key, "")).casefold()
                    for key in ("text", "label", "name", "role", "tag", "href", "value")
                ),
                context_text.casefold(),
            ]
        )
        relevance = sum(1 for term in terms if term in searchable)
        interactive = _is_interactive(element)
        visible = element.get("visible") is not False
        semantic = element.get("tag") in {"h1", "h2", "h3", "h4", "label", "th", "caption"}
        action_label = bool(interactive and _ACTION_LABEL.search(searchable))
        primary_action_label = bool(interactive and _PRIMARY_ACTION_LABEL.search(searchable))
        score = (
            relevance * 100
            + int(interactive) * 60
            + int(visible) * 25
            + int(semantic) * 15
            + int(action_label) * 80
            + int(element.get("new") is True) * 160
            + int(element.get("changed") is True) * 120
        )
        if not (interactive or semantic or relevance or visible):
            continue
        candidates.append((score, index, element, interactive, visible, primary_action_label))
    candidates.sort(key=lambda value: (-value[0], value[1]))

    element_limit = min(max_elements, 24) if unchanged else max_elements
    char_limit = min(max_chars, 6_000) if unchanged else max_chars
    interactive_quota = min(60, max(1, element_limit // 2))
    primary_quota = min(12, max(1, element_limit // 10))
    primary_reserved = [candidate for candidate in candidates if candidate[5]][:primary_quota]
    primary_bids = {str(candidate[2].get("bid")) for candidate in primary_reserved}
    reserved = [
        candidate
        for candidate in candidates
        if candidate[3] and candidate[4] and str(candidate[2].get("bid")) not in primary_bids
    ][:interactive_quota]
    reserved_bids = primary_bids | {str(candidate[2].get("bid")) for candidate in reserved}
    ordered_candidates = [
        *primary_reserved,
        *reserved,
        *(candidate for candidate in candidates if str(candidate[2].get("bid")) not in reserved_bids),
    ]
    projected: list[tuple[int, dict[str, Any]]] = []
    compact_frame_errors = []
    for error in frame_errors[:5]:
        compact_error: dict[str, Any] = {}
        if isinstance(error.get("frame"), int):
            compact_error["frame"] = error["frame"]
        if error.get("frame_error"):
            compact_error["frame_error"] = _truncate_utf8(error["frame_error"], 80)
        if error.get("url"):
            compact_error["url"] = _truncate_utf8(sanitize_url(str(error["url"])), 240)
        compact_frame_errors.append(compact_error)
    raw_url = result.get("url")
    url_limit = min(600, max(32, char_limit // 8))
    title_limit = min(300, max(24, char_limit // 12))
    identifier_limit = min(128, max(16, char_limit // 24))
    base = {
        "url": _truncate_utf8(sanitize_url(str(raw_url)), url_limit) if raw_url else raw_url,
        "title": _truncate_utf8(result.get("title") or "", title_limit),
        "dom_hash": _truncate_utf8(result.get("dom_hash") or "", identifier_limit) or None,
        "semantic_page_fingerprint": _truncate_utf8(
            result.get("semantic_page_fingerprint") or "", identifier_limit
        ) or None,
        "evidence_id": _truncate_utf8(result.get("evidence_id") or "", identifier_limit) or None,
        "unchanged": unchanged,
        "bids_remain_valid": unchanged,
        "total_element_count": len(result.get("elements", [])),
        "source_truncated": bool(result.get("truncated")),
        "frame_errors": [],
    }
    instruction = (
        "DOM 未变化；以下 bid 已在本次观察中确认有效。不要继续重复 observe，按 bid 操作或用 extract 获取正文。"
        if unchanged
        else "这里只包含优先级最高的可见、可交互或任务相关元素；正文、列表和表格请用 extract 按需获取。"
    )
    if char_limit < 1_000:
        instruction = "按有效 bid 操作；正文按需 extract。"

    def render(
        records: list[tuple[int, dict[str, Any]]],
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        elements = [item for _, item in sorted(records, key=lambda record: record[0])]
        return {
            **(metadata or base),
            "elements": elements,
            "element_count": len(elements),
            "elements_truncated_for_model": len(elements) < len(candidates),
            "instruction": instruction,
        }

    element_reserve = min(1_000, char_limit // 5) if candidates else 0
    selected_frame_errors: list[dict[str, Any]] = []
    for error in compact_frame_errors:
        candidate_errors = [*selected_frame_errors, error]
        candidate_base = {**base, "frame_errors": candidate_errors}
        if len(candidate_errors) < len(compact_frame_errors):
            candidate_base["frame_errors_truncated_for_model"] = True
        if _serialized_bytes(render([], candidate_base)) > char_limit - element_reserve:
            continue
        selected_frame_errors.append(error)
    base["frame_errors"] = selected_frame_errors
    if len(selected_frame_errors) < len(compact_frame_errors):
        base["frame_errors_truncated_for_model"] = True

    scroll_candidates = _compact_scroll_states(result.get("scroll"))
    valid_scroll_count = sum(
        1
        for state in result.get("scroll", [])
        if isinstance(state, dict) and state.get("kind") in {"page", "container"}
    ) if isinstance(result.get("scroll"), list) else 0
    scroll_budget = min(3_000, max(500, char_limit // 4))
    scroll: list[dict[str, Any]] = []
    for state in scroll_candidates:
        candidate = [*scroll, state]
        if _serialized_bytes(candidate) > scroll_budget:
            continue
        candidate_base = {**base, "scroll": candidate}
        if len(candidate) < valid_scroll_count:
            candidate_base["scroll_truncated_for_model"] = True
        if _serialized_bytes(render([], candidate_base)) > char_limit - element_reserve:
            continue
        scroll.append(state)
    if scroll:
        base["scroll"] = scroll
    if len(scroll) < valid_scroll_count:
        base["scroll_truncated_for_model"] = True

    deferred_options: dict[int, tuple[list[dict[str, Any]], bool]] = {}
    for _, index, element, _, _, _ in ordered_candidates:
        if len(projected) >= element_limit:
            break
        compact = _compact_element(element)
        options = compact.get("options")
        if isinstance(options, list) and options:
            deferred_options[index] = (options, bool(compact.get("options_truncated")))
            compact.pop("options", None)
            compact["options_truncated"] = True
        candidate = [*projected, (index, compact)]
        if _serialized_bytes(render(candidate)) > char_limit:
            continue
        projected.append((index, compact))

    for record_index, (dom_index, compact) in enumerate(projected):
        deferred = deferred_options.get(dom_index)
        if deferred is None:
            continue
        options, source_truncated = deferred
        low, high = 0, len(options)
        while low < high:
            middle = (low + high + 1) // 2
            attempted = {
                **compact,
                "options": options[:middle],
                "options_truncated": middle < len(options) or source_truncated,
            }
            candidate = [*projected]
            candidate[record_index] = (dom_index, attempted)
            if _serialized_bytes(render(candidate)) <= char_limit:
                low = middle
            else:
                high = middle - 1
        if low:
            expanded = {**compact, "options": options[:low]}
            if low < len(options) or source_truncated:
                expanded["options_truncated"] = True
            else:
                expanded.pop("options_truncated", None)
            projected[record_index] = (dom_index, expanded)
    return render(projected)


def _json(value: Any, limit: int = 140_000) -> str:
    text = json.dumps(value, ensure_ascii=False, separators=(",", ":"))
    if len(text) <= limit:
        return text
    metadata: dict[str, Any] = {}
    if isinstance(value, dict):
        for key in ("ok", "tool", "url", "evidence_id", "evidence_ids", "action_id", "accepted"):
            if key in value:
                metadata[key] = value[key]
    base = {
        "truncated_for_model": True,
        "original_chars": len(text),
        "instruction": "请缩小提取范围，或分批滚动/分页观察。完整证据仍保存在本地。",
        **metadata,
    }

    def render(preview_chars: int) -> str:
        return json.dumps(
            {**base, "preview": text[:preview_chars]},
            ensure_ascii=False,
            separators=(",", ":"),
        )

    low, high = 0, min(len(text), limit)
    while low < high:
        middle = (low + high + 1) // 2
        if len(render(middle)) <= limit:
            low = middle
        else:
            high = middle - 1
    bounded = render(low)
    if len(bounded) <= limit:
        return bounded
    return json.dumps(
        {"truncated_for_model": True, "original_chars": len(text)},
        ensure_ascii=False,
        separators=(",", ":"),
    )


def _error(name: str, exc: Exception) -> str:
    return _json({"ok": False, "tool": name, "error": sanitize_exception(exc)})


def _finish_tool_result(
    context: TaskRuntimeContext,
    name: str,
    arguments: dict[str, Any],
    result: dict[str, Any],
    started: float,
    *,
    cache_hit: bool = False,
    browser_call: bool = False,
) -> dict[str, Any]:
    if browser_call and _actor_poisoned(context):
        context.terminal_browser_error = _actor_poisoned_reason(context)
    elapsed_ms = (time.monotonic() - started) * 1_000
    outcome = context.record_tool_outcome(
        name,
        arguments,
        result,
        cache_hit=cache_hit,
        tool_latency_ms=elapsed_ms,
        browser_latency_ms=elapsed_ms if browser_call else 0.0,
    )
    if context.progress_callback is not None:
        context.progress_callback(f"tool_complete:{name}")
    decorated = {
        **result,
        "progressed": outcome["progressed"],
        "progress_reason": outcome["progress_reason"],
        "cache_hit": cache_hit,
        "repeat_count": outcome["repeat_count"],
        "cycle_count": outcome["cycle_count"],
    }
    if context.loop_detected:
        decorated["loop_guard"] = context.loop_reason
    return decorated


def _tool_exception_result(name: str, exc: Exception) -> dict[str, Any]:
    result: dict[str, Any] = {"ok": False, "tool": name, "error": sanitize_exception(exc)}
    if isinstance(exc, ActorCallDeadlineExceeded):
        result.update(
            {
                "dispatched": exc.dispatched,
                "terminal_uncertain": exc.dispatched,
                "safe_to_retry": not exc.dispatched,
                "task_generation": exc.task_generation,
                "attempt": exc.attempt,
            }
        )
    elif isinstance(exc, BrowserActorPoisonedError):
        result.update({"terminal_uncertain": True, "safe_to_retry": False})
    return result


def _audit_terminal_reason(context: TaskRuntimeContext, exc: Exception | None = None) -> str | None:
    if _actor_poisoned(context):
        return _actor_poisoned_reason(context)
    if isinstance(exc, ActorCallDeadlineExceeded) and exc.dispatched:
        return sanitize_exception(exc)
    if isinstance(exc, BrowserActorPoisonedError):
        return sanitize_exception(exc)
    return None


def _tool_failure(ctx: RunContextWrapper[TaskRuntimeContext], error: Exception) -> str:
    """Account for SDK validation and timeout failures before a tool body runs."""

    name = str(getattr(ctx, "tool_name", "unknown"))
    raw_arguments = getattr(ctx, "tool_arguments", "")
    try:
        decoded = json.loads(raw_arguments) if isinstance(raw_arguments, str) else raw_arguments
        arguments = decoded if isinstance(decoded, dict) else {"invalid_arguments": True}
    except (json.JSONDecodeError, TypeError):
        arguments = {"invalid_arguments": True}
    context = ctx.context
    if _actor_poisoned(context):
        context.terminal_browser_error = _actor_poisoned_reason(context)
    started = time.monotonic()
    try:
        pending_started_call = bool(
            context.actions
            and context.actions[-1].get("step") == context.tool_steps
            and context.actions[-1].get("tool") == name
            and not any(item.get("step") == context.tool_steps for item in context.tool_outcomes)
        )
        if pending_started_call:
            stored_arguments = context.actions[-1].get("arguments")
            if isinstance(stored_arguments, dict):
                arguments = stored_arguments
        else:
            context.record_call(name, arguments)
        result = _finish_tool_result(
            context,
            name,
            arguments,
            _tool_exception_result(name, error),
            started,
        )
        return _json(result)
    except Exception as accounting_error:
        return _json(
            {
                "ok": False,
                "tool": name,
                "error": sanitize_exception(error),
                "accounting_error": sanitize_exception(accounting_error),
            }
        )


def _tracked_tool(*, timeout: float):
    return function_tool(
        timeout=timeout,
        failure_error_function=_tool_failure,
        timeout_error_function=_tool_failure,
    )


async def _receipt_tool(
    ctx: RunContextWrapper[TaskRuntimeContext],
    name: str,
    arguments: dict[str, Any],
    call: Any,
) -> str:
    ctx.context.record_call(name, arguments)
    started = time.monotonic()
    try:
        result = await call
        ctx.context.record_receipt(result, track_outcome=False)
        ctx.context.thoughts.append(f"{name}: {'成功' if result.get('success') else '失败'}")
        result = _finish_tool_result(ctx.context, name, arguments, result, started, browser_call=True)
        return _json(result, limit=24_000)
    except Exception as exc:
        ctx.context.thoughts.append(f"{name}: 异常 {type(exc).__name__}")
        result = _finish_tool_result(
            ctx.context,
            name,
            arguments,
            _tool_exception_result(name, exc),
            started,
            browser_call=True,
        )
        return _json(result)


@_tracked_tool(timeout=30.0)
async def observe(ctx: RunContextWrapper[TaskRuntimeContext]) -> str:
    """获取当前页 DOMSnapshot、合并 iframe AXTree 和带稳定 bid 的结构化元素；截图仅落盘。"""
    ctx.context.record_call("observe", {})
    started = time.monotonic()
    try:
        result = await ctx.context.actor.observe()
        ctx.context.latest_elements = {
            item["bid"]: item for item in result["elements"] if isinstance(item, dict) and item.get("bid")
        }
        ctx.context.visited_urls.append(result["url"])
        semantic = str(result.get("semantic_page_fingerprint") or result["dom_hash"])
        observed_state = (canonical_url(result["url"]), semantic)
        unchanged = observed_state == ctx.context.latest_model_element_state
        result = _finish_tool_result(ctx.context, "observe", {}, result, started, browser_call=True)
        model_result = _project_observation(result, ctx.context.contract.task, unchanged=unchanged)
        ctx.context.latest_model_elements = list(model_result["elements"])
        ctx.context.latest_model_element_state = observed_state
        return _json(
            {
                "ok": True,
                **model_result,
                "progressed": result["progressed"],
                "progress_reason": result["progress_reason"],
                "repeat_count": result["repeat_count"],
                "cycle_count": result["cycle_count"],
                **({"loop_guard": result["loop_guard"]} if result.get("loop_guard") else {}),
            },
            limit=24_000,
        )
    except Exception as exc:
        result = _finish_tool_result(
            ctx.context, "observe", {}, _tool_exception_result("observe", exc), started, browser_call=True
        )
        return _json(result)


@_tracked_tool(timeout=12.0)
async def click(ctx: RunContextWrapper[TaskRuntimeContext], bid: str) -> str:
    """使用最新 observe 的 bid 和 Playwright Locator 点击元素；无坐标入口。"""
    return await _receipt_tool(ctx, "click", {"bid": bid}, ctx.context.actor.click(bid))


@_tracked_tool(timeout=12.0)
async def fill(ctx: RunContextWrapper[TaskRuntimeContext], bid: str, value: str) -> str:
    """使用 Locator.fill 填写原生输入，并回读 value。"""
    observed = ctx.context.latest_elements.get(bid, {})
    recorded_value = "[REDACTED]" if str(observed.get("type", "")).casefold() == "password" else value
    return await _receipt_tool(
        ctx,
        "fill",
        {"bid": bid, "value": recorded_value},
        ctx.context.actor.fill(bid, value),
    )


@_tracked_tool(timeout=12.0)
async def select(ctx: RunContextWrapper[TaskRuntimeContext], bid: str, values: list[str]) -> str:
    """使用 Locator.select_option 选择原生 select 的 value 或 label。"""
    return await _receipt_tool(ctx, "select", {"bid": bid, "values": values}, ctx.context.actor.select(bid, values))


@_tracked_tool(timeout=12.0)
async def set_checked(ctx: RunContextWrapper[TaskRuntimeContext], bid: str, checked: bool) -> str:
    """使用 Locator.set_checked 设置 checkbox/radio 并回读 checked。"""
    return await _receipt_tool(
        ctx, "set_checked", {"bid": bid, "checked": checked}, ctx.context.actor.set_checked(bid, checked)
    )


@_tracked_tool(timeout=12.0)
async def press(ctx: RunContextWrapper[TaskRuntimeContext], key: str, bid: str | None = None) -> str:
    """向指定 bid 或当前页面发送 Playwright 键盘按键，如 Enter、Escape、ArrowDown。"""
    return await _receipt_tool(ctx, "press", {"key": key, "bid": bid}, ctx.context.actor.press(key, bid))


@_tracked_tool(timeout=12.0)
async def scroll(ctx: RunContextWrapper[TaskRuntimeContext], delta_y: int, bid: str | None = None) -> str:
    """滚动页面或指定的虚拟列表容器；delta_y 范围会被限制到正负 4000。"""
    return await _receipt_tool(
        ctx, "scroll", {"delta_y": delta_y, "bid": bid}, ctx.context.actor.scroll(delta_y, bid)
    )


@_tracked_tool(timeout=12.0)
async def wait(ctx: RunContextWrapper[TaskRuntimeContext], milliseconds: int) -> str:
    """等待最多 8 秒，以处理明确延迟加载的 SPA；随后生成 DOM 变化回执。"""
    return await _receipt_tool(
        ctx, "wait", {"milliseconds": milliseconds}, ctx.context.actor.wait(milliseconds)
    )


@_tracked_tool(timeout=12.0)
async def tabs(
    ctx: RunContextWrapper[TaskRuntimeContext],
    action: Literal["list", "switch", "close", "new"],
    index: int | None = None,
    url: str | None = None,
) -> str:
    """列出、切换、关闭或新建标签页；新建页仍由 Playwright 导航。"""
    arguments = {"action": action, "index": index, "url": url}
    ctx.context.record_call("tabs", arguments)
    started = time.monotonic()
    try:
        result = await ctx.context.actor.tabs(action, index, url)
        result = _finish_tool_result(
            ctx.context, "tabs", arguments, {"ok": True, **result}, started, browser_call=True
        )
        return _json(result)
    except Exception as exc:
        result = _finish_tool_result(
            ctx.context, "tabs", arguments, _tool_exception_result("tabs", exc), started, browser_call=True
        )
        return _json(result)


@_tracked_tool(timeout=5.0)
async def dialog(
    ctx: RunContextWrapper[TaskRuntimeContext],
    action: Literal["accept", "dismiss"],
    prompt_text: str | None = None,
) -> str:
    """为下一次触发的 alert/confirm/prompt 预设 accept 或 dismiss，防止点击阻塞。"""
    arguments = {"action": action, "prompt_text": prompt_text}
    ctx.context.record_call("dialog", arguments)
    started = time.monotonic()
    try:
        result = {"ok": True, **(await ctx.context.actor.arm_dialog(action, prompt_text))}
        return _json(_finish_tool_result(ctx.context, "dialog", arguments, result, started, browser_call=True))
    except Exception as exc:
        result = _finish_tool_result(
            ctx.context, "dialog", arguments, _tool_exception_result("dialog", exc), started, browser_call=True
        )
        return _json(result)


@_tracked_tool(timeout=20.0)
async def upload(ctx: RunContextWrapper[TaskRuntimeContext], bid: str, paths: list[str]) -> str:
    """使用 Locator.set_input_files 上传已存在的本地普通文件。"""
    return await _receipt_tool(ctx, "upload", {"bid": bid, "paths": paths}, ctx.context.actor.upload(bid, paths))


@_tracked_tool(timeout=20.0)
async def download(ctx: RunContextWrapper[TaskRuntimeContext], bid: str) -> str:
    """点击 bid 并通过 Playwright expect_download 保存下载文件。"""
    output = await _receipt_tool(ctx, "download", {"bid": bid}, ctx.context.actor.download(bid))
    try:
        payload = json.loads(output)
        path = payload.get("postconditions", {}).get("download", {}).get("path")
        if path:
            ctx.context.downloaded_paths.add(path)
    except (json.JSONDecodeError, TypeError):
        pass
    return output


@_tracked_tool(timeout=20.0)
async def extract(
    ctx: RunContextWrapper[TaskRuntimeContext],
    kind: Literal["text", "links", "table", "list"],
    bid: str | None = None,
    limit: int = 1000,
) -> str:
    """从页面或 bid 子树结构化提取文本、链接、表格或列表并生成证据。"""
    arguments = {"kind": kind, "bid": bid, "limit": limit}
    ctx.context.record_call("extract", arguments)
    started = time.monotonic()
    state = ctx.context.current_semantic_fingerprint or "unknown"
    cache_key = TaskRuntimeContext._signature("extract", {"state": state, **arguments})
    cached = ctx.context.extract_cache.get(cache_key)
    if cached is not None:
        result = _finish_tool_result(
            ctx.context,
            "extract",
            arguments,
            {**cached, "ok": True},
            started,
            cache_hit=True,
        )
        return _json(result, limit=24_000)
    try:
        result = await ctx.context.actor.extract(kind, bid, limit)
        payload = {"ok": True, "kind": kind, **result}
        actual_state = str(result.get("semantic_page_fingerprint") or state)
        actual_key = TaskRuntimeContext._signature("extract", {"state": actual_state, **arguments})
        ctx.context.extract_cache[actual_key] = payload
        payload = _finish_tool_result(ctx.context, "extract", arguments, payload, started, browser_call=True)
        return _json(payload, limit=24_000)
    except Exception as exc:
        result = _finish_tool_result(
            ctx.context, "extract", arguments, _tool_exception_result("extract", exc), started, browser_call=True
        )
        return _json(result)


@_tracked_tool(timeout=30.0)
async def extract_many(
    ctx: RunContextWrapper[TaskRuntimeContext],
    requests: list[ExtractRequestInput],
) -> str:
    """在同一语义页面顺序执行最多 8 个有界 text/links/table/list 提取，并复用已有缓存。"""
    arguments = {"requests": [request.model_dump() for request in requests]}
    ctx.context.record_call("extract_many", arguments)
    started = time.monotonic()
    if not requests or len(requests) > 8:
        result = {"ok": False, "error": "extract_many requests 数量必须为 1 到 8"}
        return _json(_finish_tool_result(ctx.context, "extract_many", arguments, result, started))
    state = ctx.context.current_semantic_fingerprint or "unknown"
    output: list[dict[str, Any] | None] = [None] * len(requests)
    misses: list[tuple[int, dict[str, Any], str]] = []
    seen_request_keys: dict[str, int] = {}
    for index, request in enumerate(requests):
        value = request.model_dump()
        key = TaskRuntimeContext._signature("extract", {"state": state, **value})
        cached = ctx.context.extract_cache.get(key)
        if cached is not None:
            output[index] = {**cached, "cache_hit": True}
        elif key in seen_request_keys:
            misses.append((index, value, key))
        else:
            seen_request_keys[key] = index
            misses.append((index, value, key))
    try:
        unique_misses = [(index, value, key) for index, value, key in misses if seen_request_keys.get(key) == index]
        fetched = await ctx.context.actor.extract_many([value for _, value, _ in unique_misses]) if unique_misses else []
        pending_cache: list[tuple[str, dict[str, Any]]] = []
        for (index, value, key), item in zip(unique_misses, fetched, strict=True):
            actual_state = str(item.get("semantic_page_fingerprint") or state)
            payload = {
                "ok": True,
                "kind": value["kind"],
                **item,
                "semantic_page_fingerprint": actual_state,
                "cache_hit": False,
            }
            actual_key = TaskRuntimeContext._signature("extract", {"state": actual_state, **value})
            pending_cache.append((actual_key, payload))
            output[index] = payload
        for index, _, key in misses:
            if output[index] is None:
                source = output[seen_request_keys[key]]
                if source is None:
                    raise RuntimeError("extract_many duplicate result is unavailable")
                output[index] = {**source, "cache_hit": True}
        actual_states = {
            str(item.get("semantic_page_fingerprint") or state)
            for item in output
            if isinstance(item, dict)
        }
        if len(actual_states) != 1:
            combined = {
                "ok": False,
                "error": "extract_many 跨越多个语义页面状态，结果未发布；请重新 observe 后重试",
                "semantic_page_fingerprints": sorted(actual_states),
            }
        else:
            for key, payload in pending_cache:
                ctx.context.extract_cache[key] = payload
            combined = {
                "ok": True,
                "results": output,
                "cache_hit_count": sum(bool(item and item.get("cache_hit")) for item in output),
                "semantic_page_fingerprint": next(iter(actual_states)),
            }
        combined = _finish_tool_result(
            ctx.context,
            "extract_many",
            arguments,
            combined,
            started,
            cache_hit=not unique_misses,
            browser_call=bool(unique_misses),
        )
        return _json(combined, limit=24_000)
    except Exception as exc:
        result = _finish_tool_result(
            ctx.context,
            "extract_many",
            arguments,
            _tool_exception_result("extract_many", exc),
            started,
            browser_call=True,
        )
        return _json(result)


@_tracked_tool(timeout=20.0)
async def network(ctx: RunContextWrapper[TaskRuntimeContext], since_last: bool = True) -> str:
    """读取由当前浏览器页面真实触发的有界 XHR/Fetch 响应；敏感头已移除。"""
    arguments = {"since_last": since_last}
    ctx.context.record_call("network", arguments)
    started = time.monotonic()
    try:
        result = await ctx.context.actor.network_events(since_last)
        payload = {"ok": True, "record_count": len(result.get("records", [])), **result}
        payload = _finish_tool_result(ctx.context, "network", arguments, payload, started, browser_call=True)
        return _json(payload, limit=24_000)
    except Exception as exc:
        result = _finish_tool_result(
            ctx.context, "network", arguments, _tool_exception_result("network", exc), started, browser_call=True
        )
        return _json(result)


@_tracked_tool(timeout=30.0)
async def document(ctx: RunContextWrapper[TaskRuntimeContext], path: str) -> str:
    """提取本任务通过浏览器下载的 PDF 或文本文件；禁止读取其他路径。"""
    arguments = {"path": path}
    ctx.context.record_call("document", arguments)
    started = time.monotonic()
    if path not in ctx.context.downloaded_paths:
        result = {"ok": False, "error": "该路径不是本任务 download 工具产生的文件"}
        return _json(_finish_tool_result(ctx.context, "document", arguments, result, started))
    try:
        result = await ctx.context.actor.extract_document(path)
        if not result.get("text", "").strip():
            ctx.context.scanned_document_paths.add(path)
        payload = _finish_tool_result(
            ctx.context, "document", arguments, {"ok": True, **result}, started, browser_call=True
        )
        return _json(payload, limit=24_000)
    except Exception as exc:
        result = _finish_tool_result(
            ctx.context, "document", arguments, _tool_exception_result("document", exc), started, browser_call=True
        )
        return _json(result)


def _evidence_page(payload: dict[str, Any], offset: int, limit: int) -> dict[str, Any]:
    key = next((name for name in ("data", "records", "text") if name in payload), None)
    if key is None:
        raise ValueError("该证据不是 extract/network/document 的可分块内容")
    content = payload[key]
    if isinstance(content, list):
        serialized_items = [
            json.dumps(item, ensure_ascii=False, separators=(",", ":"), default=str)
            for item in content
        ]
        if any(len(item) > 18_000 for item in serialized_items):
            serialized = "[" + ",".join(serialized_items) + "]"
            char_limit = min(max(limit, 1), 12_000)
            page_text = serialized[offset : offset + char_limit]
            next_offset = offset + len(page_text)
            return {
                "unit": "char",
                "encoding": "json",
                "field": key,
                "offset": offset,
                "next_offset": next_offset if next_offset < len(serialized) else None,
                "total": len(serialized),
                "content": page_text,
            }
        item_limit = min(max(limit, 1), 200)
        page = content[offset : offset + item_limit]
        while len(page) > 1 and len(json.dumps(page, ensure_ascii=False, separators=(",", ":"))) > 18_000:
            page.pop()
        next_offset = offset + len(page)
        return {
            "unit": "item",
            "field": key,
            "offset": offset,
            "next_offset": next_offset if next_offset < len(content) else None,
            "total": len(content),
            "content": page,
        }
    serialized = (
        content
        if isinstance(content, str)
        else json.dumps(content, ensure_ascii=False, separators=(",", ":"), default=str)
    )
    char_limit = min(max(limit, 1), 12_000)
    page_text = serialized[offset : offset + char_limit]
    next_offset = offset + len(page_text)
    return {
        "unit": "char",
        "field": key,
        "offset": offset,
        "next_offset": next_offset if next_offset < len(serialized) else None,
        "total": len(serialized),
        "content": page_text,
    }


@_tracked_tool(timeout=5.0)
async def recall_evidence(
    ctx: RunContextWrapper[TaskRuntimeContext],
    evidence_id: str,
    offset: int = 0,
    limit: int = 100,
) -> str:
    """按 evidence_id 分块回读本任务已采集的 extract/network/document 内容；不重新访问网页。"""
    arguments = {"evidence_id": evidence_id, "offset": offset, "limit": limit}
    ctx.context.record_call("recall_evidence", arguments)
    started = time.monotonic()
    evidence = ctx.context.evidence_store.get(evidence_id)
    if evidence is None:
        result = {"ok": False, "error": "未知 evidence_id"}
        return _json(_finish_tool_result(ctx.context, "recall_evidence", arguments, result, started))
    if offset < 0:
        result = {"ok": False, "error": "offset 不得小于 0"}
        return _json(_finish_tool_result(ctx.context, "recall_evidence", arguments, result, started))
    try:
        page = _evidence_page(evidence.payload, offset, limit)
    except Exception as exc:
        result = _finish_tool_result(
            ctx.context,
            "recall_evidence",
            arguments,
            _tool_exception_result("recall_evidence", exc),
            started,
        )
        return _json(result)
    payload = {
        "ok": True,
        "evidence_id": evidence.evidence_id,
        "source": evidence.source,
        "url": evidence.url,
        "summary": evidence.summary,
        **page,
    }
    payload = _finish_tool_result(ctx.context, "recall_evidence", arguments, payload, started)
    return _json(payload, limit=24_000)


def _coverage_items(payload: dict[str, Any]) -> list[Any]:
    data = payload.get("data")
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        for key in ("items", "rows", "results", "records", "data"):
            nested = data.get(key)
            if isinstance(nested, list):
                return nested
    records = payload.get("records")
    if isinstance(records, list):
        items: list[Any] = []
        for record in records:
            body = record.get("body") if isinstance(record, dict) else None
            if isinstance(body, list):
                items.extend(body)
            elif isinstance(body, dict):
                for key in ("items", "rows", "results", "records", "data"):
                    nested = body.get(key)
                    if isinstance(nested, list):
                        items.extend(nested)
                        break
        return items
    return []


async def _vision_completion(
    context: TaskRuntimeContext,
    messages: list[dict[str, Any]],
) -> Any:
    metrics = model_input_metrics(None, messages, [])
    serialized_bytes = int(metrics["serialized_context_bytes"])
    estimate = (
        context.rate_limiter.estimate_tokens(serialized_bytes, "vision")
        if context.rate_limiter is not None
        else estimate_input_tokens(None, messages, [], channel="vision")
    )
    try:
        reservation = (
            await context.rate_limiter.acquire(estimate)
            if context.rate_limiter is not None
            else {"wait_seconds": 0.0, "reason": None, "reserved_tokens": estimate}
        )
    except (ValueError, RuntimeError) as exc:
        if context.usage_stats is not None:
            context.usage_stats.record(
                estimated_input_tokens=estimate,
                wait_seconds=0.0,
                throttle_reason=(
                    "pre_send_tpm_lock_timeout"
                    if isinstance(exc, RuntimeError)
                    else "pre_send_request_exceeds_tpm_budget"
                ),
                usage=None,
                error_type=type(exc).__name__,
                channel="vision",
                input_metrics=metrics,
            )
        raise
    except asyncio.CancelledError:
        raise
    except Exception as exc:
        if context.usage_stats is not None:
            context.usage_stats.record(
                estimated_input_tokens=estimate,
                wait_seconds=0.0,
                throttle_reason="pre_send_tpm_limiter_error",
                usage=None,
                error_type=type(exc).__name__,
                channel="vision",
                input_metrics=metrics,
            )
        raise
    model_started = time.monotonic()
    try:
        response = await context.vision_client.chat.completions.create(
            model=context.vision_model,
            messages=messages,
            max_tokens=600,
            **vision_request_options(context.vision_model),
        )
    except asyncio.CancelledError as exc:
        latency = (time.monotonic() - model_started) * 1_000
        if context.usage_stats is not None:
            context.usage_stats.record(
                estimated_input_tokens=estimate,
                wait_seconds=reservation["wait_seconds"],
                throttle_reason=reservation["reason"],
                usage=None,
                error_type=type(exc).__name__,
                channel="vision",
                input_metrics=metrics,
                model_latency_ms=latency,
                reservation=reservation,
            )
        raise
    except Exception as exc:
        latency = (time.monotonic() - model_started) * 1_000
        if context.usage_stats is not None:
            context.usage_stats.record(
                estimated_input_tokens=estimate,
                wait_seconds=reservation["wait_seconds"],
                throttle_reason=reservation["reason"],
                usage=None,
                error_type=type(exc).__name__,
                channel="vision",
                input_metrics=metrics,
                model_latency_ms=latency,
                reservation=reservation,
            )
        raise
    latency = (time.monotonic() - model_started) * 1_000
    actual_input_tokens = int(
        getattr(response.usage, "input_tokens", getattr(response.usage, "prompt_tokens", 0)) or 0
    )
    if context.rate_limiter is not None and actual_input_tokens:
        context.rate_limiter.reconcile(reservation, actual_input_tokens, serialized_bytes, "vision")
    if context.usage_stats is not None:
        context.usage_stats.record(
            estimated_input_tokens=estimate,
            wait_seconds=reservation["wait_seconds"],
            throttle_reason=reservation["reason"],
            usage=response.usage,
            channel="vision",
            input_metrics=metrics,
            model_latency_ms=latency,
            reservation=reservation,
        )
    return response


@_tracked_tool(timeout=5.0)
async def record_coverage(
    ctx: RunContextWrapper[TaskRuntimeContext],
    strategy: Literal["pagination", "cursor", "virtual_list", "declared_total"],
    item_evidence_ids: list[str],
    pages_visited: int,
    expected_total: int | None,
    terminal_reason: Literal[
        "next_disabled", "next_absent", "cursor_exhausted", "no_new_items", "total_matched"
    ],
    terminal_evidence_id: str,
) -> str:
    """从结构化提取/网络证据确定性签发全量覆盖证书；计数、去重和指纹不能由模型填写。"""
    arguments = {
        "strategy": strategy,
        "item_evidence_ids": item_evidence_ids,
        "pages_visited": pages_visited,
        "expected_total": expected_total,
        "terminal_reason": terminal_reason,
        "terminal_evidence_id": terminal_evidence_id,
    }
    ctx.context.record_call("record_coverage", arguments)
    started = time.monotonic()
    unknown = [item for item in [*item_evidence_ids, terminal_evidence_id] if not ctx.context.evidence_store.has(item)]
    if unknown:
        result = {"ok": False, "error": f"存在未知证据: {unknown}"}
        return _json(_finish_tool_result(ctx.context, "record_coverage", arguments, result, started))
    raw_items: list[Any] = []
    for evidence_id in item_evidence_ids:
        evidence = ctx.context.evidence_store.get(evidence_id)
        if evidence:
            raw_items.extend(_coverage_items(evidence.payload))
    if not raw_items:
        result = {"ok": False, "error": "引用证据中没有可计数的结构化条目"}
        return _json(_finish_tool_result(ctx.context, "record_coverage", arguments, result, started))
    canonical = [json.dumps(item, sort_keys=True, ensure_ascii=False, separators=(",", ":")) for item in raw_items]
    unique = sorted(set(canonical))
    fingerprint = hashlib.sha256("\n".join(unique).encode("utf-8")).hexdigest()
    certificate = CoverageCertificate(
        strategy=strategy,
        unique_item_count=len(unique),
        duplicate_item_count=len(canonical) - len(unique),
        pages_visited=pages_visited,
        expected_total=expected_total,
        terminal_reason=terminal_reason,
        terminal_evidence_id=terminal_evidence_id,
        item_fingerprint=fingerprint,
    )
    if expected_total is not None and expected_total != len(unique):
        result = {
            "ok": False,
            "error": f"页面声明总数 {expected_total} 与结构化证据去重数 {len(unique)} 不一致",
        }
        return _json(
            _finish_tool_result(ctx.context, "record_coverage", arguments, result, started)
        )
    existing_id = next(
        (
            coverage_id
            for coverage_id, existing in ctx.context.coverage_records.items()
            if existing == certificate and coverage_id in ctx.context.coverage_evidence_ids
        ),
        None,
    )
    if existing_id is not None:
        terminal_reason = _audit_terminal_reason(ctx.context)
        if terminal_reason is not None:
            ctx.context.terminal_browser_error = terminal_reason
            result = {
                "ok": False,
                "error": terminal_reason,
                "terminal_uncertain": True,
                "safe_to_retry": False,
            }
            return _json(
                _finish_tool_result(
                    ctx.context,
                    "record_coverage",
                    arguments,
                    result,
                    started,
                    browser_call=True,
                )
            )
        result = {
            "ok": True,
            "coverage_id": existing_id,
            "unique_item_count": certificate.unique_item_count,
            "duplicate_item_count": certificate.duplicate_item_count,
            "pages_visited": certificate.pages_visited,
            "expected_total": certificate.expected_total,
            "terminal_reason": certificate.terminal_reason,
        }
        return _json(
            _finish_tool_result(
                ctx.context, "record_coverage", arguments, result, started, cache_hit=True
            )
        )
    coverage_id = f"cov-{secrets.token_hex(6)}"
    try:
        await ctx.context.actor.audit_step("record_coverage")
        terminal_reason = _audit_terminal_reason(ctx.context)
    except Exception as exc:
        terminal_reason = _audit_terminal_reason(ctx.context, exc)
    if terminal_reason is not None:
        ctx.context.terminal_browser_error = terminal_reason
        result = {
            "ok": False,
            "error": terminal_reason,
            "terminal_uncertain": True,
            "safe_to_retry": False,
        }
        return _json(
            _finish_tool_result(
                ctx.context,
                "record_coverage",
                arguments,
                result,
                started,
                browser_call=True,
            )
        )
    try:
        coverage_evidence = ctx.context.evidence_store.add(
            "receipt",
            ctx.context.visited_urls[-1] if ctx.context.visited_urls else ctx.context.contract.website,
            f"签发覆盖证书：{len(unique)} 个唯一条目，{pages_visited} 页",
            {"certificate": certificate.to_dict(), "item_evidence_ids": item_evidence_ids},
        )
    except Exception as exc:
        result = _tool_exception_result("record_coverage", exc)
        return _json(_finish_tool_result(ctx.context, "record_coverage", arguments, result, started))
    ctx.context.coverage_records[coverage_id] = certificate
    ctx.context.coverage_evidence_ids[coverage_id] = coverage_evidence.evidence_id
    result = {
        "ok": True,
        "coverage_id": coverage_id,
        "unique_item_count": certificate.unique_item_count,
        "duplicate_item_count": certificate.duplicate_item_count,
        "pages_visited": certificate.pages_visited,
        "expected_total": certificate.expected_total,
        "terminal_reason": certificate.terminal_reason,
    }
    return _json(_finish_tool_result(ctx.context, "record_coverage", arguments, result, started))


@_tracked_tool(timeout=180.0)
async def visual_inspect(ctx: RunContextWrapper[TaskRuntimeContext], bid: str, question: str) -> str:
    """仅在 DOM/ARIA/网络/文档无法表达图片、Canvas 或图表时，对指定 bid 的局部截图做视觉分析。"""
    arguments = {"bid": bid, "question": question}
    ctx.context.record_call("visual_inspect", arguments)
    started = time.monotonic()
    element = ctx.context.latest_elements.get(bid)
    if not element:
        result = {"ok": False, "error": "bid 不在最新 observe 中"}
        return _json(_finish_tool_result(ctx.context, "visual_inspect", arguments, result, started))
    if element.get("tag") not in {"canvas", "img", "svg"}:
        result = {"ok": False, "error": "该元素不是允许视觉降级的 Canvas/图片/图表"}
        return _json(_finish_tool_result(ctx.context, "visual_inspect", arguments, result, started))
    if element.get("tag") != "canvas" and (element.get("text") or element.get("label")):
        result = {"ok": False, "error": "该元素已有 DOM/ARIA 文本，应先使用结构化证据"}
        return _json(_finish_tool_result(ctx.context, "visual_inspect", arguments, result, started))
    try:
        crop = await ctx.context.actor.visual_crop(bid, question)
        encoded = base64.b64encode(Path(crop["path"]).read_bytes()).decode("ascii")
        response = await _vision_completion(
            ctx.context,
            [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": f"只回答局部图像中可直接观察到、与问题相关的事实。问题：{question}"},
                        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{encoded}"}},
                    ],
                }
            ],
        )
        analysis = response.choices[0].message.content or ""
        evidence = ctx.context.evidence_store.add(
            "visual",
            ctx.context.visited_urls[-1] if ctx.context.visited_urls else ctx.context.contract.website,
            f"局部视觉分析: {question}",
            {"path": crop["path"], "question": question, "analysis": analysis},
        )
        result = {"ok": True, "analysis": analysis, "evidence_id": evidence.evidence_id}
        return _json(_finish_tool_result(ctx.context, "visual_inspect", arguments, result, started))
    except Exception as exc:
        result = _finish_tool_result(
            ctx.context, "visual_inspect", arguments, _tool_exception_result("visual_inspect", exc), started
        )
        return _json(result)


@_tracked_tool(timeout=180.0)
async def visual_document(
    ctx: RunContextWrapper[TaskRuntimeContext], path: str, page_number: int, question: str
) -> str:
    """仅对 document 工具确认无文本的扫描 PDF 页面进行视觉分析。"""
    arguments = {"path": path, "page_number": page_number, "question": question}
    ctx.context.record_call("visual_document", arguments)
    started = time.monotonic()
    if path not in ctx.context.scanned_document_paths:
        result = {"ok": False, "error": "该 PDF 尚未被 document 确认为无文本扫描件"}
        return _json(_finish_tool_result(ctx.context, "visual_document", arguments, result, started))
    try:
        rendered = await ctx.context.actor.render_document_page(path, page_number, question)
        encoded = base64.b64encode(Path(rendered["path"]).read_bytes()).decode("ascii")
        response = await _vision_completion(
            ctx.context,
            [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": f"只回答扫描 PDF 页面中可直接观察到的事实。问题：{question}"},
                        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{encoded}"}},
                    ],
                }
            ],
        )
        analysis = response.choices[0].message.content or ""
        evidence = ctx.context.evidence_store.add(
            "visual",
            ctx.context.visited_urls[-1] if ctx.context.visited_urls else ctx.context.contract.website,
            f"扫描 PDF 视觉分析: {question}",
            {"path": rendered["path"], "document": path, "page": page_number, "analysis": analysis},
        )
        result = {"ok": True, "analysis": analysis, "evidence_id": evidence.evidence_id}
        return _json(_finish_tool_result(ctx.context, "visual_document", arguments, result, started))
    except Exception as exc:
        result = _finish_tool_result(
            ctx.context, "visual_document", arguments, _tool_exception_result("visual_document", exc), started
        )
        return _json(result)


_LIST_TASK_PATTERN = re.compile(
    r"(?:全部|所有|完整|列出|哪些|\blist\s+all\b|\bevery\b|\bwhich\b[^?.]{0,80}\b(?:are|were|have|contain|include)\b)",
    re.IGNORECASE,
)
_NUMBER_PATTERN = re.compile(r"(?<![\w.])[-+]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?(?![\w.])")
_UNIT_PATTERN = re.compile(
    r"(?:%|[$€£¥]|°[cf]|\b(?:percent(?:age)?|m|km|kilometers?|kilometres?|mi|miles?|ft|feet|foot|meters?|metres?|cm|mm|kg|kilograms?|lb|lbs|pounds?|usd|eur|gbp|cny|dollars?|euros?|seconds?|minutes?|hours?|days?|years?|degrees?)\b)",
    re.IGNORECASE,
)
_CHINESE_DIGITS = {"一": 1, "二": 2, "三": 3, "四": 4, "五": 5, "六": 6, "七": 7, "八": 8, "九": 9}


def _structured_answer(answer: str) -> Any | None:
    stripped = answer.strip()
    if not stripped.startswith(("[", "{")):
        return None
    try:
        value = json.loads(stripped)
    except json.JSONDecodeError:
        return None
    return value if isinstance(value, (list, dict)) else None


def _answer_item_count(answer: str, parsed: Any | None) -> int | None:
    if isinstance(parsed, (list, dict)):
        return len(parsed)
    lines = [
        re.sub(r"^\s*(?:[-*•]|\d+[.)])\s*", "", line).strip()
        for line in answer.strip().splitlines()
        if line.strip()
    ]
    if len(lines) > 1:
        return len(lines)
    if not lines:
        return 0
    parts = [part.strip() for part in re.split(r"[,，、;；|]", lines[0]) if part.strip()]
    return len(parts) if len(parts) > 1 else None


def _answer_leaves(value: Any, prefix: str = "$") -> list[tuple[str, Any]]:
    if isinstance(value, dict):
        if not value:
            return [(prefix, value)]
        return [
            leaf
            for key, child in value.items()
            for leaf in _answer_leaves(child, f"{prefix}.{key}")
        ]
    if isinstance(value, list):
        if not value:
            return [(prefix, value)]
        return [
            leaf
            for index, child in enumerate(value)
            for leaf in _answer_leaves(child, f"{prefix}[{index}]")
        ]
    return [(prefix, value)]


def _normalized_numbers(value: str) -> set[str]:
    normalized: set[str] = set()
    for match in _NUMBER_PATTERN.findall(value):
        try:
            number = format(Decimal(match.replace(",", "")).normalize(), "f")
            normalized.add(number.rstrip("0").rstrip(".") if "." in number else number)
        except InvalidOperation:
            continue
    return normalized


def _normalized_units(value: str) -> set[str]:
    aliases = {
        "kilometer": "km",
        "kilometers": "km",
        "kilometre": "km",
        "kilometres": "km",
        "mile": "mi",
        "miles": "mi",
        "feet": "ft",
        "foot": "ft",
        "meter": "m",
        "meters": "m",
        "metre": "m",
        "metres": "m",
        "kilogram": "kg",
        "kilograms": "kg",
        "lb": "lb",
        "lbs": "lb",
        "pound": "lb",
        "pounds": "lb",
        "dollar": "usd",
        "dollars": "usd",
        "euro": "eur",
        "euros": "eur",
        "$": "usd",
        "€": "eur",
        "£": "gbp",
        "¥": "cny",
        "percent": "%",
        "percentage": "%",
    }
    return {aliases.get(match.casefold(), match.casefold()) for match in _UNIT_PATTERN.findall(value)}


def _chinese_number(value: str) -> int | None:
    if value.isdigit():
        return int(value)
    if value == "十":
        return 10
    if "十" in value:
        left, right = value.split("十", 1)
        tens = _CHINESE_DIGITS.get(left, 1 if not left else 0)
        ones = _CHINESE_DIGITS.get(right, 0 if not right else -1)
        return tens * 10 + ones if tens > 0 and ones >= 0 else None
    return _CHINESE_DIGITS.get(value)


def _requested_item_count(task: str) -> int | None:
    match = re.search(
        r"(?:前\s*(\d+|[一二三四五六七八九十]+)|top\s*(?!\d+\s*%)(\d+)(?!\d)|哪\s*(\d+|[一二三四五六七八九十]+)\s*个)",
        task,
        re.IGNORECASE,
    )
    if not match:
        return None
    return _chinese_number(next(value for value in match.groups() if value is not None))


def _required_answer_units(task: str) -> set[str]:
    """Return only units explicitly requested for the answer, excluding filter labels."""

    candidates = [match.group(1) for match in re.finditer(r"[（(]([^()（）]{1,24})[)）]", task)]
    candidates.extend(
        match.group(1)
        for match in re.finditer(
            r"(?:单位(?:是|为|[:：])?|以)\s*([^\s,，。;；?？()（）]{1,20})(?:\s*为单位)?",
            task,
            re.IGNORECASE,
        )
    )
    candidates.extend(
        match.group(1)
        for match in re.finditer(
            r"\b(?:answer|result|value)\s+in\s+([^\s,.;?()]{1,20})",
            task,
            re.IGNORECASE,
        )
    )
    return {unit for candidate in candidates for unit in _normalized_units(candidate)}


def _evidence_alignment_reasons(
    answer_value: Any,
    bindings: dict[str, list[str]],
    evidence_store: EvidenceStore,
) -> list[str]:
    reasons: list[str] = []
    for path, leaf in _answer_leaves(answer_value):
        evidence_text = " ".join(
            json.dumps(ref.payload, ensure_ascii=False, default=str)
            for evidence_id in bindings.get(path, [])
            if (ref := evidence_store.get(evidence_id)) is not None
        )
        if not evidence_text:
            continue
        leaf_text = str(leaf)
        missing_numbers = _normalized_numbers(leaf_text) - _normalized_numbers(evidence_text)
        if missing_numbers:
            reasons.append(f"答案字段 {path} 的数值 {sorted(missing_numbers)} 与绑定证据不一致")
        missing_units = _normalized_units(leaf_text) - _normalized_units(evidence_text)
        if missing_units:
            reasons.append(f"答案字段 {path} 的单位 {sorted(missing_units)} 与绑定证据不一致")
    return reasons


def _answer_shape_reasons(
    contract: TaskContract,
    answer: str,
    certificate: CoverageCertificate | None,
) -> list[str]:
    reasons: list[str] = []
    stripped = answer.strip()
    if len(stripped) > 2_000:
        reasons.append("答案过长；只提交题目要求的值或列表，不要附加解释性长文")
    parsed = _structured_answer(answer)
    item_count = _answer_item_count(answer, parsed)
    requested_count = _requested_item_count(contract.task)
    if requested_count is not None and item_count != requested_count:
        actual = "无法确定" if item_count is None else str(item_count)
        reasons.append(f"题目要求 {requested_count} 项，但答案列表项数为 {actual}")
    required_units = _required_answer_units(contract.task)
    missing_required_units = required_units - _normalized_units(stripped)
    if missing_required_units:
        reasons.append(f"答案缺少题目明确要求的单位 {sorted(missing_required_units)}")
    if certificate and re.search(r"(?:总数|共有多少|多少个|how many|total)", contract.task, re.IGNORECASE):
        numbers = {int(value) for value in re.findall(r"(?<![\d.])\d+(?![\d.])", stripped)}
        if certificate.unique_item_count not in numbers:
            reasons.append(f"答案总数必须与覆盖证据的去重数 {certificate.unique_item_count} 一致")
    if certificate and _LIST_TASK_PATTERN.search(contract.task):
        expected = requested_count if requested_count is not None else certificate.unique_item_count
        if item_count != expected:
            actual = "无法确定" if item_count is None else str(item_count)
            reasons.append(f"答案列表项数为 {actual}，但覆盖与题目要求对应 {expected} 项")
    return reasons


@_tracked_tool(timeout=5.0)
async def finish(
    ctx: RunContextWrapper[TaskRuntimeContext],
    answer: str,
    evidence_ids: list[str],
    evidence_bindings: list[EvidenceBindingInput] | None = None,
    coverage_id: str | None = None,
) -> str:
    """提交候选答案；只有确定性验证通过才会终止，否则原因会回灌并要求继续。"""
    binding_map = {binding.path: binding.evidence_ids for binding in (evidence_bindings or [])}
    structured_answer = _structured_answer(answer)
    if "$" not in binding_map and structured_answer is None and evidence_ids:
        binding_map["$"] = list(dict.fromkeys(evidence_ids))
    arguments = {
        "answer": answer,
        "evidence_ids": evidence_ids,
        "evidence_bindings": binding_map,
        "coverage_id": coverage_id,
    }
    ctx.context.record_call("finish", arguments)
    started = time.monotonic()
    try:
        await ctx.context.actor.audit_step("finish")
        terminal_reason = _audit_terminal_reason(ctx.context)
    except Exception as exc:
        terminal_reason = _audit_terminal_reason(ctx.context, exc)
    if terminal_reason is not None:
        ctx.context.terminal_browser_error = terminal_reason
        result = {
            "accepted": False,
            "reasons": [terminal_reason],
            "instruction": "浏览器终态不确定，当前任务必须终止",
            "agent_answer": None,
            "terminal_uncertain": True,
            "safe_to_retry": False,
        }
        return _json(
            _finish_tool_result(
                ctx.context,
                "finish",
                arguments,
                result,
                started,
                browser_call=True,
            )
        )
    selected_coverage_id = coverage_id or ctx.context.latest_coverage_id
    coverage_evidence_id: str | None = None
    if ctx.context.contract.requires_coverage:
        if selected_coverage_id is None and len(ctx.context.coverage_records) == 1:
            selected_coverage_id = next(iter(ctx.context.coverage_records))
        certificate = ctx.context.coverage_records.get(selected_coverage_id or "")
        if certificate is None:
            reason = (
                f"未知或失效的 coverage_id: {selected_coverage_id}"
                if selected_coverage_id
                else "全量任务缺少 coverage_id；请先调用 record_coverage"
            )
            result = {
                "accepted": False,
                "reasons": [reason],
                "instruction": "继续收集结构化条目与终止证据，使用 record_coverage 返回的短 coverage_id",
                "agent_answer": None,
            }
            return _json(_finish_tool_result(ctx.context, "finish", arguments, result, started))
        coverage_evidence_id = ctx.context.coverage_evidence_ids.get(selected_coverage_id or "")
    else:
        certificate = ctx.context.coverage_records.get(selected_coverage_id or "") if selected_coverage_id else None
    shape_reasons = _answer_shape_reasons(ctx.context.contract, answer, certificate)
    answer_for_verifier = structured_answer if structured_answer is not None else answer
    shape_reasons.extend(
        _evidence_alignment_reasons(answer_for_verifier, binding_map, ctx.context.evidence_store)
    )
    if shape_reasons:
        result = {
            "accepted": False,
            "reasons": shape_reasons,
            "instruction": "仅提交题目要求的值或完整列表，并使数值、单位、条目数量与证据一致",
            "agent_answer": None,
        }
        return _json(_finish_tool_result(ctx.context, "finish", arguments, result, started))
    submitted_evidence_ids = list(
        dict.fromkeys([*evidence_ids, *([coverage_evidence_id] if coverage_evidence_id else [])])
    )
    result = ctx.context.verifier.verify(
        ctx.context.contract,
        answer_for_verifier,
        submitted_evidence_ids,
        binding_map,
        certificate,
        ctx.context.evidence_store,
        ctx.context.receipts,
        ctx.context.visited_urls,
    )
    if result.accepted and _actor_poisoned(ctx.context):
        terminal_reason = _actor_poisoned_reason(ctx.context)
        ctx.context.terminal_browser_error = terminal_reason
        payload = {
            "accepted": False,
            "reasons": [terminal_reason],
            "instruction": "浏览器终态不确定，当前任务必须终止",
            "agent_answer": None,
            "terminal_uncertain": True,
            "safe_to_retry": False,
        }
        return _json(
            _finish_tool_result(
                ctx.context,
                "finish",
                arguments,
                payload,
                started,
                browser_call=True,
            )
        )
    if result.accepted:
        ctx.context.final_answer = answer
        ctx.context.final_evidence_ids = submitted_evidence_ids
        ctx.context.final_bindings = binding_map
        ctx.context.final_coverage = certificate
        ctx.context.finish_accepted = True
    payload = {
        "accepted": result.accepted,
        "reasons": list(result.reasons),
        "instruction": "验证失败，请根据原因继续操作并重新 finish" if not result.accepted else "验证通过",
        "agent_answer": answer if result.accepted else None,
        "coverage_id": selected_coverage_id,
    }
    return _json(_finish_tool_result(ctx.context, "finish", arguments, payload, started))


async def _verified_finish_behavior(
    context: RunContextWrapper[TaskRuntimeContext],
    tool_results: list[FunctionToolResult],
) -> ToolsToFinalOutputResult:
    for tool_result in tool_results:
        if tool_result.tool.name != "finish":
            continue
        try:
            payload = json.loads(str(tool_result.output))
        except json.JSONDecodeError:
            payload = {}
        if payload.get("accepted") is True and context.context.finish_accepted:
            return ToolsToFinalOutputResult(is_final_output=True, final_output=tool_result.output)
    return ToolsToFinalOutputResult(is_final_output=False, final_output=None)


TOOLS = [
    observe,
    click,
    fill,
    select,
    set_checked,
    press,
    scroll,
    wait,
    tabs,
    dialog,
    upload,
    download,
    extract,
    extract_many,
    network,
    document,
    recall_evidence,
    record_coverage,
    visual_inspect,
    visual_document,
    finish,
]


def _model_settings(model: str) -> ModelSettings:
    return build_model_settings(model)


class ProtocolIIIAgent:
    def __init__(
        self,
        model: str,
        api_base: str,
        api_key: str,
        *,
        timeout_seconds: float = 180.0,
        rate_limiter: SharedTPMLimiter | None = None,
        worker_id: int | None = None,
    ) -> None:
        self.model_name = model
        self.rate_limiter = rate_limiter
        self.worker_id = worker_id
        self.client = AsyncOpenAI(
            base_url=api_base,
            api_key=api_key,
            timeout=min(max(timeout_seconds, 1.0), 180.0),
            max_retries=0,
        )
        chat_model = OpenAIChatCompletionsModel(model=model, openai_client=self.client)
        self.model = ThrottledModel(chat_model, rate_limiter)
        self.agent: Agent[TaskRuntimeContext] = Agent(
            name="WebRetriever Protocol III Agent",
            instructions=SYSTEM_INSTRUCTIONS,
            model=self.model,
            tools=TOOLS,
            model_settings=_model_settings(model),
            tool_use_behavior=_verified_finish_behavior,
            reset_tool_choice=False,
        )

    async def run(
        self,
        actor: BrowserActor,
        contract: TaskContract,
        evidence_store: EvidenceStore,
        progress_callback: Callable[[str], None] | None = None,
    ) -> dict[str, Any]:
        usage_stats = TaskUsageStats(self.worker_id, contract.task_idx, contract.task_id)
        self.model.usage_stats = usage_stats
        context = TaskRuntimeContext(
            actor=actor,
            contract=contract,
            evidence_store=evidence_store,
            verifier=CompletionVerifier(),
            vision_client=self.client,
            vision_model=self.model_name,
            usage_stats=usage_stats,
            rate_limiter=self.rate_limiter,
            progress_callback=progress_callback,
        )
        try:
            await Runner.run(
                self.agent,
                input=(
                    f"网站：{contract.website}\n任务：{contract.task}\n"
                    f"模型请求预算：基础 30，按可证明进展增加，绝对上限 {context.hard_step_limit}"
                ),
                context=context,
                max_turns=context.hard_step_limit,
                hooks=ProtocolRunHooks(),
                run_config=RunConfig(
                    tracing_disabled=True,
                    trace_include_sensitive_data=False,
                    call_model_input_filter=BoundedToolOutputFilter(),
                    tool_execution=ToolExecutionConfig(max_function_tool_concurrency=1),
                    tool_not_found_behavior="return_error_to_model",
                ),
            )
            status = "SUCCESS" if context.finish_accepted else "FAIL_UNVERIFIED_FINISH"
            error = None
        except NoProgressLoopError as exc:
            status = "FAIL_NO_PROGRESS"
            error = sanitize_exception(exc)
        except TerminalBrowserError as exc:
            status = "FAIL_BROWSER_POISONED"
            error = sanitize_exception(exc)
        except MaxTurnsExceeded:
            status = "FAIL_MAX_STEPS"
            error = f"达到 Protocol III 模型请求硬上限 {context.hard_step_limit}，且没有通过验证的 finish"
        except Exception as exc:
            status = "FAIL_AGENT_ERROR"
            error = sanitize_exception(exc)
        finally:
            self.model.usage_stats = None
        return {
            "status": status,
            "agent_answer": context.final_answer if status == "SUCCESS" else None,
            "evidence_ids": context.final_evidence_ids,
            "evidence_bindings": context.final_bindings,
            "coverage": context.final_coverage.to_dict() if context.final_coverage else None,
            "actions": context.actions,
            "thoughts": context.thoughts,
            "urls": list(dict.fromkeys(context.visited_urls)),
            "receipts": [receipt.to_dict() for receipt in context.receipts],
            "predict_length": context.tool_steps,
            "progress_credit": context.progress_credit,
            "adaptive_step_budget": context.adaptive_step_budget,
            "tool_outcomes": context.tool_outcomes,
            "model_usage": usage_stats.to_dict(),
            "error": error,
        }
