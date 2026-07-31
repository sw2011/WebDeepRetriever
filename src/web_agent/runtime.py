from __future__ import annotations

import asyncio
import base64
import hashlib
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

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

from .browser_actor import BrowserActor
from .contracts import ActionReceipt, CoverageCertificate, TaskContract
from .evidence import EvidenceStore
from .sanitization import redact_value, sanitize_exception, sanitize_url
from .token_control import SharedTPMLimiter, TaskUsageStats, ThrottledModel, estimate_input_tokens
from .verifier import CompletionVerifier


SYSTEM_INSTRUCTIONS = """你是 WebRetriever Protocol III 的网页任务执行 Agent。
你只能通过给定的 Playwright 工具观察和操作比赛方浏览器，禁止搜索引擎、AnySearch、主动 HTTP/API 请求和坐标点击。

执行规则：
1. 先 observe。所有可定位交互都使用最新观察中的 bid；陈旧 bid 失败后必须重新 observe。
2. 原生 input/select/checkbox 分别使用 fill/select/set_checked。自定义下拉先 click，再 observe 新出现的 listbox/option。
3. SPA、弹窗、多标签、iframe、Shadow DOM、分页和虚拟列表均通过工具处理。不要猜测页面状态。
4. 答案必须来自访问过 URL 的 DOM、页面触发的 XHR/Fetch、下载文档或获准的局部视觉证据。截图默认不会发送给你。
5. 只有结构化页面、ARIA、网络响应和文档文本均无法表达 Canvas、图片、图表或扫描 PDF 时，才调用 visual_inspect。
6. 对“全部、完整、列出、前 N、总数、排名”等任务，必须耗尽分页/游标/虚拟列表，先用 record_coverage 从真实证据签发证书，再在 finish 原样提交。
7. finish 的 evidence_bindings 要覆盖答案文本根字段 `$`；所有 evidence_ids 都必须真实存在。
8. 不得直接输出最终文本。每一轮必须调用且只调用一个工具，最终只能调用 finish。
9. 工具 Schema 中标记 required 但允许 null 的参数必须显式传 null，不得省略或添加未知字段。
10. observe 返回 unchanged=true 时不得继续重复 observe/tabs(list)/wait；应使用现有 bid 操作、按需 extract，或在证据不足时 finish 失败原因。
11. 历史摘要中的结构化证据如需尾部或完整分块，使用 recall_evidence 按 evidence_id 和 offset 回读，不要重复抓取页面。
"""


class CoverageInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    strategy: Literal["pagination", "cursor", "virtual_list", "declared_total", "not_required"]
    unique_item_count: int = Field(ge=0)
    duplicate_item_count: int = Field(default=0, ge=0)
    pages_visited: int = Field(default=1, ge=1)
    expected_total: int | None = Field(default=None, ge=0)
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


class EvidenceBindingInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    path: str
    evidence_ids: list[str]


class BoundedToolOutputFilter:
    """Keep only the latest necessary payloads and summarize older tool history."""

    _CONTENT_TOOLS = {"extract", "network", "document", "recall_evidence"}

    def __init__(
        self,
        keep_recent_outputs: int = 4,
        old_output_chars: int = 800,
        max_current_output_chars: int = 24_000,
        max_total_output_chars: int = 72_000,
    ) -> None:
        self.keep_recent_outputs = keep_recent_outputs
        self.old_output_chars = old_output_chars
        self.max_current_output_chars = max_current_output_chars
        self.max_total_output_chars = max_total_output_chars

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
                "certificate",
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

    def __call__(self, data: CallModelData[Any]) -> ModelInputData:
        model_data = data.model_data
        call_names = {
            str(item.get("call_id")): str(item.get("name", "unknown"))
            for item in model_data.input
            if isinstance(item, dict) and item.get("type") == "function_call"
        }
        output_indexes = [
            index
            for index, item in enumerate(model_data.input)
            if isinstance(item, dict) and item.get("type") == "function_call_output"
        ]
        names = {
            index: call_names.get(str(model_data.input[index].get("call_id")), "unknown")
            for index in output_indexes
        }
        observe_indexes = [index for index in output_indexes if names[index] == "observe"]
        latest_observe = observe_indexes[-1] if observe_indexes else None
        repeated_observe_base: int | None = None
        if latest_observe is not None:
            try:
                latest_observe_payload = json.loads(self._text(model_data.input[latest_observe].get("output")))
            except (json.JSONDecodeError, TypeError):
                latest_observe_payload = {}
            if latest_observe_payload.get("unchanged") is True:
                for observe_index in reversed(observe_indexes[:-1]):
                    try:
                        payload = json.loads(self._text(model_data.input[observe_index].get("output")))
                    except (json.JSONDecodeError, TypeError):
                        continue
                    if payload.get("unchanged") is not True:
                        repeated_observe_base = observe_index
                        break
        latest_content = next(
            (index for index in reversed(output_indexes) if names[index] in self._CONTENT_TOOLS), None
        )
        non_content = [
            index
            for index in output_indexes
            if names[index] not in {"observe", *self._CONTENT_TOOLS}
        ]
        keep_full = set(non_content[-self.keep_recent_outputs :])
        keep_full.update(
            index
            for index in (latest_observe, latest_content, repeated_observe_base)
            if index is not None
        )
        bounded: list[Any] = []
        output_positions = {index: position for position, index in enumerate(output_indexes)}
        for index, item in enumerate(model_data.input):
            if index not in names or not isinstance(item, dict):
                bounded.append(item)
                continue
            output = item.get("output")
            text = self._text(output)
            replacement = dict(item)
            if index in keep_full and len(text) <= self.max_current_output_chars:
                bounded.append(item)
                continue
            if index in keep_full:
                replacement["output"] = self._summary(
                    names[index], text, max(0, self.max_current_output_chars - 1_200)
                )
            else:
                age = len(output_indexes) - output_positions[index]
                preview = self.old_output_chars if age <= 12 else min(180, self.old_output_chars)
                replacement["output"] = self._summary(names[index], text, preview)
            bounded.append(replacement)

        def output_chars(items: list[Any]) -> int:
            return sum(
                len(self._text(item.get("output")))
                for item in items
                if isinstance(item, dict) and item.get("type") == "function_call_output"
            )

        if output_chars(bounded) > self.max_total_output_chars:
            for index, item in enumerate(bounded):
                if output_chars(bounded) <= self.max_total_output_chars:
                    break
                if not isinstance(item, dict) or item.get("type") != "function_call_output":
                    continue
                original_index = index
                if original_index in keep_full:
                    continue
                tool = call_names.get(str(item.get("call_id")), "unknown")
                replacement = dict(item)
                replacement["output"] = self._summary(tool, self._text(item.get("output")), 0)
                bounded[index] = replacement
        protected = {
            index
            for index in (latest_observe, latest_content, repeated_observe_base)
            if index is not None
        }
        if output_chars(bounded) > self.max_total_output_chars:
            for index, item in enumerate(bounded):
                if output_chars(bounded) <= self.max_total_output_chars:
                    break
                if index in protected or not isinstance(item, dict) or item.get("type") != "function_call_output":
                    continue
                tool = call_names.get(str(item.get("call_id")), "unknown")
                replacement = dict(item)
                replacement["output"] = self._summary(tool, self._text(item.get("output")), 0)
                bounded[index] = replacement
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
    seen_tab_states: set[str] = field(default_factory=set)
    no_progress_streak: int = 0
    no_progress_limit: int = 8
    loop_detected: bool = False
    loop_reason: str | None = None

    def record_call(self, name: str, arguments: dict[str, Any]) -> None:
        if self.tool_steps >= self.contract.max_steps:
            raise RuntimeError(f"STEP_LIMIT: 工具调用不得超过 {self.contract.max_steps} 步")
        self.tool_steps += 1
        self.actions.append(
            {"step": self.tool_steps, "tool": name, "arguments": redact_value(arguments)}
        )

    def record_receipt(self, result: dict[str, Any]) -> None:
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
        )
        self.receipts.append(receipt)
        if result.get("after_url"):
            self.visited_urls.append(result["after_url"])

        postconditions = result.get("postconditions", {})
        progress = receipt.changed
        if receipt.success and receipt.action == "scroll":
            progress = progress or postconditions.get("after") != postconditions.get("before")
        if receipt.success and receipt.action in {"fill", "select", "set_checked"}:
            progress = progress or bool(postconditions.get("value_changed"))
        if receipt.success and receipt.action in {"upload", "download"}:
            progress = True
        progress = progress or bool(
            postconditions.get("new_tab_count")
            or postconditions.get("dialog_events")
            or postconditions.get("network_response_count")
            or postconditions.get("confirmation")
        )
        self.note_progress(progress, f"action:{receipt.action}")

    def note_page_state(self, url: str, dom_hash: str) -> bool:
        state = (url, dom_hash)
        unchanged = state == self.last_observation_state
        is_new = state not in self.seen_page_states
        self.seen_page_states.add(state)
        self.last_observation_state = state
        self.note_progress(is_new, "observe")
        return unchanged

    def note_tab_state(self, tabs_value: list[dict[str, Any]]) -> None:
        signature = hashlib.sha256(
            json.dumps(tabs_value, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        ).hexdigest()[:16]
        is_new = signature not in self.seen_tab_states
        self.seen_tab_states.add(signature)
        self.note_progress(is_new, "tabs")

    def note_progress(self, progressed: bool, source: str) -> None:
        if progressed:
            self.no_progress_streak = 0
            return
        self.no_progress_streak += 1
        if self.no_progress_streak >= self.no_progress_limit and not self.loop_detected:
            self.loop_detected = True
            self.loop_reason = (
                f"NO_PROGRESS_LOOP: 连续 {self.no_progress_streak} 次 observe/tabs/wait/动作未产生新页面状态"
            )
            self.thoughts.append(f"loop_guard: {source} 触发无进展保护")


class NoProgressLoopError(RuntimeError):
    pass


class ProtocolRunHooks(RunHooksBase[TaskRuntimeContext, Agent[TaskRuntimeContext]]):
    async def on_llm_start(
        self,
        context: RunContextWrapper[TaskRuntimeContext],
        agent: Agent[TaskRuntimeContext],
        system_prompt: str | None,
        input_items: list[Any],
    ) -> None:
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
    r"(?:next|previous|prev|continue|more|search|submit|download|下一|上一|继续|更多|搜索|提交|下载)",
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
    for key in ("checked", "selected"):
        if element.get(key) is not None:
            compact[key] = bool(element[key])
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


def _project_observation(
    result: dict[str, Any],
    task: str,
    *,
    unchanged: bool,
    max_elements: int = 120,
    max_chars: int = 20_000,
) -> dict[str, Any]:
    terms = _task_terms(task)
    candidates: list[tuple[int, int, dict[str, Any], bool, bool]] = []
    frame_errors: list[dict[str, Any]] = []
    for index, element in enumerate(result.get("elements", [])):
        if not isinstance(element, dict) or not element.get("bid"):
            if isinstance(element, dict) and element.get("frame_error"):
                frame_errors.append(element)
            continue
        searchable = " ".join(
            str(element.get(key, "")).casefold()
            for key in ("text", "label", "name", "role", "tag", "href")
        )
        relevance = sum(1 for term in terms if term in searchable)
        interactive = _is_interactive(element)
        visible = element.get("visible") is not False
        semantic = element.get("tag") in {"h1", "h2", "h3", "h4", "label", "th", "caption"}
        action_label = bool(interactive and _ACTION_LABEL.search(searchable))
        score = (
            relevance * 100
            + int(interactive) * 60
            + int(visible) * 25
            + int(semantic) * 15
            + int(action_label) * 80
        )
        if not (interactive or semantic or relevance or visible):
            continue
        candidates.append((score, -index, element, interactive, visible))
    candidates.sort(reverse=True, key=lambda value: (value[0], value[1]))

    element_limit = min(max_elements, 24) if unchanged else max_elements
    char_limit = min(max_chars, 6_000) if unchanged else max_chars
    interactive_quota = min(60, max(1, element_limit // 2))
    reserved = [candidate for candidate in candidates if candidate[3] and candidate[4]][:interactive_quota]
    reserved_bids = {str(candidate[2].get("bid")) for candidate in reserved}
    ordered_candidates = [
        *reserved,
        *(candidate for candidate in candidates if str(candidate[2].get("bid")) not in reserved_bids),
    ]
    projected: list[dict[str, Any]] = []
    base = {
        "url": result.get("url"),
        "title": result.get("title"),
        "dom_hash": result.get("dom_hash"),
        "evidence_id": result.get("evidence_id"),
        "unchanged": unchanged,
        "bids_remain_valid": unchanged,
        "total_element_count": len(result.get("elements", [])),
        "source_truncated": bool(result.get("truncated")),
        "frame_errors": frame_errors[:5],
    }
    for _, _, element, _, _ in ordered_candidates:
        if len(projected) >= element_limit:
            break
        compact = _compact_element(element)
        options = compact.get("options")
        if isinstance(options, list):
            low, high = 0, len(options)
            while low < high:
                middle = (low + high + 1) // 2
                attempted = {
                    **compact,
                    "options": options[:middle],
                    "options_truncated": middle < len(options) or compact.get("options_truncated", False),
                }
                candidate = {**base, "elements": [*projected, attempted]}
                if len(json.dumps(candidate, ensure_ascii=False, separators=(",", ":"))) <= char_limit:
                    low = middle
                else:
                    high = middle - 1
            compact["options"] = options[:low]
            if low < len(options):
                compact["options_truncated"] = True
            if low == 0:
                compact.pop("options", None)
        candidate = {**base, "elements": [*projected, compact]}
        if len(json.dumps(candidate, ensure_ascii=False, separators=(",", ":"))) > char_limit:
            continue
        projected.append(compact)
    return {
        **base,
        "elements": projected,
        "element_count": len(projected),
        "elements_truncated_for_model": len(projected) < len(candidates),
        "instruction": (
            "DOM 未变化；以下 bid 已在本次观察中确认有效。不要继续重复 observe，按 bid 操作或用 extract 获取正文。"
            if unchanged
            else "这里只包含优先级最高的可见、可交互或任务相关元素；正文、列表和表格请用 extract 按需获取。"
        ),
    }


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


async def _receipt_tool(
    ctx: RunContextWrapper[TaskRuntimeContext],
    name: str,
    arguments: dict[str, Any],
    call: Any,
) -> str:
    ctx.context.record_call(name, arguments)
    try:
        result = await call
        ctx.context.record_receipt(result)
        ctx.context.thoughts.append(f"{name}: {'成功' if result.get('success') else '失败'}")
        if ctx.context.loop_detected:
            result = {**result, "loop_guard": ctx.context.loop_reason}
        return _json(result, limit=24_000)
    except Exception as exc:
        ctx.context.thoughts.append(f"{name}: 异常 {type(exc).__name__}")
        return _error(name, exc)


@function_tool(timeout=30.0)
async def observe(ctx: RunContextWrapper[TaskRuntimeContext]) -> str:
    """获取当前页 DOMSnapshot、合并 iframe AXTree 和带稳定 bid 的结构化元素；截图仅落盘。"""
    ctx.context.record_call("observe", {})
    try:
        result = await ctx.context.actor.observe()
        ctx.context.latest_elements = {
            item["bid"]: item for item in result["elements"] if isinstance(item, dict) and item.get("bid")
        }
        ctx.context.visited_urls.append(result["url"])
        unchanged = ctx.context.note_page_state(result["url"], result["dom_hash"])
        model_result = _project_observation(result, ctx.context.contract.task, unchanged=unchanged)
        if ctx.context.loop_detected:
            model_result["loop_guard"] = ctx.context.loop_reason
        return _json({"ok": True, **model_result}, limit=24_000)
    except Exception as exc:
        return _error("observe", exc)


@function_tool(timeout=12.0)
async def click(ctx: RunContextWrapper[TaskRuntimeContext], bid: str) -> str:
    """使用最新 observe 的 bid 和 Playwright Locator 点击元素；无坐标入口。"""
    return await _receipt_tool(ctx, "click", {"bid": bid}, ctx.context.actor.click(bid))


@function_tool(timeout=12.0)
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


@function_tool(timeout=12.0)
async def select(ctx: RunContextWrapper[TaskRuntimeContext], bid: str, values: list[str]) -> str:
    """使用 Locator.select_option 选择原生 select 的 value 或 label。"""
    return await _receipt_tool(ctx, "select", {"bid": bid, "values": values}, ctx.context.actor.select(bid, values))


@function_tool(timeout=12.0)
async def set_checked(ctx: RunContextWrapper[TaskRuntimeContext], bid: str, checked: bool) -> str:
    """使用 Locator.set_checked 设置 checkbox/radio 并回读 checked。"""
    return await _receipt_tool(
        ctx, "set_checked", {"bid": bid, "checked": checked}, ctx.context.actor.set_checked(bid, checked)
    )


@function_tool(timeout=12.0)
async def press(ctx: RunContextWrapper[TaskRuntimeContext], key: str, bid: str | None = None) -> str:
    """向指定 bid 或当前页面发送 Playwright 键盘按键，如 Enter、Escape、ArrowDown。"""
    return await _receipt_tool(ctx, "press", {"key": key, "bid": bid}, ctx.context.actor.press(key, bid))


@function_tool(timeout=12.0)
async def scroll(ctx: RunContextWrapper[TaskRuntimeContext], delta_y: int, bid: str | None = None) -> str:
    """滚动页面或指定的虚拟列表容器；delta_y 范围会被限制到正负 4000。"""
    return await _receipt_tool(
        ctx, "scroll", {"delta_y": delta_y, "bid": bid}, ctx.context.actor.scroll(delta_y, bid)
    )


@function_tool(timeout=12.0)
async def wait(ctx: RunContextWrapper[TaskRuntimeContext], milliseconds: int) -> str:
    """等待最多 8 秒，以处理明确延迟加载的 SPA；随后生成 DOM 变化回执。"""
    return await _receipt_tool(
        ctx, "wait", {"milliseconds": milliseconds}, ctx.context.actor.wait(milliseconds)
    )


@function_tool(timeout=12.0)
async def tabs(
    ctx: RunContextWrapper[TaskRuntimeContext],
    action: Literal["list", "switch", "close", "new"],
    index: int | None = None,
    url: str | None = None,
) -> str:
    """列出、切换、关闭或新建标签页；新建页仍由 Playwright 导航。"""
    ctx.context.record_call("tabs", {"action": action, "index": index, "url": url})
    try:
        result = await ctx.context.actor.tabs(action, index, url)
        for tab in result.get("tabs", []):
            ctx.context.visited_urls.append(tab["url"])
        ctx.context.note_tab_state(result.get("tabs", []))
        if ctx.context.loop_detected:
            result["loop_guard"] = ctx.context.loop_reason
        return _json({"ok": True, **result})
    except Exception as exc:
        return _error("tabs", exc)


@function_tool(timeout=5.0)
async def dialog(
    ctx: RunContextWrapper[TaskRuntimeContext],
    action: Literal["accept", "dismiss"],
    prompt_text: str | None = None,
) -> str:
    """为下一次触发的 alert/confirm/prompt 预设 accept 或 dismiss，防止点击阻塞。"""
    ctx.context.record_call("dialog", {"action": action, "prompt_text": prompt_text})
    try:
        return _json({"ok": True, **(await ctx.context.actor.arm_dialog(action, prompt_text))})
    except Exception as exc:
        return _error("dialog", exc)


@function_tool(timeout=20.0)
async def upload(ctx: RunContextWrapper[TaskRuntimeContext], bid: str, paths: list[str]) -> str:
    """使用 Locator.set_input_files 上传已存在的本地普通文件。"""
    return await _receipt_tool(ctx, "upload", {"bid": bid, "paths": paths}, ctx.context.actor.upload(bid, paths))


@function_tool(timeout=20.0)
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


@function_tool(timeout=20.0)
async def extract(
    ctx: RunContextWrapper[TaskRuntimeContext],
    kind: Literal["text", "links", "table", "list"],
    bid: str | None = None,
    limit: int = 1000,
) -> str:
    """从页面或 bid 子树结构化提取文本、链接、表格或列表并生成证据。"""
    ctx.context.record_call("extract", {"kind": kind, "bid": bid, "limit": limit})
    try:
        result = await ctx.context.actor.extract(kind, bid, limit)
        return _json({"ok": True, "kind": kind, **result}, limit=24_000)
    except Exception as exc:
        return _error("extract", exc)


@function_tool(timeout=20.0)
async def network(ctx: RunContextWrapper[TaskRuntimeContext], since_last: bool = True) -> str:
    """读取由当前浏览器页面真实触发的有界 XHR/Fetch 响应；敏感头已移除。"""
    ctx.context.record_call("network", {"since_last": since_last})
    try:
        result = await ctx.context.actor.network_events(since_last)
        return _json(
            {"ok": True, "record_count": len(result.get("records", [])), **result},
            limit=24_000,
        )
    except Exception as exc:
        return _error("network", exc)


@function_tool(timeout=30.0)
async def document(ctx: RunContextWrapper[TaskRuntimeContext], path: str) -> str:
    """提取本任务通过浏览器下载的 PDF 或文本文件；禁止读取其他路径。"""
    ctx.context.record_call("document", {"path": path})
    if path not in ctx.context.downloaded_paths:
        return _json({"ok": False, "error": "该路径不是本任务 download 工具产生的文件"})
    try:
        result = await ctx.context.actor.extract_document(path)
        if not result.get("text", "").strip():
            ctx.context.scanned_document_paths.add(path)
        return _json({"ok": True, **result}, limit=24_000)
    except Exception as exc:
        return _error("document", exc)


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


@function_tool(timeout=5.0)
async def recall_evidence(
    ctx: RunContextWrapper[TaskRuntimeContext],
    evidence_id: str,
    offset: int = 0,
    limit: int = 100,
) -> str:
    """按 evidence_id 分块回读本任务已采集的 extract/network/document 内容；不重新访问网页。"""
    ctx.context.record_call(
        "recall_evidence",
        {"evidence_id": evidence_id, "offset": offset, "limit": limit},
    )
    evidence = ctx.context.evidence_store.get(evidence_id)
    if evidence is None:
        return _json({"ok": False, "error": "未知 evidence_id"})
    if offset < 0:
        return _json({"ok": False, "error": "offset 不得小于 0"})
    try:
        page = _evidence_page(evidence.payload, offset, limit)
    except Exception as exc:
        return _error("recall_evidence", exc)
    return _json(
        {
            "ok": True,
            "evidence_id": evidence.evidence_id,
            "source": evidence.source,
            "url": evidence.url,
            "summary": evidence.summary,
            **page,
        },
        limit=24_000,
    )


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
    estimate = estimate_input_tokens(None, messages, [])
    try:
        reservation = (
            await context.rate_limiter.acquire(estimate)
            if context.rate_limiter is not None
            else {"wait_seconds": 0.0, "reason": None}
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
            )
        raise
    try:
        response = await context.vision_client.chat.completions.create(
            model=context.vision_model,
            messages=messages,
            max_tokens=600,
            **_kimi_request_options(context.vision_model),
        )
    except asyncio.CancelledError as exc:
        if context.usage_stats is not None:
            context.usage_stats.record(
                estimated_input_tokens=estimate,
                wait_seconds=reservation["wait_seconds"],
                throttle_reason=reservation["reason"],
                usage=None,
                error_type=type(exc).__name__,
                channel="vision",
            )
        raise
    except Exception as exc:
        if context.usage_stats is not None:
            context.usage_stats.record(
                estimated_input_tokens=estimate,
                wait_seconds=reservation["wait_seconds"],
                throttle_reason=reservation["reason"],
                usage=None,
                error_type=type(exc).__name__,
                channel="vision",
            )
        raise
    if context.usage_stats is not None:
        context.usage_stats.record(
            estimated_input_tokens=estimate,
            wait_seconds=reservation["wait_seconds"],
            throttle_reason=reservation["reason"],
            usage=response.usage,
            channel="vision",
        )
    return response


@function_tool(timeout=5.0)
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
    ctx.context.record_call(
        "record_coverage",
        {
            "strategy": strategy,
            "item_evidence_ids": item_evidence_ids,
            "pages_visited": pages_visited,
            "expected_total": expected_total,
            "terminal_reason": terminal_reason,
            "terminal_evidence_id": terminal_evidence_id,
        },
    )
    unknown = [item for item in [*item_evidence_ids, terminal_evidence_id] if not ctx.context.evidence_store.has(item)]
    if unknown:
        return _json({"ok": False, "error": f"存在未知证据: {unknown}"})
    raw_items: list[Any] = []
    for evidence_id in item_evidence_ids:
        evidence = ctx.context.evidence_store.get(evidence_id)
        if evidence:
            raw_items.extend(_coverage_items(evidence.payload))
    if not raw_items:
        return _json({"ok": False, "error": "引用证据中没有可计数的结构化条目"})
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
        return _json(
            {
                "ok": False,
                "error": f"页面声明总数 {expected_total} 与结构化证据去重数 {len(unique)} 不一致",
                "certificate": certificate.to_dict(),
            }
        )
    ctx.context.coverage_records[fingerprint] = certificate
    try:
        await ctx.context.actor.audit_step("record_coverage")
    except Exception:
        pass
    coverage_evidence = ctx.context.evidence_store.add(
        "receipt",
        ctx.context.visited_urls[-1] if ctx.context.visited_urls else ctx.context.contract.website,
        f"签发覆盖证书：{len(unique)} 个唯一条目，{pages_visited} 页",
        {"certificate": certificate.to_dict(), "item_evidence_ids": item_evidence_ids},
    )
    ctx.context.coverage_evidence_ids[fingerprint] = coverage_evidence.evidence_id
    return _json(
        {"ok": True, "certificate": certificate.to_dict(), "coverage_evidence_id": coverage_evidence.evidence_id}
    )


@function_tool(timeout=180.0)
async def visual_inspect(ctx: RunContextWrapper[TaskRuntimeContext], bid: str, question: str) -> str:
    """仅在 DOM/ARIA/网络/文档无法表达图片、Canvas 或图表时，对指定 bid 的局部截图做视觉分析。"""
    ctx.context.record_call("visual_inspect", {"bid": bid, "question": question})
    element = ctx.context.latest_elements.get(bid)
    if not element:
        return _json({"ok": False, "error": "bid 不在最新 observe 中"})
    if element.get("tag") not in {"canvas", "img", "svg"}:
        return _json({"ok": False, "error": "该元素不是允许视觉降级的 Canvas/图片/图表"})
    if element.get("tag") != "canvas" and (element.get("text") or element.get("label")):
        return _json({"ok": False, "error": "该元素已有 DOM/ARIA 文本，应先使用结构化证据"})
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
        return _json({"ok": True, "analysis": analysis, "evidence_id": evidence.evidence_id})
    except Exception as exc:
        return _error("visual_inspect", exc)


@function_tool(timeout=180.0)
async def visual_document(
    ctx: RunContextWrapper[TaskRuntimeContext], path: str, page_number: int, question: str
) -> str:
    """仅对 document 工具确认无文本的扫描 PDF 页面进行视觉分析。"""
    ctx.context.record_call(
        "visual_document", {"path": path, "page_number": page_number, "question": question}
    )
    if path not in ctx.context.scanned_document_paths:
        return _json({"ok": False, "error": "该 PDF 尚未被 document 确认为无文本扫描件"})
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
        return _json({"ok": True, "analysis": analysis, "evidence_id": evidence.evidence_id})
    except Exception as exc:
        return _error("visual_document", exc)


@function_tool(timeout=5.0)
async def finish(
    ctx: RunContextWrapper[TaskRuntimeContext],
    answer: str,
    evidence_ids: list[str],
    evidence_bindings: list[EvidenceBindingInput],
    coverage: CoverageInput | None = None,
) -> str:
    """提交候选答案；只有确定性验证通过才会终止，否则原因会回灌并要求继续。"""
    binding_map = {binding.path: binding.evidence_ids for binding in evidence_bindings}
    ctx.context.record_call(
        "finish",
        {
            "answer": answer,
            "evidence_ids": evidence_ids,
            "evidence_bindings": binding_map,
            "coverage": coverage.model_dump() if coverage else None,
        },
    )
    try:
        await ctx.context.actor.audit_step("finish")
    except Exception:
        pass
    certificate = CoverageCertificate.from_dict(coverage.model_dump() if coverage else None)
    coverage_evidence_id: str | None = None
    if ctx.context.contract.requires_coverage:
        recorded = ctx.context.coverage_records.get(certificate.item_fingerprint) if certificate else None
        if recorded is None or recorded != certificate:
            return _json(
                {
                    "accepted": False,
                    "reasons": ["CoverageCertificate 未由 record_coverage 基于真实证据签发，或字段被篡改"],
                    "instruction": "继续收集结构化条目与终止证据，先调用 record_coverage，再原样提交证书",
                    "agent_answer": None,
                }
            )
        coverage_evidence_id = ctx.context.coverage_evidence_ids.get(certificate.item_fingerprint)
    submitted_evidence_ids = list(
        dict.fromkeys([*evidence_ids, *([coverage_evidence_id] if coverage_evidence_id else [])])
    )
    result = ctx.context.verifier.verify(
        ctx.context.contract,
        answer,
        submitted_evidence_ids,
        binding_map,
        certificate,
        ctx.context.evidence_store,
        ctx.context.receipts,
        ctx.context.visited_urls,
    )
    if result.accepted:
        ctx.context.final_answer = answer
        ctx.context.final_evidence_ids = submitted_evidence_ids
        ctx.context.final_bindings = binding_map
        ctx.context.final_coverage = certificate
        ctx.context.finish_accepted = True
    return _json(
        {
            "accepted": result.accepted,
            "reasons": list(result.reasons),
            "instruction": "验证失败，请根据原因继续操作并重新 finish" if not result.accepted else "验证通过",
            "agent_answer": answer if result.accepted else None,
        }
    )


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
    network,
    document,
    recall_evidence,
    record_coverage,
    visual_inspect,
    visual_document,
    finish,
]


def _model_settings(model: str) -> ModelSettings:
    settings: dict[str, Any] = {
        "tool_choice": "required",
        "parallel_tool_calls": False,
        "temperature": 0,
    }
    settings.update(_kimi_request_options(model))
    return ModelSettings(**settings)


def _kimi_request_options(model: str) -> dict[str, Any]:
    if model.strip().lower().rsplit("/", 1)[-1] != "kimi-k2.6":
        return {}
    return {
        "temperature": 0.6,
        "extra_body": {"thinking": {"type": "disabled"}},
    }


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
        )
        try:
            await Runner.run(
                self.agent,
                input=f"网站：{contract.website}\n任务：{contract.task}\n最大工具步数：{contract.max_steps}",
                context=context,
                max_turns=contract.max_steps,
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
        except MaxTurnsExceeded:
            status = "FAIL_MAX_STEPS"
            error = f"达到 Protocol III 最大步数 {contract.max_steps}，且没有通过验证的 finish"
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
            "model_usage": usage_stats.to_dict(),
            "error": error,
        }
