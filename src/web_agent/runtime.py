from __future__ import annotations

import base64
import hashlib
import json
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
from agents.extensions import ToolOutputTrimmer
from agents.run_config import CallModelData, ModelInputData
from openai import AsyncOpenAI
from pydantic import BaseModel, ConfigDict, Field

from .browser_actor import BrowserActor
from .contracts import ActionReceipt, CoverageCertificate, TaskContract
from .evidence import EvidenceStore
from .sanitization import redact_value, sanitize_exception
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
    """Compose the SDK trimmer with a tool-count window for single-user-message runs."""

    def __init__(self, keep_recent_outputs: int = 8, old_output_chars: int = 4_000) -> None:
        self.keep_recent_outputs = keep_recent_outputs
        self.old_output_chars = old_output_chars
        self.sdk_trimmer = ToolOutputTrimmer(
            recent_turns=1,
            max_output_chars=24_000,
            preview_chars=4_000,
            trimmable_tools={"observe", "extract", "network", "document"},
        )

    def __call__(self, data: CallModelData[Any]) -> ModelInputData:
        model_data = self.sdk_trimmer(data)
        output_indexes = [
            index
            for index, item in enumerate(model_data.input)
            if isinstance(item, dict) and item.get("type") == "function_call_output"
        ]
        old_indexes = set(output_indexes[: -self.keep_recent_outputs]) if len(output_indexes) > self.keep_recent_outputs else set()
        if not old_indexes:
            return model_data
        bounded: list[Any] = []
        for index, item in enumerate(model_data.input):
            if index not in old_indexes or not isinstance(item, dict):
                bounded.append(item)
                continue
            output = item.get("output")
            text = output if isinstance(output, str) else json.dumps(output, ensure_ascii=False)
            if len(text) <= self.old_output_chars:
                bounded.append(item)
                continue
            replacement = dict(item)
            replacement["output"] = _json(
                {
                    "trimmed_old_tool_output": True,
                    "original_chars": len(text),
                    "preview": text[: self.old_output_chars],
                },
                limit=self.old_output_chars + 500,
            )
            bounded.append(replacement)
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

    def record_call(self, name: str, arguments: dict[str, Any]) -> None:
        if self.tool_steps >= self.contract.max_steps:
            raise RuntimeError(f"STEP_LIMIT: 工具调用不得超过 {self.contract.max_steps} 步")
        self.tool_steps += 1
        self.actions.append(
            {"step": self.tool_steps, "tool": name, "arguments": redact_value(arguments)}
        )

    def record_receipt(self, result: dict[str, Any]) -> None:
        self.receipts.append(
            ActionReceipt(
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
        )
        if result.get("after_url"):
            self.visited_urls.append(result["after_url"])


def _json(value: Any, limit: int = 140_000) -> str:
    text = json.dumps(value, ensure_ascii=False, separators=(",", ":"))
    if len(text) <= limit:
        return text
    return json.dumps(
        {
            "truncated_for_model": True,
            "original_chars": len(text),
            "preview": text[:limit],
            "instruction": "请缩小提取范围，或分批滚动/分页观察。完整证据仍保存在本地。",
        },
        ensure_ascii=False,
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
        return _json(result)
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
        model_result = {key: value for key, value in result.items() if key != "screenshot_path"}
        return _json({"ok": True, **model_result})
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
        return _json({"ok": True, **(await ctx.context.actor.extract(kind, bid, limit))})
    except Exception as exc:
        return _error("extract", exc)


@function_tool(timeout=20.0)
async def network(ctx: RunContextWrapper[TaskRuntimeContext], since_last: bool = True) -> str:
    """读取由当前浏览器页面真实触发的有界 XHR/Fetch 响应；敏感头已移除。"""
    ctx.context.record_call("network", {"since_last": since_last})
    try:
        return _json({"ok": True, **(await ctx.context.actor.network_events(since_last))})
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
        return _json({"ok": True, **result})
    except Exception as exc:
        return _error("document", exc)


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
        response = await ctx.context.vision_client.chat.completions.create(
            model=ctx.context.vision_model,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": f"只回答局部图像中可直接观察到、与问题相关的事实。问题：{question}"},
                        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{encoded}"}},
                    ],
                }
            ],
            max_tokens=600,
            **_kimi_request_options(ctx.context.vision_model),
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
        response = await ctx.context.vision_client.chat.completions.create(
            model=ctx.context.vision_model,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": f"只回答扫描 PDF 页面中可直接观察到的事实。问题：{question}"},
                        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{encoded}"}},
                    ],
                }
            ],
            max_tokens=600,
            **_kimi_request_options(ctx.context.vision_model),
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
    def __init__(self, model: str, api_base: str, api_key: str, *, timeout_seconds: float = 180.0) -> None:
        self.model_name = model
        self.client = AsyncOpenAI(
            base_url=api_base,
            api_key=api_key,
            timeout=min(max(timeout_seconds, 1.0), 180.0),
            max_retries=0,
        )
        chat_model = OpenAIChatCompletionsModel(model=model, openai_client=self.client)
        self.agent: Agent[TaskRuntimeContext] = Agent(
            name="WebRetriever Protocol III Agent",
            instructions=SYSTEM_INSTRUCTIONS,
            model=chat_model,
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
        context = TaskRuntimeContext(
            actor=actor,
            contract=contract,
            evidence_store=evidence_store,
            verifier=CompletionVerifier(),
            vision_client=self.client,
            vision_model=self.model_name,
        )
        try:
            await Runner.run(
                self.agent,
                input=f"网站：{contract.website}\n任务：{contract.task}\n最大工具步数：{contract.max_steps}",
                context=context,
                max_turns=contract.max_steps,
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
        except MaxTurnsExceeded:
            status = "FAIL_MAX_STEPS"
            error = f"达到 Protocol III 最大步数 {contract.max_steps}，且没有通过验证的 finish"
        except Exception as exc:
            status = "FAIL_AGENT_ERROR"
            error = sanitize_exception(exc)
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
            "error": error,
        }
