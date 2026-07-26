from __future__ import annotations

import asyncio
import gzip
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Callable, TypeVar
from urllib.parse import parse_qs, urlparse

from pypdf import PdfReader

from .contracts import ActionReceipt
from .evidence import EvidenceStore

_VENDORED_BROWSERGYM = Path(__file__).resolve().parents[2] / "vendor" / "browsergym" / "src"
if str(_VENDORED_BROWSERGYM) not in sys.path:
    sys.path.insert(0, str(_VENDORED_BROWSERGYM))

from browsergym.core.action.utils import get_elem_by_bid  # noqa: E402
from browsergym.core.observation import (  # noqa: E402
    _post_extract as browsergym_post_extract,
    _pre_extract as browsergym_pre_extract,
    extract_dom_snapshot as browsergym_extract_dom_snapshot,
    extract_merged_axtree as browsergym_extract_merged_axtree,
)

T = TypeVar("T")

_SENSITIVE_HEADER = re.compile(r"(?:authorization|cookie|token|secret|api[-_]?key)", re.I)
_SENSITIVE_FIELD = re.compile(r"(?:password|passwd|token|secret|api[-_]?key|credential)", re.I)
_CONFIRMATION = re.compile(
    r"(?:成功|已提交|已发送|已保存|已创建|感谢|确认号|success|submitted|sent|saved|created|thank you|confirmation)",
    re.I,
)
_SUBMIT_TARGET = re.compile(
    r"(?:submit|send|save|create|confirm|apply|register|login|sign in|checkout|place order|提交|发送|保存|创建|确认|申请|注册|登录|下单)",
    re.I,
)
_MARK_AND_COLLECT = r"""
({frameIndex, maxElements}) => {
  const output = [];
  const seen = new Set();
  const interactive = new Set([
    'A','BUTTON','INPUT','SELECT','TEXTAREA','OPTION','SUMMARY','DETAILS',
    'VIDEO','AUDIO','IFRAME'
  ]);
  const semantic = new Set([
    'H1','H2','H3','H4','H5','H6','P','LI','DT','DD','TH','TD','TR','TABLE',
    'LABEL','IMG','CANVAS','SVG','PRE','CODE','BLOCKQUOTE','ARTICLE','SECTION'
  ]);
  function walk(root) {
    if (!root || output.length >= maxElements) return;
    const nodes = root.querySelectorAll ? root.querySelectorAll('*') : [];
    for (const el of nodes) {
      if (output.length >= maxElements) break;
      if (seen.has(el)) continue;
      seen.add(el);
      if (el.shadowRoot) walk(el.shadowRoot);
      const rect = el.getBoundingClientRect();
      const style = getComputedStyle(el);
      const role = el.getAttribute('role') || '';
      const hasText = Boolean((el.innerText || el.textContent || '').trim());
      const meaningful = interactive.has(el.tagName) || semantic.has(el.tagName) ||
        Boolean(role) || el.tabIndex >= 0 || el.isContentEditable ||
        (hasText && el.children.length === 0);
      if (!meaningful || style.display === 'none' || style.visibility === 'hidden') continue;
      const bid = el.getAttribute('bid');
      if (!bid) continue;
      const label = el.getAttribute('aria-label') ||
        (el.labels && el.labels.length ? Array.from(el.labels).map(x => x.innerText).join(' ') : '') ||
        el.getAttribute('alt') || el.getAttribute('title') || '';
      let text = (el.innerText || el.textContent || '').replace(/\s+/g, ' ').trim();
      if (text.length > 500) text = text.slice(0, 500) + '...';
      const item = {
        bid, frame: frameIndex, tag: el.tagName.toLowerCase(), role, label,
        text, value: 'value' in el ? String(el.value || '') : '',
        checked: 'checked' in el ? Boolean(el.checked) : null,
        selected: 'selected' in el ? Boolean(el.selected) : null,
        disabled: Boolean(el.disabled || el.getAttribute('aria-disabled') === 'true'),
        href: el.href || '', type: el.type || '', name: el.name || '',
        rect: [Math.round(rect.x), Math.round(rect.y), Math.round(rect.width), Math.round(rect.height)],
        visible: rect.width > 0 && rect.height > 0,
        shadow: Boolean(el.getRootNode() instanceof ShadowRoot),
      };
      if (el.tagName === 'SELECT') {
        item.options = Array.from(el.options).slice(0, 200).map(o => ({
          value: o.value, label: o.label || o.textContent, selected: o.selected,
        }));
      }
      output.push(item);
    }
  }
  walk(document);
  return {elements: output, truncated: output.length >= maxElements};
}
"""


def _sanitize_headers(headers: dict[str, str]) -> dict[str, str]:
    return {key: value for key, value in headers.items() if not _SENSITIVE_HEADER.search(key)}


def _redact(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: ("[REDACTED]" if _SENSITIVE_FIELD.search(str(key)) else _redact(item)) for key, item in value.items()}
    if isinstance(value, list):
        return [_redact(item) for item in value]
    return value


def _safe_json_body(text: str) -> Any:
    try:
        return _redact(json.loads(text))
    except (json.JSONDecodeError, TypeError):
        return text


class BrowserActor:
    """Owns every sync Playwright object on one dedicated thread."""

    def __init__(
        self,
        cdp_url: str,
        output_dir: Path,
        evidence_store: EvidenceStore,
        *,
        click_timeout_ms: int = 8_000,
        response_body_limit: int = 65_536,
        max_network_records: int = 300,
    ) -> None:
        self.cdp_url = cdp_url
        self.output_dir = Path(output_dir)
        self.evidence_store = evidence_store
        self.click_timeout_ms = min(max(click_timeout_ms, 1_000), 8_000)
        self.response_body_limit = response_body_limit
        self.max_network_records = max_network_records
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="browser-actor")
        self._owner_thread: int | None = None
        self._playwright: Any = None
        self._browser: Any = None
        self._context: Any = None
        self._page: Any = None
        self._cdp: Any = None
        self._closed = False
        self._connected = False
        self._bid_counter = 1
        self._step_counter = 0
        self._action_counter = 0
        self._network_cursor = 0
        self._network_records: list[dict[str, Any]] = []
        self._dialogs: list[dict[str, Any]] = []
        self._downloads: list[dict[str, Any]] = []
        self._next_dialog_action: tuple[str, str | None] = ("dismiss", None)
        self._new_pages: list[str] = []
        self._attached_pages: set[int] = set()
        self._pdf_responses: list[Any] = []

    async def _call(self, function: Callable[..., T], *args: Any) -> T:
        if self._closed and function is not self._close_sync:
            raise RuntimeError("BrowserActor 已关闭")
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(self._executor, function, *args)

    def _assert_thread(self) -> None:
        current = threading.get_ident()
        if self._owner_thread is None:
            self._owner_thread = current
        elif self._owner_thread != current:
            raise RuntimeError("Playwright 操作越过 BrowserActor 线程边界")

    async def start(self, initial_url: str) -> dict[str, Any]:
        return await self._call(self._start_sync, initial_url)

    async def begin_task(
        self,
        initial_url: str,
        output_dir: Path,
        evidence_store: EvidenceStore,
    ) -> dict[str, Any]:
        return await self._call(self._begin_task_sync, initial_url, Path(output_dir), evidence_store)

    def _start_sync(self, initial_url: str) -> dict[str, Any]:
        self._assert_thread()
        from playwright.sync_api import sync_playwright

        self._playwright = sync_playwright().start()
        self._playwright.selectors.set_test_id_attribute("bid")
        headers: dict[str, str] = {}
        token = parse_qs(urlparse(self.cdp_url).query).get("access_token", [None])[0]
        if token:
            headers["X-Access-Token"] = token
        kwargs = {"headers": headers} if headers else {}
        self._browser = self._playwright.chromium.connect_over_cdp(self.cdp_url, **kwargs)
        self._connected = True
        self._browser.on("disconnected", self._on_disconnected)
        self._context = self._browser.contexts[0] if self._browser.contexts else self._browser.new_context()
        self._context.on("page", self._on_page)
        return self._prepare_task(initial_url)

    def _begin_task_sync(
        self,
        initial_url: str,
        output_dir: Path,
        evidence_store: EvidenceStore,
    ) -> dict[str, Any]:
        self._assert_thread()
        self.output_dir = output_dir
        self.evidence_store = evidence_store
        self._bid_counter = 1
        self._step_counter = 0
        self._action_counter = 0
        self._network_cursor = 0
        self._network_records = []
        self._dialogs = []
        self._downloads = []
        self._new_pages = []
        self._pdf_responses = []
        if self._playwright is None:
            return self._start_sync(initial_url)
        return self._prepare_task(initial_url)

    def _prepare_task(self, initial_url: str) -> dict[str, Any]:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        for dirname in ("trajectory", "trajectory_visual", "observations", "downloads"):
            (self.output_dir / dirname).mkdir(parents=True, exist_ok=True)
        for old_page in list(self._context.pages):
            try:
                old_page.close(run_before_unload=False)
            except Exception:
                pass
        self._page = self._context.new_page()
        self._attach_page(self._page)
        self._page.goto(initial_url, wait_until="domcontentloaded", timeout=60_000)
        self._settle()
        self._refresh_cdp()
        self._capture_step("start")
        return {"url": self._page.url, "connected": self._connected}

    def _refresh_cdp(self) -> None:
        if self._cdp:
            try:
                self._cdp.detach()
            except Exception:
                pass
        self._cdp = self._context.new_cdp_session(self._page)

    def _attach_page(self, page: Any) -> None:
        identity = id(page)
        if identity in self._attached_pages:
            return
        self._attached_pages.add(identity)
        page.set_default_timeout(self.click_timeout_ms)
        page.on("response", self._on_response)
        page.on("dialog", self._on_dialog)
        page.on("crash", lambda: setattr(self, "_connected", False))

    def _on_disconnected(self) -> None:
        self._connected = False

    def _on_page(self, page: Any) -> None:
        self._assert_thread()
        self._attach_page(page)
        self._new_pages.append(page.url)

    def _on_dialog(self, dialog: Any) -> None:
        self._assert_thread()
        action, prompt_text = self._next_dialog_action
        record = {"type": dialog.type, "message": dialog.message, "action": action, "url": self._page.url}
        try:
            if action == "accept":
                dialog.accept(prompt_text=prompt_text)
            else:
                dialog.dismiss()
        except Exception as exc:
            record["error"] = str(exc)
        self._dialogs.append(record)
        self._next_dialog_action = ("dismiss", None)

    def _on_response(self, response: Any) -> None:
        self._assert_thread()
        content_type = response.headers.get("content-type", "").lower()
        if "application/pdf" in content_type:
            self._pdf_responses.append(response)
        if len(self._network_records) >= self.max_network_records:
            return
        request = response.request
        if request.resource_type not in {"xhr", "fetch"}:
            return
        record: dict[str, Any] = {
            "url": response.url,
            "status": response.status,
            "method": request.method,
            "resource_type": request.resource_type,
            "request_headers": _sanitize_headers(request.headers),
            "response_headers": _sanitize_headers(response.headers),
        }
        post_data = request.post_data
        if post_data:
            record["post_data"] = _safe_json_body(post_data[: self.response_body_limit])
        if any(token in content_type for token in ("json", "text", "xml", "javascript")):
            try:
                body = response.body()[: self.response_body_limit]
                record["body"] = _safe_json_body(body.decode("utf-8", errors="replace"))
                record["body_truncated"] = len(body) >= self.response_body_limit
            except Exception as exc:
                record["body_error"] = str(exc)
        self._network_records.append(record)

    def _ensure_live(self) -> None:
        if not self._connected or self._page is None or self._page.is_closed():
            raise RuntimeError("浏览器已崩溃、页面已关闭或 CDP 已断开")

    def _settle(self) -> None:
        self._ensure_live()
        deadline = time.monotonic() + 2.0
        previous = ""
        stable = 0
        while time.monotonic() < deadline:
            try:
                current = self._dom_hash()
            except Exception:
                break
            if current == previous:
                stable += 1
                if stable >= 2:
                    break
            else:
                stable = 0
                previous = current
            self._page.wait_for_timeout(150)

    def _dom_hash(self) -> str:
        parts: list[str] = []
        for frame in self._page.frames:
            try:
                parts.append(
                    frame.evaluate(
                        """() => {
                          const shadows=[];
                          const visit=root => {
                            for (const el of root.querySelectorAll('*')) {
                              if (el.shadowRoot) { shadows.push(el.shadowRoot.innerHTML); visit(el.shadowRoot); }
                            }
                          };
                          visit(document);
                          return document.documentElement.outerHTML + '\\n' + shadows.join('\\n');
                        }"""
                    )
                )
            except Exception as exc:
                parts.append(f"FRAME_ERROR:{frame.url}:{exc}")
        content = "\nFRAME_BOUNDARY\n".join(parts)
        return hashlib.sha256(content.encode("utf-8", errors="replace")).hexdigest()[:16]

    def _capture_step(self, label: str) -> str:
        self._ensure_live()
        index = self._step_counter
        self._step_counter += 1
        raw = self.output_dir / "trajectory" / f"{index:03d}.png"
        visual = self.output_dir / "trajectory_visual" / f"{index:03d}.png"
        self._page.screenshot(path=str(raw), full_page=False, timeout=30_000)
        shutil.copy2(raw, visual)
        return str(raw)

    async def observe(self) -> dict[str, Any]:
        return await self._call(self._observe_sync)

    def _observe_sync(self) -> dict[str, Any]:
        self._assert_thread()
        self._ensure_live()
        screenshot_path = self._capture_step("observe")
        elements: list[dict[str, Any]] = []
        truncated = False
        browsergym_pre_extract(self._page, tags_to_mark="all", lenient=True)
        try:
            for frame_index, frame in enumerate(self._page.frames):
                try:
                    result = frame.evaluate(
                        _MARK_AND_COLLECT,
                        {"frameIndex": frame_index, "maxElements": 4_000},
                    )
                except Exception as exc:
                    elements.append({"frame": frame_index, "frame_error": str(exc), "url": frame.url})
                    continue
                truncated = truncated or bool(result["truncated"])
                for item in result["elements"]:
                    item["frame_url"] = frame.url
                elements.extend(result["elements"])
            raw_snapshot, ax_tree = self._protocol_observation()
        finally:
            browsergym_post_extract(self._page)
        dom_hash = self._dom_hash()
        artifact = self.output_dir / "observations" / f"{self._step_counter - 1:03d}.json.gz"
        with gzip.open(artifact, "wt", encoding="utf-8") as output:
            json.dump({"dom_snapshot": raw_snapshot, "ax_tree": ax_tree}, output, ensure_ascii=False)
        payload = {
            "title": self._page.title(),
            "dom_hash": dom_hash,
            "elements": elements,
            "element_count": len(elements),
            "truncated": truncated,
            "artifact": str(artifact),
            "screenshot": screenshot_path,
            "screenshot_sent_to_model": False,
        }
        evidence = self.evidence_store.add(
            "dom", self._page.url, f"DOM/AX 观察，共 {len(elements)} 个结构化元素", payload
        )
        return {
            "url": self._page.url,
            "title": self._page.title(),
            "dom_hash": dom_hash,
            "elements": elements,
            "truncated": truncated,
            "evidence_id": evidence.evidence_id,
            "screenshot_path": screenshot_path,
        }

    def _protocol_observation(self) -> tuple[dict[str, Any], list[dict[str, Any]]]:
        try:
            dom_snapshot = browsergym_extract_dom_snapshot(
                self._page,
                computed_styles=[],
                include_dom_rects=True,
                include_paint_order=True,
                temp_data_cleanup=True,
            )
        except Exception as exc:
            dom_snapshot = {"error": str(exc)}
        try:
            merged_ax_tree = browsergym_extract_merged_axtree(self._page)
        except Exception as exc:
            merged_ax_tree = {"error": str(exc), "nodes": []}
        return dom_snapshot, [merged_ax_tree]

    def _locator(self, bid: str) -> Any:
        self._ensure_live()
        if not re.fullmatch(r"[a-zA-Z0-9]+", bid):
            raise ValueError("bid 格式非法")
        try:
            locator = get_elem_by_bid(self._page, bid)
            if locator.count() != 1:
                raise LookupError(f"bid {bid} 不唯一")
            return locator
        except Exception as exc:
            raise LookupError(f"STALE_BID: {bid} 不存在，请重新 observe") from exc

    async def click(self, bid: str) -> dict[str, Any]:
        return await self._call(self._action_sync, "click", lambda: self._click_op(bid), bid)

    def _click_op(self, bid: str) -> dict[str, Any]:
        locator = self._locator(bid)
        target = self._element_state(locator)
        locator.click(timeout=self.click_timeout_ms)
        return {"target": target}

    async def fill(self, bid: str, value: str) -> dict[str, Any]:
        return await self._call(self._action_sync, "fill", lambda: self._fill_op(bid, value), bid)

    def _fill_op(self, bid: str, value: str) -> dict[str, Any]:
        locator = self._locator(bid)
        locator.fill(value, timeout=self.click_timeout_ms)
        return {"value": locator.input_value(), "expected_value": value}

    async def select(self, bid: str, values: list[str]) -> dict[str, Any]:
        return await self._call(self._action_sync, "select", lambda: self._select_op(bid, values), bid)

    def _select_op(self, bid: str, values: list[str]) -> dict[str, Any]:
        locator = self._locator(bid)
        selected = locator.select_option(values, timeout=self.click_timeout_ms)
        return {"selected": selected, "requested": values, "value": locator.input_value()}

    async def set_checked(self, bid: str, checked: bool) -> dict[str, Any]:
        return await self._call(
            self._action_sync, "set_checked", lambda: self._set_checked_op(bid, checked), bid
        )

    def _set_checked_op(self, bid: str, checked: bool) -> dict[str, Any]:
        locator = self._locator(bid)
        locator.set_checked(checked, timeout=self.click_timeout_ms)
        return {"checked": locator.is_checked(), "expected_checked": checked}

    async def press(self, key: str, bid: str | None = None) -> dict[str, Any]:
        return await self._call(self._action_sync, "press", lambda: self._press_op(key, bid), bid)

    def _press_op(self, key: str, bid: str | None) -> dict[str, Any]:
        if bid:
            self._locator(bid).press(key, timeout=self.click_timeout_ms)
        else:
            self._page.keyboard.press(key)
        return {"key": key, "bid": bid}

    async def scroll(self, delta_y: int, bid: str | None = None) -> dict[str, Any]:
        delta_y = min(max(int(delta_y), -4_000), 4_000)
        return await self._call(self._action_sync, "scroll", lambda: self._scroll_op(delta_y, bid), bid)

    def _scroll_op(self, delta_y: int, bid: str | None) -> dict[str, Any]:
        if bid:
            locator = self._locator(bid)
            result = locator.evaluate(
                "(el, dy) => { const before=el.scrollTop; el.scrollBy(0,dy); return {before,after:el.scrollTop,height:el.scrollHeight,client:el.clientHeight}; }",
                delta_y,
            )
        else:
            result = self._page.evaluate(
                "dy => { const before=scrollY; scrollBy(0,dy); return {before,after:scrollY,height:document.documentElement.scrollHeight,client:innerHeight}; }",
                delta_y,
            )
        return {"delta_y": delta_y, "bid": bid, **result}

    async def wait(self, milliseconds: int) -> dict[str, Any]:
        milliseconds = min(max(int(milliseconds), 0), 8_000)
        return await self._call(
            self._action_sync,
            "wait",
            lambda: self._wait_op(milliseconds),
            None,
        )

    def _wait_op(self, milliseconds: int) -> dict[str, Any]:
        self._page.wait_for_timeout(milliseconds)
        return {"milliseconds": milliseconds}

    async def arm_dialog(self, action: str, prompt_text: str | None = None) -> dict[str, Any]:
        return await self._call(self._arm_dialog_sync, action, prompt_text)

    def _arm_dialog_sync(self, action: str, prompt_text: str | None) -> dict[str, Any]:
        self._assert_thread()
        if action not in {"accept", "dismiss"}:
            raise ValueError("dialog action 仅支持 accept 或 dismiss")
        self._next_dialog_action = (action, prompt_text)
        self._capture_step("dialog:arm")
        return {"armed": True, "action": action}

    async def tabs(self, action: str, index: int | None = None, url: str | None = None) -> dict[str, Any]:
        return await self._call(self._tabs_sync, action, index, url)

    def _tabs_sync(self, action: str, index: int | None, url: str | None) -> dict[str, Any]:
        self._assert_thread()
        self._ensure_live()
        pages = [page for page in self._context.pages if not page.is_closed()]
        if action == "list":
            self._capture_step("tabs:list")
            return {"tabs": [{"index": i, "url": p.url, "active": p == self._page} for i, p in enumerate(pages)]}
        if action == "switch":
            if index is None or index < 0 or index >= len(pages):
                raise ValueError("tab index 越界")
            self._page = pages[index]
            self._page.bring_to_front()
            self._refresh_cdp()
        elif action == "close":
            if len(pages) == 1:
                raise ValueError("不能关闭最后一个标签页")
            target = pages[index if index is not None else pages.index(self._page)]
            target.close(run_before_unload=False)
            pages = [page for page in self._context.pages if not page.is_closed()]
            self._page = pages[-1]
            self._refresh_cdp()
        elif action == "new":
            if not url:
                raise ValueError("新建标签页必须提供 URL")
            if urlparse(url).scheme not in {"http", "https"}:
                raise ValueError("新建标签页仅允许 http/https URL")
            self._page = self._context.new_page()
            self._page.goto(url, wait_until="domcontentloaded", timeout=60_000)
            self._refresh_cdp()
        else:
            raise ValueError("tabs action 仅支持 list/switch/close/new")
        self._settle()
        self._capture_step(f"tabs:{action}")
        pages = [page for page in self._context.pages if not page.is_closed()]
        return {"tabs": [{"index": i, "url": p.url, "active": p == self._page} for i, p in enumerate(pages)]}

    async def upload(self, bid: str, paths: list[str]) -> dict[str, Any]:
        return await self._call(self._action_sync, "upload", lambda: self._upload_op(bid, paths), bid)

    def _upload_op(self, bid: str, paths: list[str]) -> dict[str, Any]:
        resolved = [Path(path).expanduser().resolve() for path in paths]
        if not resolved or any(not path.is_file() for path in resolved):
            raise ValueError("上传文件不存在或不是普通文件")
        configured = [value for value in os.getenv("WEBRETRIEVER_UPLOAD_ROOTS", "").split(os.pathsep) if value]
        allowed_roots = [(self.output_dir.parent).resolve(), *(Path(value).expanduser().resolve() for value in configured)]
        if any(not any(path == root or root in path.parents for root in allowed_roots) for path in resolved):
            raise ValueError("上传路径不在任务输出父目录或 WEBRETRIEVER_UPLOAD_ROOTS 白名单中")
        self._locator(bid).set_input_files([str(path) for path in resolved])
        return {"files": [path.name for path in resolved]}

    async def download(self, bid: str) -> dict[str, Any]:
        return await self._call(self._action_sync, "download", lambda: self._download_op(bid), bid)

    def _download_op(self, bid: str) -> dict[str, Any]:
        from playwright.sync_api import TimeoutError as PlaywrightTimeoutError

        pdf_cursor = len(self._pdf_responses)
        try:
            with self._page.expect_download(timeout=self.click_timeout_ms) as info:
                self._locator(bid).click(timeout=self.click_timeout_ms)
            download = info.value
            destination = self.output_dir / "downloads" / download.suggested_filename
            download.save_as(str(destination))
            record = {"filename": download.suggested_filename, "path": str(destination), "url": download.url}
        except PlaywrightTimeoutError as exc:
            candidates = self._pdf_responses[pdf_cursor:]
            if not candidates:
                raise RuntimeError("点击后既未触发下载，也未收到浏览器内联 PDF 响应") from exc
            response = candidates[-1]
            parsed_name = Path(urlparse(response.url).path).name
            filename = parsed_name if parsed_name.lower().endswith(".pdf") else "browser-inline.pdf"
            destination = self.output_dir / "downloads" / filename
            destination.write_bytes(response.body())
            record = {"filename": filename, "path": str(destination), "url": response.url, "inline_pdf": True}
        self._downloads.append(record)
        return {"download": record}

    async def extract(self, kind: str, bid: str | None = None, limit: int = 1_000) -> dict[str, Any]:
        return await self._call(self._extract_sync, kind, bid, min(max(limit, 1), 5_000))

    def _extract_sync(self, kind: str, bid: str | None, limit: int) -> dict[str, Any]:
        self._assert_thread()
        self._ensure_live()
        locator = self._locator(bid) if bid else self._page.locator("body")
        if kind == "text":
            data: Any = locator.inner_text()[:200_000]
        elif kind == "links":
            data = locator.locator("a").evaluate_all(
                "(els, limit) => els.slice(0,limit).map(a => ({text:(a.innerText||'').trim(),href:a.href}))", limit
            )
        elif kind == "table":
            data = locator.locator("table tr").evaluate_all(
                "(rows, limit) => rows.slice(0,limit).map(r => Array.from(r.cells).map(c => (c.innerText||'').trim()))",
                limit,
            )
        elif kind == "list":
            data = locator.locator("li,[role=listitem],[role=row]").evaluate_all(
                "(els, limit) => els.slice(0,limit).map(x => (x.innerText||x.textContent||'').trim())", limit
            )
        else:
            raise ValueError("extract kind 仅支持 text/links/table/list")
        evidence = self.evidence_store.add(
            "dom", self._page.url, f"结构化提取 {kind}", {"kind": kind, "bid": bid, "data": data}
        )
        self._capture_step(f"extract:{kind}")
        return {"data": data, "evidence_id": evidence.evidence_id, "url": self._page.url}

    async def network_events(self, since_last: bool = True) -> dict[str, Any]:
        return await self._call(self._network_events_sync, since_last)

    def _network_events_sync(self, since_last: bool) -> dict[str, Any]:
        self._assert_thread()
        start = self._network_cursor if since_last else 0
        records = self._network_records[start:]
        self._network_cursor = len(self._network_records)
        evidence = self.evidence_store.add(
            "network",
            self._page.url,
            f"页面触发的 XHR/Fetch 响应 {len(records)} 条",
            {"records": records},
        )
        self._capture_step("network")
        return {"records": records, "evidence_id": evidence.evidence_id}

    async def extract_document(self, path: str) -> dict[str, Any]:
        return await self._call(self._extract_document_sync, path)

    def _extract_document_sync(self, path: str) -> dict[str, Any]:
        self._assert_thread()
        target = Path(path).resolve()
        downloads = (self.output_dir / "downloads").resolve()
        if downloads not in target.parents or not target.is_file():
            raise ValueError("只能读取本任务通过浏览器下载的文档")
        if target.suffix.lower() != ".pdf":
            data = target.read_text(encoding="utf-8", errors="replace")[:200_000]
            pages = 1
        else:
            reader = PdfReader(str(target))
            chunks = [(page.extract_text() or "") for page in reader.pages]
            data = "\n\n".join(chunks)[:500_000]
            pages = len(reader.pages)
        evidence = self.evidence_store.add(
            "document", self._page.url, f"下载文档 {target.name}，{pages} 页", {"path": str(target), "pages": pages, "text": data}
        )
        self._capture_step("document")
        return {"path": str(target), "pages": pages, "text": data, "evidence_id": evidence.evidence_id}

    async def visual_crop(self, bid: str, question: str) -> dict[str, Any]:
        return await self._call(self._visual_crop_sync, bid, question)

    def _visual_crop_sync(self, bid: str, question: str) -> dict[str, Any]:
        self._assert_thread()
        locator = self._locator(bid)
        path = self.output_dir / "trajectory_visual" / f"visual-{self._step_counter:03d}.png"
        locator.screenshot(path=str(path), timeout=30_000)
        self._capture_step("visual:element")
        evidence = self.evidence_store.add(
            "visual", self._page.url, f"局部视觉检查: {question}", {"path": str(path), "question": question, "analysis": None}
        )
        return {"path": str(path), "question": question, "evidence_id": evidence.evidence_id}

    async def render_document_page(self, path: str, page_number: int, question: str) -> dict[str, Any]:
        return await self._call(self._render_document_page_sync, path, page_number, question)

    def _render_document_page_sync(self, path: str, page_number: int, question: str) -> dict[str, Any]:
        self._assert_thread()
        target = Path(path).resolve()
        downloads = (self.output_dir / "downloads").resolve()
        if downloads not in target.parents or target.suffix.lower() != ".pdf":
            raise ValueError("只能渲染本任务通过浏览器下载的 PDF")
        page_count = len(PdfReader(str(target)).pages)
        if page_number < 1 or page_number > page_count:
            raise ValueError("PDF 页码越界")
        output_prefix = self.output_dir / "trajectory_visual" / f"document-{page_number:03d}"
        try:
            subprocess.run(
                [
                    "pdftoppm",
                    "-f",
                    str(page_number),
                    "-l",
                    str(page_number),
                    "-singlefile",
                    "-png",
                    "-r",
                    "144",
                    str(target),
                    str(output_prefix),
                ],
                check=True,
                timeout=30,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
                text=True,
            )
        except FileNotFoundError as exc:
            raise RuntimeError("缺少 pdftoppm；请安装 poppler-utils") from exc
        except subprocess.CalledProcessError as exc:
            raise RuntimeError(f"PDF 页面渲染失败: {exc.stderr[:500]}") from exc
        output = output_prefix.with_suffix(".png")
        evidence = self.evidence_store.add(
            "visual",
            self._page.url,
            f"扫描 PDF 第 {page_number} 页局部视觉检查: {question}",
            {"path": str(output), "document": str(target), "page": page_number, "question": question, "analysis": None},
        )
        self._capture_step("visual:document")
        return {"path": str(output), "question": question, "evidence_id": evidence.evidence_id}

    async def audit_step(self, label: str) -> str:
        return await self._call(self._capture_step, label)

    def _element_state(self, locator: Any) -> dict[str, Any]:
        return locator.evaluate(
            "el => ({bid:el.getAttribute('bid'),tag:el.tagName.toLowerCase(),type:el.type||'',text:(el.innerText||el.textContent||'').trim().slice(0,300),value:'value' in el?String(el.value||''):'',checked:'checked' in el?Boolean(el.checked):null,selected:'selected' in el?Boolean(el.selected):null})"
        )

    def _action_sync(self, action: str, operation: Callable[[], dict[str, Any]], bid: str | None) -> dict[str, Any]:
        self._assert_thread()
        self._action_counter += 1
        action_id = f"act-{self._action_counter:04d}"
        try:
            self._ensure_live()
            before_url = self._page.url
            before_hash = self._dom_hash()
        except Exception as exc:
            receipt = ActionReceipt(action_id, action, False, "", "", "", "", {}, error=str(exc))
            return receipt.to_dict()
        dialogs_before = len(self._dialogs)
        pages_before = len(self._context.pages)
        network_before = len(self._network_records)
        stale = False
        error: str | None = None
        postconditions: dict[str, Any] = {}
        try:
            postconditions.update(operation())
            self._settle()
            pages = [page for page in self._context.pages if not page.is_closed()]
            if len(pages) > pages_before:
                self._page = pages[-1]
                self._page.bring_to_front()
                self._refresh_cdp()
                self._settle()
            success = True
        except Exception as exc:
            error = str(exc)
            stale = "STALE_BID" in error
            success = False
        try:
            after_url = self._page.url
            after_hash = self._dom_hash()
            page_text = self._page.locator("body").inner_text(timeout=2_000)[:50_000]
        except Exception as exc:
            after_url, after_hash, page_text = before_url, before_hash, ""
            self._connected = False
            success = False
            error = error or str(exc)
        postconditions.update(
            {
                "bid": bid,
                "dialog_events": self._dialogs[dialogs_before:],
                "new_tab_count": max(0, len(self._context.pages) - pages_before) if self._context else 0,
                "network_response_count": max(0, len(self._network_records) - network_before),
                "confirmation": self._is_confirmed_submission(
                    action,
                    success,
                    before_url != after_url or before_hash != after_hash,
                    postconditions,
                    page_text,
                ),
            }
        )
        receipt_evidence = self.evidence_store.add(
            "receipt",
            after_url,
            f"{action} {'成功' if success else '失败'}",
            {"action_id": action_id, "postconditions": postconditions, "error": error},
        )
        receipt = ActionReceipt(
            action_id,
            action,
            success,
            before_url,
            after_url,
            before_hash,
            after_hash,
            postconditions,
            (receipt_evidence.evidence_id,),
            error,
            stale,
        )
        try:
            self._capture_step(action)
        except Exception:
            pass
        return receipt.to_dict()

    @staticmethod
    def _is_confirmed_submission(
        action: str,
        success: bool,
        changed: bool,
        postconditions: dict[str, Any],
        page_text: str,
    ) -> bool:
        if not success or not changed or not _CONFIRMATION.search(page_text):
            return False
        if action == "press":
            return str(postconditions.get("key", "")).lower() in {"enter", "return"}
        if action != "click":
            return False
        target = postconditions.get("target", {})
        target_text = " ".join(
            str(target.get(key, "")) for key in ("type", "text", "value")
        )
        return target.get("type") == "submit" or bool(_SUBMIT_TARGET.search(target_text))

    async def close(self) -> None:
        if self._closed:
            return
        try:
            await self._call(self._close_sync)
        finally:
            self._closed = True
            self._executor.shutdown(wait=True, cancel_futures=True)

    async def flush_artifacts(self) -> None:
        await self._call(self._flush_artifacts_sync)

    def _flush_artifacts_sync(self) -> None:
        self._assert_thread()
        capture_path = self.output_dir / "capture.json"
        capture_path.write_text(json.dumps(self._network_records, ensure_ascii=False, indent=2), encoding="utf-8")

    def _close_sync(self) -> None:
        self._assert_thread()
        self._flush_artifacts_sync()
        try:
            if self._page and not self._page.is_closed():
                self._page.close(run_before_unload=False)
        except Exception:
            pass
        try:
            if self._browser:
                self._browser.close()
        except Exception:
            pass
        try:
            if self._playwright:
                self._playwright.stop()
        except Exception:
            pass
        self._connected = False
