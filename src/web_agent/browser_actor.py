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
from urllib.parse import parse_qs, parse_qsl, urlencode, urlparse, urlsplit, urlunsplit

from pypdf import PdfReader

from .contracts import ActionReceipt
from .evidence import EvidenceStore
from .sanitization import SENSITIVE_FIELD, redact_text, redact_value, sanitize_exception, sanitize_url

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
_CONFIRMATION = re.compile(
    r"(?:成功|已提交|已发送|已保存|已创建|感谢|确认号|success|submitted|sent|saved|created|thank you|confirmation)",
    re.I,
)
_SUBMIT_TARGET = re.compile(
    r"(?:submit|send|save|create|confirm|apply|register|login|sign in|checkout|place order|提交|发送|保存|创建|确认|申请|注册|登录|下单)",
    re.I,
)
_VOLATILE_TEXT = re.compile(
    r"(?:\b\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}:\d{2}(?:\.\d+)?Z?\b|\b\d{2}:\d{2}:\d{2}(?:\.\d+)?\b|\b1\d{12}\b)"
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
        text, value: 'value' in el ? (el.type === 'password' ? '[REDACTED]' : String(el.value || '')) : '',
        checked: 'checked' in el ? Boolean(el.checked) : null,
        selected: 'selected' in el ? Boolean(el.selected) : null,
        disabled: Boolean(el.disabled || el.getAttribute('aria-disabled') === 'true'),
        href: el.href || '', type: el.type || '', name: el.name || '',
        rect: [Math.round(rect.x), Math.round(rect.y), Math.round(rect.width), Math.round(rect.height)],
        visible: rect.width > 0 && rect.height > 0 && rect.bottom > 0 && rect.right > 0 &&
          rect.top < window.innerHeight && rect.left < window.innerWidth,
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
    return {
        key: redact_text(value)
        for key, value in headers.items()
        if not _SENSITIVE_HEADER.search(key)
    }


def _redact(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: ("[REDACTED]" if SENSITIVE_FIELD.search(str(key)) else _redact(item)) for key, item in value.items()}
    if isinstance(value, list):
        return [_redact(item) for item in value]
    if isinstance(value, str):
        return redact_text(value)
    return value


def _request_body_size(request: Any, headers: dict[str, str]) -> int | None:
    length_header = headers.get("content-length", "")
    if length_header:
        try:
            length = int(length_header)
            return length if length >= 0 else None
        except ValueError:
            return None
    try:
        sizes = request.sizes()
        length = int(sizes.get("requestBodySize", -1))
        return length if length >= 0 else None
    except Exception:
        return None


def _error_type(exc: Exception) -> str:
    return type(exc).__name__


def canonical_url(value: str) -> str:
    """Normalize a URL for state identity while retaining meaningful query parameters."""

    parsed = urlsplit(value)
    host = (parsed.hostname or "").casefold()
    port = parsed.port
    if port and not ((parsed.scheme.casefold() == "http" and port == 80) or (parsed.scheme.casefold() == "https" and port == 443)):
        host = f"{host}:{port}"
    path = parsed.path or "/"
    query = urlencode(sorted(parse_qsl(parsed.query, keep_blank_values=True)), doseq=True)
    return urlunsplit((parsed.scheme.casefold(), host, path, query, parsed.fragment))


def semantic_page_fingerprint(
    url: str,
    title: str,
    elements: list[dict[str, Any]],
    *,
    scroll_state: list[Any] | None = None,
) -> str:
    """Hash bounded semantic state, excluding raw markup, bids, layout and volatile clocks."""

    projected_by_frame: dict[str, list[dict[str, Any]]] = {}
    for element in elements:
        if not isinstance(element, dict) or element.get("frame_error"):
            continue
        item: dict[str, Any] = {}
        for key in (
            "frame_url",
            "tag",
            "role",
            "label",
            "text",
            "value",
            "checked",
            "selected",
            "disabled",
            "href",
            "type",
            "name",
            "options",
        ):
            value = element.get(key)
            if value in (None, "", [], {}):
                continue
            if isinstance(value, str):
                value = _VOLATILE_TEXT.sub("<volatile-time>", re.sub(r"\s+", " ", value).strip())
                if key in {"href", "frame_url"}:
                    value = canonical_url(value)
            item[key] = value
        if item:
            frame_key = str(element.get("frame", item.get("frame_url", "main")))
            projected_by_frame.setdefault(frame_key, []).append(item)
    frame_summaries = []
    for frame_key, projected in projected_by_frame.items():
        encoded_frame = json.dumps(
            projected,
            sort_keys=True,
            ensure_ascii=False,
            separators=(",", ":"),
            default=str,
        )
        frame_summaries.append(
            {
                "frame": frame_key,
                "count": len(projected),
                "digest": hashlib.sha256(encoded_frame.encode("utf-8", errors="replace")).hexdigest(),
            }
        )
    payload = {
        "url": canonical_url(url),
        "title": _VOLATILE_TEXT.sub("<volatile-time>", title.strip()),
        "frames": frame_summaries,
        "scroll": scroll_state or [],
    }
    encoded = json.dumps(payload, sort_keys=True, ensure_ascii=False, separators=(",", ":"), default=str)
    return hashlib.sha256(encoded.encode("utf-8", errors="replace")).hexdigest()[:20]


def _opaque_body(raw: bytes, limit: int) -> dict[str, Any]:
    return {
        "binary_or_unstructured": True,
        "byte_length": len(raw),
        "sha256": hashlib.sha256(raw).hexdigest(),
        "truncated": len(raw) > limit,
    }


def _safe_post_data(raw: bytes, limit: int, content_type: str = "") -> Any:
    normalized_type = content_type.lower()
    if len(raw) > limit or "multipart/" in normalized_type:
        return _opaque_body(raw, limit)
    bounded = raw[:limit]
    try:
        text = bounded.decode("utf-8")
    except UnicodeDecodeError:
        return _opaque_body(raw, limit)
    if "application/x-www-form-urlencoded" in normalized_type:
        return _redact(parse_qs(text, keep_blank_values=True))
    try:
        return _redact(json.loads(text))
    except (json.JSONDecodeError, TypeError):
        return _opaque_body(raw, limit)


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
        self._observation_counter = 0
        self._visual_counter = 0
        self._network_cursor = 0
        self._network_records: list[dict[str, Any]] = []
        self._dialogs: list[dict[str, Any]] = []
        self._downloads: list[dict[str, Any]] = []
        self._next_dialog_action: tuple[str, str | None] = ("dismiss", None)
        self._new_pages: list[str] = []
        self._attached_pages: set[int] = set()
        self._pdf_responses: list[Any] = []
        self._last_dom_hash: str | None = None
        self._last_semantic_state: dict[str, Any] | None = None
        self._last_capture_semantic: str | None = None
        self._last_capture_path: str | None = None
        self._last_observation_semantic: str | None = None

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
        self._observation_counter = 0
        self._visual_counter = 0
        self._network_cursor = 0
        self._network_records = []
        self._dialogs = []
        self._downloads = []
        self._new_pages = []
        self._pdf_responses = []
        self._last_dom_hash = None
        self._last_semantic_state = None
        self._last_capture_semantic = None
        self._last_capture_path = None
        self._last_observation_semantic = None
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
        self._capture_step("start", force=True)
        return {"url": sanitize_url(self._page.url), "connected": self._connected}

    def _refresh_cdp(self) -> None:
        if self._cdp:
            try:
                self._cdp.detach()
            except Exception:
                pass
        self._cdp = self._context.new_cdp_session(self._page)
        self._last_dom_hash = None
        self._last_semantic_state = None

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
        self._new_pages.append(sanitize_url(page.url))

    def _on_dialog(self, dialog: Any) -> None:
        self._assert_thread()
        action, prompt_text = self._next_dialog_action
        record = {
            "type": dialog.type,
            "message": redact_text(dialog.message),
            "action": action,
            "url": sanitize_url(self._page.url),
        }
        try:
            if action == "accept":
                dialog.accept(prompt_text=prompt_text)
            else:
                dialog.dismiss()
        except Exception as exc:
            record["error"] = _error_type(exc)
        self._dialogs.append(record)
        self._next_dialog_action = ("dismiss", None)

    def _on_response(self, response: Any) -> None:
        self._assert_thread()
        try:
            self._capture_response(response)
        except Exception as exc:
            if len(self._network_records) < self.max_network_records:
                self._network_records.append({"capture_error": _error_type(exc)})

    def _capture_response(self, response: Any) -> None:
        content_type = response.headers.get("content-type", "").lower()
        if "application/pdf" in content_type:
            self._pdf_responses.append(response)
        if len(self._network_records) >= self.max_network_records:
            return
        request = response.request
        if request.resource_type not in {"xhr", "fetch"}:
            return
        record: dict[str, Any] = {
            "url": sanitize_url(response.url),
            "status": response.status,
            "method": request.method,
            "resource_type": request.resource_type,
            "request_headers": _sanitize_headers(request.headers),
            "response_headers": _sanitize_headers(response.headers),
        }
        request_content_type = record["request_headers"].get("content-type", "")
        if request.method.upper() not in {"GET", "HEAD"}:
            request_body_size = _request_body_size(request, record["request_headers"])
            if request_body_size is None:
                record["post_data_skipped"] = "请求未声明可信正文长度"
            elif request_body_size > self.response_body_limit:
                record["post_data_skipped"] = f"请求正文长度 {request_body_size} 超过采集上限"
                record["post_data_truncated"] = True
            elif request_body_size > 0:
                try:
                    post_data = request.post_data_buffer
                except Exception as exc:
                    record["post_data_error"] = _error_type(exc)
                else:
                    if post_data:
                        record["post_data"] = _safe_post_data(
                            post_data,
                            self.response_body_limit,
                            request_content_type,
                        )
                        record["post_data_truncated"] = len(post_data) > self.response_body_limit
        if any(token in content_type for token in ("json", "text", "xml", "javascript")):
            length_header = record["response_headers"].get("content-length", "")
            content_encoding = record["response_headers"].get("content-encoding", "").lower()
            try:
                declared_length = int(length_header) if length_header else None
            except ValueError:
                declared_length = None
            if declared_length is None or declared_length < 0:
                record["body_skipped"] = "响应未声明可信 Content-Length"
            elif content_encoding and content_encoding != "identity":
                record["body_skipped"] = f"响应使用 {content_encoding} 压缩，跳过解压后大小未知的正文"
            elif declared_length > self.response_body_limit:
                record["body_skipped"] = f"声明长度 {declared_length} 超过采集上限"
                record["body_truncated"] = True
            else:
                try:
                    body = response.body()
                    record["body"] = _safe_post_data(body, self.response_body_limit, content_type)
                    record["body_truncated"] = len(body) > self.response_body_limit
                except Exception as exc:
                    record["body_error"] = _error_type(exc)
        self._network_records.append(record)

    def _ensure_live(self) -> None:
        if not self._connected or self._page is None or self._page.is_closed():
            raise RuntimeError("浏览器已崩溃、页面已关闭或 CDP 已断开")

    def _settle(self) -> dict[str, Any] | None:
        self._ensure_live()
        deadline = time.monotonic() + 2.0
        previous = ""
        stable = 0
        latest: dict[str, Any] | None = None
        while time.monotonic() < deadline:
            try:
                latest = self._semantic_page_state()
                current = str(latest["semantic_page_fingerprint"])
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
        return latest

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
        value = hashlib.sha256(content.encode("utf-8", errors="replace")).hexdigest()[:16]
        self._last_dom_hash = value
        return value

    def _semantic_page_state(self) -> dict[str, Any]:
        elements: list[dict[str, Any]] = []
        scroll_state: list[Any] = []
        for frame_index, frame in enumerate(self._page.frames):
            try:
                value = frame.evaluate(
                    """() => {
                      const elements=[];
                      const scrollables=[];
                      const queue=[];
                      for (const el of document.body?.children||[]) queue.push({el,hidden:false});
                      let cursor=0;
                      while (cursor<queue.length && cursor<12000 && elements.length<3000) {
                        const current=queue[cursor++];
                        const el=current.el;
                        if (['SCRIPT','STYLE','NOSCRIPT','TEMPLATE'].includes(el.tagName)) continue;
                        const style=getComputedStyle(el);
                        const hidden=current.hidden||el.hidden||el.getAttribute('aria-hidden')==='true'||style.display==='none'||style.visibility==='hidden';
                        if (!hidden) {
                          for (const child of el.children) queue.push({el:child,hidden:false});
                          if (el.shadowRoot) for (const child of el.shadowRoot.children) queue.push({el:child,hidden:false});
                        }
                        if (hidden) continue;
                        const tag=el.tagName.toLowerCase();
                        const role=el.getAttribute('role')||'';
                        const directText=Array.from(el.childNodes).some(node => node.nodeType===Node.TEXT_NODE && (node.textContent||'').trim());
                        const meaningful=directText||role||['a','button','input','select','textarea','option','li','tr','th','td','h1','h2','h3','h4','img','canvas','svg'].includes(tag);
                        if (!meaningful) continue;
                        let text=(el.textContent||'').replace(/\\s+/g,' ').trim();
                        if (text.length>240) text=text.slice(0,240);
                        elements.push({
                          tag, role, text,
                          label:el.getAttribute('aria-label')||el.getAttribute('alt')||el.getAttribute('title')||'',
                          value:'value' in el?(el.type==='password'?'[REDACTED]':String(el.value||'').slice(0,160)):'',
                          checked:'checked' in el?Boolean(el.checked):null,
                          selected:'selected' in el?Boolean(el.selected):null,
                          disabled:Boolean(el.disabled||el.getAttribute('aria-disabled')==='true'),
                          href:el.href||'', type:el.type||'', name:el.name||''
                        });
                        if (el.scrollHeight>el.clientHeight) scrollables.push(el);
                      }
                      const scroll=[Math.round(scrollX),Math.round(scrollY)];
                      for (const el of scrollables) {
                        if (scroll.length>=22) break;
                        scroll.push([el.getAttribute('role')||el.tagName,Math.round(el.scrollTop)]);
                      }
                      return {elements,scroll};
                    }"""
                )
                for item in value.get("elements", []):
                    item["frame"] = frame_index
                    item["frame_url"] = sanitize_url(frame.url)
                elements.extend(value.get("elements", []))
                scroll_state.append([sanitize_url(frame.url), *value.get("scroll", [])])
            except Exception:
                continue
        url = sanitize_url(self._page.url)
        title = self._page.title()
        fingerprint = semantic_page_fingerprint(url, title, elements, scroll_state=scroll_state)
        state = {
            "url": url,
            "title": title,
            "semantic_page_fingerprint": fingerprint,
            "scroll_state": scroll_state,
            "semantic_element_count": len(elements),
        }
        self._last_semantic_state = state
        return state

    def _capture_step(self, label: str, force: bool = False) -> str:
        self._ensure_live()
        semantic = (self._last_semantic_state or self._semantic_page_state()).get("semantic_page_fingerprint")
        if not force and semantic == self._last_capture_semantic and self._last_capture_path:
            return self._last_capture_path
        index = self._step_counter
        self._step_counter += 1
        raw = self.output_dir / "trajectory" / f"{index:03d}.png"
        visual = self.output_dir / "trajectory_visual" / f"{index:03d}.png"
        self._page.screenshot(path=str(raw), full_page=False, timeout=30_000)
        try:
            os.link(raw, visual)
        except OSError:
            shutil.copy2(raw, visual)
        self._last_capture_semantic = str(semantic)
        self._last_capture_path = str(raw)
        return self._last_capture_path

    async def observe(self) -> dict[str, Any]:
        return await self._call(self._observe_sync)

    def _observe_sync(self) -> dict[str, Any]:
        self._assert_thread()
        self._ensure_live()
        current_state = self._semantic_page_state()
        screenshot_path = self._capture_step(
            "observe",
            force=self._last_observation_semantic != current_state["semantic_page_fingerprint"],
        )
        self._last_observation_semantic = str(current_state["semantic_page_fingerprint"])
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
                    elements.append(
                        {"frame": frame_index, "frame_error": _error_type(exc), "url": sanitize_url(frame.url)}
                    )
                    continue
                truncated = truncated or bool(result["truncated"])
                for item in result["elements"]:
                    item["frame_url"] = sanitize_url(frame.url)
                elements.extend(result["elements"])
            raw_snapshot, ax_tree = self._protocol_observation()
        finally:
            browsergym_post_extract(self._page)
        dom_hash = self._dom_hash()
        artifact = self.output_dir / "observations" / f"{self._observation_counter:03d}.json.gz"
        self._observation_counter += 1
        with gzip.open(artifact, "wt", encoding="utf-8") as output:
            json.dump(
                {
                    "url": sanitize_url(self._page.url),
                    "title": self._page.title(),
                    "dom_hash": dom_hash,
                    "semantic_page_fingerprint": current_state["semantic_page_fingerprint"],
                    "elements": elements,
                    "elements_truncated": truncated,
                    "dom_snapshot": raw_snapshot,
                    "ax_tree": ax_tree,
                },
                output,
                ensure_ascii=False,
            )
        payload = {
            "title": self._page.title(),
            "dom_hash": dom_hash,
            "semantic_page_fingerprint": current_state["semantic_page_fingerprint"],
            "elements": elements,
            "element_count": len(elements),
            "truncated": truncated,
            "artifact": str(artifact),
            "screenshot": screenshot_path,
            "screenshot_sent_to_model": False,
        }
        evidence = self.evidence_store.add(
            "dom", sanitize_url(self._page.url), f"DOM/AX 观察，共 {len(elements)} 个结构化元素", payload
        )
        return {
            "url": sanitize_url(self._page.url),
            "title": self._page.title(),
            "dom_hash": dom_hash,
            "semantic_page_fingerprint": current_state["semantic_page_fingerprint"],
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
            dom_snapshot = {"error": _error_type(exc)}
        try:
            merged_ax_tree = browsergym_extract_merged_axtree(self._page)
        except Exception as exc:
            merged_ax_tree = {"error": _error_type(exc), "nodes": []}
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
        before = locator.input_value()
        locator.fill(value, timeout=self.click_timeout_ms)
        is_password = (locator.get_attribute("type") or "").casefold() == "password"
        after = locator.input_value()
        return {
            "before_value": "[REDACTED]" if is_password else before,
            "value": "[REDACTED]" if is_password else after,
            "expected_value": "[REDACTED]" if is_password else value,
            "value_changed": before != after,
        }

    async def select(self, bid: str, values: list[str]) -> dict[str, Any]:
        return await self._call(self._action_sync, "select", lambda: self._select_op(bid, values), bid)

    def _select_op(self, bid: str, values: list[str]) -> dict[str, Any]:
        locator = self._locator(bid)
        before = locator.input_value()
        selected = locator.select_option(values, timeout=self.click_timeout_ms)
        after = locator.input_value()
        return {
            "selected": selected,
            "requested": values,
            "before_value": before,
            "value": after,
            "value_changed": before != after,
        }

    async def set_checked(self, bid: str, checked: bool) -> dict[str, Any]:
        return await self._call(
            self._action_sync, "set_checked", lambda: self._set_checked_op(bid, checked), bid
        )

    def _set_checked_op(self, bid: str, checked: bool) -> dict[str, Any]:
        locator = self._locator(bid)
        before = locator.is_checked()
        locator.set_checked(checked, timeout=self.click_timeout_ms)
        after = locator.is_checked()
        return {
            "before_checked": before,
            "checked": after,
            "expected_checked": checked,
            "value_changed": before != after,
        }

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
            state = self._last_semantic_state or self._semantic_page_state()
            return {
                "tabs": [
                    {"index": i, "url": sanitize_url(p.url), "active": p == self._page}
                    for i, p in enumerate(pages)
                ],
                "semantic_page_fingerprint": state["semantic_page_fingerprint"],
            }
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
            target_url = canonical_url(url)
            existing = next((page for page in pages if canonical_url(page.url) == target_url), None)
            if existing is not None:
                self._page = existing
                self._page.bring_to_front()
                self._refresh_cdp()
                reused = True
            else:
                self._page = self._context.new_page()
                self._page.goto(url, wait_until="domcontentloaded", timeout=60_000)
                self._refresh_cdp()
                reused = False
        else:
            raise ValueError("tabs action 仅支持 list/switch/close/new")
        state = self._settle()
        self._capture_step(f"tabs:{action}")
        pages = [page for page in self._context.pages if not page.is_closed()]
        return {
            "tabs": [
                {"index": i, "url": sanitize_url(p.url), "active": p == self._page}
                for i, p in enumerate(pages)
            ],
            "reused": reused if action == "new" else False,
            "semantic_page_fingerprint": (state or self._semantic_page_state())["semantic_page_fingerprint"],
        }

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
            record = {
                "filename": download.suggested_filename,
                "path": str(destination),
                "url": sanitize_url(download.url),
            }
        except PlaywrightTimeoutError as exc:
            candidates = self._pdf_responses[pdf_cursor:]
            if not candidates:
                raise RuntimeError("点击后既未触发下载，也未收到浏览器内联 PDF 响应") from exc
            response = candidates[-1]
            parsed_name = Path(urlparse(response.url).path).name
            filename = parsed_name if parsed_name.lower().endswith(".pdf") else "browser-inline.pdf"
            destination = self.output_dir / "downloads" / filename
            destination.write_bytes(response.body())
            record = {
                "filename": filename,
                "path": str(destination),
                "url": sanitize_url(response.url),
                "inline_pdf": True,
            }
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
            "dom", sanitize_url(self._page.url), f"结构化提取 {kind}", {"kind": kind, "bid": bid, "data": data}
        )
        content_hash = hashlib.sha256(
            json.dumps(data, sort_keys=True, ensure_ascii=False, separators=(",", ":"), default=str).encode("utf-8")
        ).hexdigest()[:20]
        state = self._semantic_page_state()
        self._capture_step(f"extract:{kind}")
        return {
            "data": data,
            "evidence_id": evidence.evidence_id,
            "url": sanitize_url(self._page.url),
            "content_hash": content_hash,
            "semantic_page_fingerprint": state["semantic_page_fingerprint"],
        }

    async def extract_many(self, requests: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return await self._call(self._extract_many_sync, requests)

    def _extract_many_sync(self, requests: list[dict[str, Any]]) -> list[dict[str, Any]]:
        self._assert_thread()
        results: list[dict[str, Any]] = []
        for request in requests[:8]:
            results.append(
                self._extract_sync(
                    str(request["kind"]),
                    request.get("bid"),
                    min(max(int(request.get("limit", 1_000)), 1), 5_000),
                )
            )
        self._capture_step("extract_many")
        return results

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
        path = self.output_dir / "trajectory_visual" / f"visual-{self._visual_counter:03d}.png"
        self._visual_counter += 1
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
        output_prefix = self.output_dir / "trajectory_visual" / (
            f"document-{self._visual_counter:03d}-page-{page_number:03d}"
        )
        self._visual_counter += 1
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
        return await self._call(self._capture_step, label, label in {"finish", "record_coverage"})

    def _element_state(self, locator: Any) -> dict[str, Any]:
        return locator.evaluate(
            "el => ({bid:el.getAttribute('bid'),tag:el.tagName.toLowerCase(),type:el.type||'',text:(el.innerText||el.textContent||'').trim().slice(0,300),value:'value' in el?(el.type==='password'?'[REDACTED]':String(el.value||'')):'',checked:'checked' in el?Boolean(el.checked):null,selected:'selected' in el?Boolean(el.selected):null})"
        )

    def _action_sync(self, action: str, operation: Callable[[], dict[str, Any]], bid: str | None) -> dict[str, Any]:
        self._assert_thread()
        self._action_counter += 1
        action_id = f"act-{self._action_counter:04d}"
        try:
            self._ensure_live()
            before_url = self._page.url
            before_hash = self._last_dom_hash or ""
            before_semantic = self._semantic_page_state()
        except Exception as exc:
            receipt = ActionReceipt(
                action_id,
                action,
                False,
                "",
                "",
                "",
                "",
                {},
                error=sanitize_exception(exc),
            )
            return receipt.to_dict()
        dialogs_before = len(self._dialogs)
        pages_before = len(self._context.pages)
        network_before = len(self._network_records)
        stale = False
        error: str | None = None
        postconditions: dict[str, Any] = {}
        settled_state: dict[str, Any] | None = None
        after_semantic = before_semantic
        try:
            postconditions.update(operation())
            settled_state = self._settle()
            pages = [page for page in self._context.pages if not page.is_closed()]
            if len(pages) > pages_before:
                self._page = pages[-1]
                self._page.bring_to_front()
                self._refresh_cdp()
                settled_state = self._settle()
            success = True
        except Exception as exc:
            stale = "STALE_BID" in str(exc)
            error = sanitize_exception(exc)
            success = False
        try:
            after_url = self._page.url
            after_hash = before_hash
            after_semantic = settled_state or self._semantic_page_state()
            target = postconditions.get("target", {})
            target_text = " ".join(str(target.get(key, "")) for key in ("type", "text", "value"))
            may_submit = (
                action == "press" and str(postconditions.get("key", "")).lower() in {"enter", "return"}
            ) or (
                action == "click"
                and (target.get("type") == "submit" or bool(_SUBMIT_TARGET.search(target_text)))
            )
            page_text = (
                self._page.locator("body").inner_text(timeout=2_000)[:50_000]
                if may_submit
                else ""
            )
        except Exception as exc:
            after_url, after_hash, page_text = before_url, before_hash, ""
            after_semantic = before_semantic
            self._connected = False
            success = False
            error = error or sanitize_exception(exc)
        postconditions.update(
            {
                "bid": bid,
                "dialog_events": self._dialogs[dialogs_before:],
                "new_tab_count": max(0, len(self._context.pages) - pages_before) if self._context else 0,
                "network_response_count": max(0, len(self._network_records) - network_before),
                "confirmation": self._is_confirmed_submission(
                    action,
                    success,
                    before_url != after_url
                    or before_semantic["semantic_page_fingerprint"]
                    != after_semantic["semantic_page_fingerprint"],
                    postconditions,
                    page_text,
                ),
                "before_semantic_page_fingerprint": before_semantic["semantic_page_fingerprint"],
                "after_semantic_page_fingerprint": after_semantic["semantic_page_fingerprint"],
                "post_observation": {
                    "url": sanitize_url(after_url),
                    "title": after_semantic["title"],
                    "semantic_page_fingerprint": after_semantic["semantic_page_fingerprint"],
                    "semantic_element_count": after_semantic["semantic_element_count"],
                },
            }
        )
        public_postconditions = redact_value(postconditions)
        receipt_evidence = self.evidence_store.add(
            "receipt",
            sanitize_url(after_url),
            f"{action} {'成功' if success else '失败'}",
            {"action_id": action_id, "postconditions": public_postconditions, "error": error},
        )
        receipt = ActionReceipt(
            action_id,
            action,
            success,
            sanitize_url(before_url),
            sanitize_url(after_url),
            before_hash,
            after_hash,
            public_postconditions,
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
