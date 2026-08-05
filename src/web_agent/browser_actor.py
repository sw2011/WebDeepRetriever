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
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, TypeVar
from urllib.parse import parse_qs, parse_qsl, urlencode, urljoin, urlparse, urlsplit, urlunsplit

from pypdf import PdfReader
from playwright.sync_api import TimeoutError as PlaywrightTimeoutError

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

_PRE_DISPATCH_ACTIONABILITY = (
    "element is not visible",
    "element is not enabled",
    "element is not stable",
    "element is outside of the viewport",
    "intercepts pointer events",
    "element was detached from the dom",
    "waiting for element to be visible",
    "waiting for element to be enabled",
    "waiting for element to receive pointer events",
)
_POST_DISPATCH_CALL_LOG = (
    "performing click action",
    "click action done",
    "waiting for scheduled navigations",
)


def _is_pre_dispatch_actionability_timeout(exc: Exception) -> bool:
    if not isinstance(exc, PlaywrightTimeoutError):
        return False
    message = str(exc).casefold()
    return not any(marker in message for marker in _POST_DISPATCH_CALL_LOG) and any(
        marker in message for marker in _PRE_DISPATCH_ACTIONABILITY
    )


class BrowserActorPoisonedError(RuntimeError):
    pass


class ActorCallDeadlineExceeded(TimeoutError):
    def __init__(
        self,
        operation: str,
        *,
        dispatched: bool,
        task_generation: int = 0,
        attempt: int = 0,
    ) -> None:
        self.operation = operation
        self.dispatched = dispatched
        self.task_generation = task_generation
        self.attempt = attempt
        state = "DISPATCHED_TERMINAL_UNCERTAIN" if dispatched else "NOT_DISPATCHED_SAFE_FAILURE"
        super().__init__(f"ACTOR_DEADLINE_EXCEEDED_{state}: {operation}")


class _CallDispatchState:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._dispatched = False
        self._cancelled = False

    def begin(self, deadline: float | None) -> bool:
        with self._lock:
            if self._cancelled or (deadline is not None and time.monotonic() >= deadline):
                self._cancelled = True
                return False
            self._dispatched = True
            return True

    def can_prepare(self, deadline: float | None) -> bool:
        with self._lock:
            if self._cancelled or (deadline is not None and time.monotonic() >= deadline):
                self._cancelled = True
                return False
            return True

    def retract(self) -> None:
        with self._lock:
            if not self._cancelled:
                self._dispatched = False

    def cancel_if_queued(self) -> bool:
        with self._lock:
            if self._dispatched:
                return False
            self._cancelled = True
            return True


@dataclass(frozen=True)
class _ScheduledCall:
    future: asyncio.Future[Any]
    concurrent_future: Any
    generation: int
    attempt: int
    dispatch_state: _CallDispatchState
    mutation_state: _CallDispatchState | None
    operation: str
    deadline: float | None

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
  const scroll = [];
  const seen = new Set();
  const interactive = new Set([
    'A','BUTTON','INPUT','SELECT','TEXTAREA','OPTION','SUMMARY','DETAILS',
    'VIDEO','AUDIO','IFRAME'
  ]);
  const semantic = new Set([
    'H1','H2','H3','H4','H5','H6','P','LI','DT','DD','TH','TD','TR','TABLE',
    'LABEL','IMG','CANVAS','SVG','PRE','CODE','BLOCKQUOTE','ARTICLE','SECTION'
  ]);
  const contextTags = new Set([
    'ARTICLE','ASIDE','DIALOG','FIELDSET','FIGURE','FOOTER','FORM','HEADER','LI',
    'MAIN','NAV','SECTION','TD','TH','TR'
  ]);
  const contextRoles = new Set([
    'alertdialog','dialog','form','group','list','listbox','main','menu','navigation',
    'region','row','search','table'
  ]);
  const compactText = (value, limit) => String(value || '').replace(/\s+/g, ' ').trim().slice(0, limit);
  function parentOf(el) {
    if (el.parentElement) return el.parentElement;
    const root = el.getRootNode && el.getRootNode();
    return root && root.host ? root.host : null;
  }
  function isHidden(el) {
    let current = el;
    while (current) {
      const style = getComputedStyle(current);
      if (current.hidden || current.getAttribute('aria-hidden') === 'true' ||
          style.display === 'none' || style.visibility === 'hidden') return true;
      current = parentOf(current);
    }
    return false;
  }
  function parentContext(el) {
    const result = [];
    let parent = parentOf(el);
    while (parent && result.length < 2) {
      const tag = parent.tagName || '';
      const role = (parent.getAttribute && parent.getAttribute('role')) || '';
      if (contextTags.has(tag) || contextRoles.has(role)) {
        let label = parent.getAttribute('aria-label') || parent.getAttribute('title') || '';
        if (!label && parent.children) {
          const descriptor = Array.from(parent.children).find(child =>
            child.tagName === 'LEGEND' || /^H[1-6]$/.test(child.tagName)
          );
          if (descriptor) label = descriptor.textContent || '';
        }
        if (!label && ['LI','TD','TH','TR'].includes(tag)) {
          label = Array.from(parent.childNodes)
            .filter(node => node.nodeType === Node.TEXT_NODE)
            .map(node => node.textContent || '').join(' ');
        }
        const kind = role || tag.toLowerCase();
        const compactLabel = compactText(label, 120);
        result.push(compactLabel ? `${kind}:${compactLabel}` : kind);
      }
      parent = parentOf(parent);
    }
    return result;
  }
  function scrollState(kind, el) {
    const position = Math.max(0, Math.round(el.scrollTop || 0));
    const viewport = Math.max(0, Math.round(kind === 'page' ? window.innerHeight : el.clientHeight));
    const extent = Math.max(viewport, Math.round(el.scrollHeight || 0));
    const remaining = Math.max(0, extent - viewport - position);
    const state = {
      kind, frame: frameIndex, top: position <= 1, bottom: remaining <= 1,
      position, remaining, viewport, extent,
    };
    if (kind === 'container') {
      state.bid = el.getAttribute('bid') || '';
      state.tag = el.tagName.toLowerCase();
      state.role = el.getAttribute('role') || '';
      state.label = compactText(
        el.getAttribute('aria-label') || el.getAttribute('title') || el.getAttribute('name') || '',
        120,
      );
      const rect = el.getBoundingClientRect();
      state.visible = rect.width > 0 && rect.height > 0 && rect.bottom > 0 && rect.right > 0 &&
        rect.top < window.innerHeight && rect.left < window.innerWidth;
    }
    return state;
  }
  const pageScroller = document.scrollingElement || document.documentElement;
  scroll.push(scrollState('page', pageScroller));
  function walk(root) {
    if (!root || output.length >= maxElements) return;
    const nodes = root.querySelectorAll ? root.querySelectorAll('*') : [];
    for (const el of nodes) {
      if (output.length >= maxElements) break;
      if (seen.has(el)) continue;
      seen.add(el);
      const rect = el.getBoundingClientRect();
      const style = getComputedStyle(el);
      const role = el.getAttribute('role') || '';
      const hasText = Boolean((el.innerText || el.textContent || '').trim());
      const meaningful = interactive.has(el.tagName) || semantic.has(el.tagName) ||
        Boolean(role) || el.tabIndex >= 0 || el.isContentEditable ||
        (hasText && el.children.length === 0);
      if (isHidden(el)) continue;
      const bid = el.getAttribute('bid');
      if (!bid) continue;
      if (scroll.length < 21 && el.scrollHeight > el.clientHeight + 1 &&
          ['auto','scroll'].includes(style.overflowY)) {
        scroll.push(scrollState('container', el));
      }
      if (!meaningful) {
        if (el.shadowRoot) walk(el.shadowRoot);
        continue;
      }
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
        expanded: el.hasAttribute('aria-expanded') ? el.getAttribute('aria-expanded') === 'true' : null,
        disabled: Boolean(el.disabled || el.getAttribute('aria-disabled') === 'true'),
        href: el.href || '', type: el.type || '', name: el.name || '',
        context: parentContext(el),
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
      if (el.shadowRoot) walk(el.shadowRoot);
    }
  }
  walk(document);
  return {
    documentId: String(performance.timeOrigin),
    elements: output,
    scroll,
    truncated: output.length >= maxElements,
  };
}
"""


_OBSERVATION_CHANGE_FIELDS = (
    "tag",
    "role",
    "label",
    "text",
    "value",
    "checked",
    "selected",
    "expanded",
    "disabled",
    "href",
    "type",
    "name",
    "options",
    "context",
)


def _observation_element_identity(element: dict[str, Any]) -> str:
    return "|".join(
        (
            str(element.get("document_id") or element.get("frame_url", "")),
            str(element.get("bid", "")),
        )
    )


def _observation_element_signature(element: dict[str, Any]) -> str:
    projected = {
        key: element[key]
        for key in _OBSERVATION_CHANGE_FIELDS
        if element.get(key) not in (None, "", [], {})
    }
    return hashlib.sha256(
        json.dumps(
            projected,
            sort_keys=True,
            ensure_ascii=False,
            separators=(",", ":"),
            default=str,
        ).encode("utf-8", errors="replace")
    ).hexdigest()[:16]


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
        self._scheduling_lock = asyncio.Lock()
        self._transition_pending = False
        self._state_lock = threading.Lock()
        self._thread_context = threading.local()
        self._owner_thread: int | None = None
        self._playwright: Any = None
        self._browser: Any = None
        self._context: Any = None
        self._page: Any = None
        self._cdp: Any = None
        self._closed = False
        self._poisoned_reason: str | None = None
        self._generation = 0
        self._attempt_counter = 0
        self._invalid_attempts: set[tuple[int, int]] = set()
        self._attempt_windows: dict[tuple[int, int], dict[str, float | None]] = {}
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
        self._next_dialog_action: tuple[str, str | None, int] = ("dismiss", None, self._generation)
        self._new_pages: list[dict[str, Any]] = []
        self._attached_pages: set[int] = set()
        self._pdf_responses: list[Any] = []
        self._last_dom_hash: str | None = None
        self._last_semantic_state: dict[str, Any] | None = None
        self._last_capture_semantic: str | None = None
        self._last_capture_path: str | None = None
        self._last_observation_semantic: str | None = None
        self._last_observation_scope: str | None = None
        self._last_observed_elements: dict[str, tuple[str, str]] = {}
        self._frame_document_scopes: dict[str, str] = {}
        self._bind_evidence_store(self.evidence_store, self._generation)

    @property
    def poisoned(self) -> bool:
        with self._state_lock:
            return self._poisoned_reason is not None

    @property
    def poisoned_reason(self) -> str | None:
        with self._state_lock:
            return self._poisoned_reason

    @property
    def task_generation(self) -> int:
        with self._state_lock:
            return self._generation

    def _current_attempt(self) -> int:
        return int(getattr(self._thread_context, "attempt", self._attempt_counter))

    def _binding(self) -> tuple[int, int]:
        return (
            int(getattr(self._thread_context, "generation", self.task_generation)),
            self._current_attempt(),
        )

    def _reserve_generation(self) -> int:
        with self._state_lock:
            if self._closed:
                raise RuntimeError("BrowserActor 已关闭")
            if self._poisoned_reason is not None:
                raise BrowserActorPoisonedError(self._poisoned_reason)
            self._generation += 1
            self._attempt_counter = 0
            self._invalid_attempts.clear()
            self._attempt_windows.clear()
            return self._generation

    def _reserve_attempt(self) -> tuple[int, int]:
        with self._state_lock:
            if self._closed:
                raise RuntimeError("BrowserActor 已关闭")
            if self._poisoned_reason is not None:
                raise BrowserActorPoisonedError(self._poisoned_reason)
            self._attempt_counter += 1
            return self._generation, self._attempt_counter

    def _mark_poisoned(self, generation: int, reason: str) -> None:
        with self._state_lock:
            if generation == self._generation and self._poisoned_reason is None:
                self._poisoned_reason = reason

    def _assert_publishable(
        self,
        generation: int | None = None,
        attempt: int | None = None,
    ) -> None:
        bound_generation, bound_attempt = self._binding()
        expected_generation = bound_generation if generation is None else generation
        expected_attempt = bound_attempt if attempt is None else attempt
        with self._state_lock:
            if expected_generation != self._generation:
                raise BrowserActorPoisonedError("STALE_TASK_GENERATION: 迟到结果已隔离")
            if (expected_generation, expected_attempt) in self._invalid_attempts:
                raise BrowserActorPoisonedError("STALE_TASK_ATTEMPT: 迟到结果已隔离")
            if self._poisoned_reason is not None:
                raise BrowserActorPoisonedError(self._poisoned_reason)

    def _bind_evidence_store(self, store: EvidenceStore, generation: int) -> None:
        store.bind(
            generation,
            self._current_attempt,
            lambda operation, expected_generation=generation: self._publish_evidence(
                expected_generation,
                operation,
            ),
        )

    def _publish_evidence(self, expected_generation: int, operation: Callable[[], T]) -> T:
        generation, attempt = self._binding()
        with self._state_lock:
            if expected_generation != generation or generation != self._generation:
                raise BrowserActorPoisonedError("STALE_TASK_GENERATION: 迟到证据已隔离")
            if (generation, attempt) in self._invalid_attempts:
                raise BrowserActorPoisonedError("STALE_TASK_ATTEMPT: 迟到证据已隔离")
            if self._poisoned_reason is not None:
                raise BrowserActorPoisonedError(self._poisoned_reason)
            if self._closed:
                raise BrowserActorPoisonedError("ACTOR_CLOSED: 证据写入已拒绝")
            return operation()

    def _publish_event(
        self,
        generation: int,
        operation: Callable[[], None],
        attempt: int | None = None,
    ) -> bool:
        expected_attempt = self._binding()[1] if attempt is None else attempt
        with self._state_lock:
            if (
                generation != self._generation
                or (generation, expected_attempt) in self._invalid_attempts
                or self._poisoned_reason is not None
                or self._closed
            ):
                return False
            operation()
            return True

    def _invalidate_attempt(self, generation: int, attempt: int, poison_reason: str | None = None) -> None:
        with self._state_lock:
            self._invalid_attempts.add((generation, attempt))
            if poison_reason is not None and generation == self._generation and self._poisoned_reason is None:
                self._poisoned_reason = poison_reason
            self._network_records = [
                item
                for item in self._network_records
                if (item.get("task_generation"), item.get("attempt")) != (generation, attempt)
            ]
            self._dialogs = [
                item
                for item in self._dialogs
                if (item.get("task_generation"), item.get("attempt")) != (generation, attempt)
            ]
            self._downloads = [
                item
                for item in self._downloads
                if (item.get("task_generation"), item.get("attempt")) != (generation, attempt)
            ]
            self._new_pages = [
                item
                for item in self._new_pages
                if (item.get("task_generation"), item.get("attempt")) != (generation, attempt)
            ]
            self._pdf_responses = [
                item
                for item in self._pdf_responses
                if (item.get("task_generation"), item.get("attempt")) != (generation, attempt)
            ]
        self.evidence_store.discard_attempt(generation, attempt)

    def _event_is_current(self, generation: int) -> bool:
        with self._state_lock:
            return generation == self._generation and self._poisoned_reason is None and not self._closed

    def _execute_call(
        self,
        generation: int,
        attempt: int,
        dispatch_state: _CallDispatchState,
        mutation_state: _CallDispatchState | None,
        deadline: float | None,
        operation: str,
        function: Callable[..., T],
        args: tuple[Any, ...],
        allow_poisoned: bool,
    ) -> T:
        with self._state_lock:
            if generation != self._generation and not allow_poisoned:
                raise BrowserActorPoisonedError("STALE_TASK_GENERATION: 调用未派发")
            if self._poisoned_reason is not None and not allow_poisoned:
                raise BrowserActorPoisonedError(self._poisoned_reason)
        if not dispatch_state.begin(deadline):
            raise ActorCallDeadlineExceeded(
                operation,
                dispatched=False,
                task_generation=generation,
                attempt=attempt,
            )
        self._thread_context.generation = generation
        self._thread_context.attempt = attempt
        self._thread_context.mutation_state = mutation_state
        self._thread_context.deadline = deadline
        self._thread_context.operation = operation
        self._thread_context.mutation_dispatched = False
        with self._state_lock:
            self._attempt_windows[(generation, attempt)] = {
                "started_ms": time.time() * 1_000,
                "ended_ms": None,
                "mutation_dispatched_ms": None,
            }
        try:
            result = function(*args)
            if not allow_poisoned:
                self._assert_publishable(generation, attempt)
            if isinstance(result, dict):
                result.setdefault("task_generation", generation)
                result.setdefault("attempt", attempt)
            return result
        finally:
            with self._state_lock:
                window = self._attempt_windows.get((generation, attempt))
                if window is not None:
                    window["ended_ms"] = time.time() * 1_000
            self._thread_context.generation = generation
            self._thread_context.attempt = 0
            self._thread_context.mutation_state = None
            self._thread_context.deadline = None
            self._thread_context.mutation_dispatched = False

    def _schedule_call(
        self,
        function: Callable[..., T],
        *args: Any,
        deadline: float | None = None,
        operation: str | None = None,
        allow_poisoned: bool = False,
        mutation_aware: bool = False,
    ) -> _ScheduledCall:
        if allow_poisoned:
            with self._state_lock:
                generation = self._generation
                self._attempt_counter += 1
                attempt = self._attempt_counter
        else:
            generation, attempt = self._reserve_attempt()
        dispatch_state = _CallDispatchState()
        mutation_state = _CallDispatchState() if mutation_aware else None
        operation_name = operation or getattr(function, "__name__", "call")
        concurrent_future = self._executor.submit(
            self._execute_call,
            generation,
            attempt,
            dispatch_state,
            mutation_state,
            deadline,
            operation_name,
            function,
            args,
            allow_poisoned,
        )
        future = asyncio.wrap_future(concurrent_future)
        return _ScheduledCall(
            future=future,
            concurrent_future=concurrent_future,
            generation=generation,
            attempt=attempt,
            dispatch_state=dispatch_state,
            mutation_state=mutation_state,
            operation=operation_name,
            deadline=deadline,
        )

    async def _await_scheduled_call(self, call: _ScheduledCall) -> T:
        future = call.future

        def consume_late_result(done: asyncio.Future[Any]) -> None:
            try:
                done.exception()
            except (asyncio.CancelledError, Exception):
                pass

        timeout = max(0.0, call.deadline - time.monotonic()) if call.deadline is not None else None
        try:
            return await asyncio.wait_for(asyncio.shield(future), timeout=timeout)
        except asyncio.TimeoutError:
            if future.done():
                return future.result()
            terminal_state = call.mutation_state or call.dispatch_state
            was_dispatched = not terminal_state.cancel_if_queued()
            poison_reason = (
                f"ACTOR_POISONED: {call.operation} 超时且终态不确定"
                if was_dispatched
                else None
            )
            self._invalidate_attempt(call.generation, call.attempt, poison_reason)
            if was_dispatched:
                future.add_done_callback(consume_late_result)
            else:
                cancelled = call.concurrent_future.cancel()
                if not cancelled:
                    future.add_done_callback(consume_late_result)
            raise ActorCallDeadlineExceeded(
                call.operation,
                dispatched=was_dispatched,
                task_generation=call.generation,
                attempt=call.attempt,
            ) from None
        except asyncio.CancelledError:
            terminal_state = call.mutation_state or call.dispatch_state
            was_dispatched = not terminal_state.cancel_if_queued()
            poison_reason = (
                f"ACTOR_POISONED: {call.operation} 被取消且终态不确定"
                if was_dispatched
                else None
            )
            self._invalidate_attempt(call.generation, call.attempt, poison_reason)
            if was_dispatched:
                future.add_done_callback(consume_late_result)
            else:
                cancelled = call.concurrent_future.cancel()
                if not cancelled:
                    future.add_done_callback(consume_late_result)
            raise

    async def _call(
        self,
        function: Callable[..., T],
        *args: Any,
        deadline: float | None = None,
        operation: str | None = None,
        allow_poisoned: bool = False,
        mutation_aware: bool = False,
    ) -> T:
        if self._transition_pending and not allow_poisoned:
            raise BrowserActorPoisonedError("STALE_TASK_TRANSITION: 任务切换期间拒绝新动作")
        async with self._scheduling_lock:
            if self._transition_pending and not allow_poisoned:
                raise BrowserActorPoisonedError("STALE_TASK_TRANSITION: 任务切换期间拒绝新动作")
            call = self._schedule_call(
                function,
                *args,
                deadline=deadline,
                operation=operation,
                allow_poisoned=allow_poisoned,
                mutation_aware=mutation_aware,
            )
        return await self._await_scheduled_call(call)

    async def _transition_task(
        self,
        function: Callable[..., T],
        *args: Any,
        deadline: float,
        operation: str,
        evidence_store: EvidenceStore,
    ) -> T:
        if self._transition_pending:
            raise BrowserActorPoisonedError("STALE_TASK_TRANSITION: 已有任务切换正在进行")
        self._transition_pending = True
        try:
            async with self._scheduling_lock:
                barrier = self._schedule_call(
                    lambda: None,
                    deadline=deadline,
                    operation=f"{operation}_barrier",
                )
                await self._await_scheduled_call(barrier)
                generation = self._reserve_generation()
                self._bind_evidence_store(evidence_store, generation)
                call = self._schedule_call(
                    function,
                    *args,
                    deadline=deadline,
                    operation=operation,
                )
            return await self._await_scheduled_call(call)
        finally:
            self._transition_pending = False

    def _assert_thread(self) -> None:
        current = threading.get_ident()
        if self._owner_thread is None:
            self._owner_thread = current
        elif self._owner_thread != current:
            raise RuntimeError("Playwright 操作越过 BrowserActor 线程边界")

    async def start(self, initial_url: str) -> dict[str, Any]:
        return await self._transition_task(
            self._start_sync,
            initial_url,
            deadline=time.monotonic() + 100.0,
            operation="start",
            evidence_store=self.evidence_store,
        )

    async def begin_task(
        self,
        initial_url: str,
        output_dir: Path,
        evidence_store: EvidenceStore,
    ) -> dict[str, Any]:
        return await self._transition_task(
            self._begin_task_sync,
            initial_url,
            Path(output_dir),
            evidence_store,
            deadline=time.monotonic() + 100.0,
            operation="begin_task",
            evidence_store=evidence_store,
        )

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
        generation, _ = self._binding()
        self._browser.on("disconnected", self._on_disconnected)
        self._context = self._browser.contexts[0] if self._browser.contexts else self._browser.new_context()
        self._context.on("page", lambda page: self._on_page(page, generation))
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
        generation, _ = self._binding()
        self._bind_evidence_store(self.evidence_store, generation)
        if self._context is not None:
            self._context.on("page", lambda page: self._on_page(page, generation))
        self._attached_pages = set()
        self._bid_counter = 1
        self._step_counter = 0
        self._action_counter = 0
        self._observation_counter = 0
        self._visual_counter = 0
        self._network_cursor = 0
        self._network_records = []
        self._dialogs = []
        self._downloads = []
        self._next_dialog_action = ("dismiss", None, generation)
        self._new_pages = []
        self._pdf_responses = []
        self._last_dom_hash = None
        self._last_semantic_state = None
        self._last_capture_semantic = None
        self._last_capture_path = None
        self._last_observation_semantic = None
        self._last_observation_scope = None
        self._last_observed_elements = {}
        self._frame_document_scopes = {}
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
        generation, _ = self._binding()
        self._attach_page(self._page, generation)
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

    def _attach_page(self, page: Any, generation: int | None = None) -> None:
        generation = self.task_generation if generation is None else generation
        if not self._event_is_current(generation):
            return
        identity = id(page)
        if identity in self._attached_pages:
            return
        self._attached_pages.add(identity)
        page.set_default_timeout(self.click_timeout_ms)
        page.on("response", lambda response: self._on_response(response, generation))
        page.on("dialog", lambda dialog: self._on_dialog(dialog, generation))
        page.on("crash", lambda: self._on_page_crash(generation))

    def _on_disconnected(self, generation: int | None = None) -> None:
        generation = self.task_generation if generation is None else generation
        with self._state_lock:
            if generation != self._generation or self._poisoned_reason is not None or self._closed:
                return
            self._connected = False
            self._poisoned_reason = "ACTOR_POISONED: CDP_DISCONNECTED"

    def _on_page_crash(self, generation: int) -> None:
        with self._state_lock:
            if generation != self._generation or self._poisoned_reason is not None or self._closed:
                return
            self._connected = False
            self._poisoned_reason = "ACTOR_POISONED: PAGE_CRASHED"

    def _on_page(self, page: Any, generation: int | None = None) -> None:
        self._assert_thread()
        generation = self._binding()[0] if generation is None else generation
        if not self._event_is_current(generation):
            return
        self._attach_page(page, generation)
        task_generation, attempt = self._binding()
        record = {
            "url": sanitize_url(page.url),
            "task_generation": task_generation,
            "attempt": attempt,
        }
        self._publish_event(
            generation,
            lambda: self._new_pages.append(record),
        )

    def _on_dialog(self, dialog: Any, generation: int | None = None) -> None:
        self._assert_thread()
        generation = self._binding()[0] if generation is None else generation
        action, prompt_text, armed_generation = self._next_dialog_action
        if not self._event_is_current(generation):
            try:
                dialog.dismiss()
            except Exception:
                pass
            return
        if armed_generation != generation:
            action, prompt_text = "dismiss", None
        task_generation, attempt = self._binding()
        record = {
            "type": dialog.type,
            "message": redact_text(dialog.message),
            "action": action,
            "url": sanitize_url(self._page.url),
            "task_generation": task_generation,
            "attempt": attempt,
        }
        try:
            if action == "accept":
                dialog.accept(prompt_text=prompt_text)
            else:
                dialog.dismiss()
        except Exception as exc:
            record["error"] = _error_type(exc)
            self._publish_event(generation, lambda: self._dialogs.append(record), attempt)
            self._next_dialog_action = ("dismiss", None, generation)
            reason = f"ACTOR_POISONED: dialog {action} 异常且终态不确定"
            self._mark_poisoned(task_generation, reason)
            raise BrowserActorPoisonedError(reason) from exc
        self._publish_event(generation, lambda: self._dialogs.append(record), attempt)
        self._next_dialog_action = ("dismiss", None, generation)

    def _on_response(self, response: Any, generation: int | None = None) -> None:
        self._assert_thread()
        generation = self._binding()[0] if generation is None else generation
        if not self._event_is_current(generation):
            return
        try:
            self._capture_response(response, generation)
        except Exception as exc:
            task_generation, attempt = self._binding()
            record = {
                "capture_error": _error_type(exc),
                "task_generation": task_generation,
                "attempt": attempt,
            }
            self._publish_event(
                generation,
                lambda: self._network_records.append(record)
                if len(self._network_records) < self.max_network_records
                else None,
            )

    def _capture_response(self, response: Any, generation: int) -> None:
        content_type = response.headers.get("content-type", "").lower()
        is_pdf = "application/pdf" in content_type
        request = response.request
        task_generation, attempt, request_started_ms, mutation_correlated = self._request_binding(
            request,
            generation,
        )
        pdf_entry = {
            "response": response,
            "task_generation": task_generation,
            "attempt": attempt,
            "request_started_ms": request_started_ms,
            "mutation_correlated": mutation_correlated,
        }
        if request.resource_type not in {"xhr", "fetch"}:
            if is_pdf:
                self._publish_event(
                    generation,
                    lambda: self._pdf_responses.append(pdf_entry),
                    attempt,
                )
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
        record.update({"task_generation": task_generation, "attempt": attempt})

        def publish() -> None:
            if is_pdf:
                self._pdf_responses.append(pdf_entry)
            if len(self._network_records) < self.max_network_records:
                self._network_records.append(record)

        self._publish_event(generation, publish, attempt)

    def _request_binding(
        self,
        request: Any,
        generation: int,
    ) -> tuple[int, int, float | None, bool]:
        try:
            request_started_ms = float(request.timing.get("startTime"))
            if request_started_ms < 10_000_000_000:
                request_started_ms *= 1_000
        except (AttributeError, TypeError, ValueError):
            request_started_ms = None
        current_generation, current_attempt = self._binding()
        if request_started_ms is None:
            return (
                generation,
                current_attempt if generation == current_generation else 0,
                None,
                False,
            )
        with self._state_lock:
            candidates = [
                (attempt, window)
                for (window_generation, attempt), window in self._attempt_windows.items()
                if window_generation == generation
                and float(window["started_ms"] or 0) <= request_started_ms
                and (
                    window["ended_ms"] is None
                    or request_started_ms <= float(window["ended_ms"])
                )
            ]
        if not candidates:
            return generation, 0, request_started_ms, False
        attempt, window = max(candidates, key=lambda item: float(item[1]["started_ms"] or 0))
        mutation_started_ms = window.get("mutation_dispatched_ms")
        mutation_correlated = bool(
            mutation_started_ms is not None
            and request_started_ms >= float(mutation_started_ms)
        )
        return generation, attempt, request_started_ms, mutation_correlated

    @staticmethod
    def _response_matches_target(response: Any, target_url: str) -> bool:
        expected = urlsplit(target_url)._replace(fragment="").geturl()
        candidates = [getattr(response, "url", "")]
        request = getattr(response, "request", None)
        seen: set[int] = set()
        while request is not None and id(request) not in seen:
            seen.add(id(request))
            candidates.append(getattr(request, "url", ""))
            candidates.append(getattr(getattr(request, "frame", None), "url", ""))
            request = getattr(request, "redirected_from", None)
        return any(
            isinstance(candidate, str)
            and urlsplit(candidate)._replace(fragment="").geturl() == expected
            for candidate in candidates
        )

    def _ensure_live(self) -> None:
        self._assert_publishable()
        if not self._connected or self._page is None or self._page.is_closed():
            generation, _ = self._binding()
            self._mark_poisoned(generation, "ACTOR_POISONED: PAGE_OR_CDP_UNAVAILABLE")
            raise BrowserActorPoisonedError(self.poisoned_reason or "ACTOR_POISONED")

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
        return await self._call(
            self._observe_sync,
            deadline=time.monotonic() + 29.0,
            operation="observe",
        )

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
        scroll: list[dict[str, Any]] = []
        document_scope: str | None = None
        incomplete_document_scopes: set[str] = set()
        truncated = False
        browsergym_pre_extract(self._page, tags_to_mark="all", lenient=True)
        try:
            for frame_index, frame in enumerate(self._page.frames):
                frame_key = str(getattr(getattr(frame, "_impl_obj", None), "_guid", id(frame)))
                try:
                    result = frame.evaluate(
                        _MARK_AND_COLLECT,
                        {"frameIndex": frame_index, "maxElements": 4_000},
                    )
                except Exception as exc:
                    frame_url = sanitize_url(frame.url)
                    cached_scope = self._frame_document_scopes.get(frame_key)
                    if cached_scope:
                        incomplete_document_scopes.add(cached_scope)
                    elements.append(
                        {"frame": frame_index, "frame_error": _error_type(exc), "url": frame_url}
                    )
                    continue
                frame_url = sanitize_url(frame.url)
                document_id = f"{frame_key}:{result.get('documentId') or frame_url}"
                self._frame_document_scopes[frame_key] = document_id
                if frame_index == 0:
                    document_scope = document_id
                if result["truncated"]:
                    incomplete_document_scopes.add(document_id)
                    truncated = True
                for item in result["elements"]:
                    item["frame_url"] = frame_url
                    item["document_id"] = document_id
                for item in result.get("scroll", []):
                    item["frame_url"] = frame_url
                elements.extend(result["elements"])
                scroll.extend(result.get("scroll", []))
            self._annotate_observation_changes(
                elements,
                document_scope=document_scope or self._last_observation_scope or "unknown-document",
                incomplete_document_scopes=incomplete_document_scopes,
            )
            raw_snapshot, ax_tree = self._protocol_observation()
        finally:
            browsergym_post_extract(self._page)
        dom_hash = self._dom_hash()
        self._assert_publishable()
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
                    "scroll": scroll,
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
            "scroll": scroll,
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
            "scroll": scroll,
            "truncated": truncated,
            "evidence_id": evidence.evidence_id,
            "screenshot_path": screenshot_path,
        }

    def _annotate_observation_changes(
        self,
        elements: list[dict[str, Any]],
        *,
        document_scope: str,
        incomplete_document_scopes: set[str] | None = None,
    ) -> None:
        compare = self._last_observation_scope == document_scope
        current: dict[str, tuple[str, str]] = {}
        for element in elements:
            if not element.get("bid") or element.get("frame_error"):
                continue
            identity = _observation_element_identity(element)
            signature = _observation_element_signature(element)
            element_document_scope = str(element.get("document_id") or element.get("frame_url", ""))
            current[identity] = (signature, element_document_scope)
            if not compare:
                continue
            previous = self._last_observed_elements.get(identity)
            if previous is None:
                element["new"] = True
            elif previous[0] != signature:
                element["changed"] = True
        if compare and incomplete_document_scopes:
            for identity, previous in self._last_observed_elements.items():
                if identity not in current and previous[1] in incomplete_document_scopes:
                    current[identity] = previous
        self._last_observation_scope = document_scope
        self._last_observed_elements = current

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
        return await self._call(
            self._action_sync,
            "click",
            lambda: self._click_op(bid),
            bid,
            deadline=time.monotonic() + 11.0,
            operation="click",
            mutation_aware=True,
        )

    def _click_op(self, bid: str) -> dict[str, Any]:
        locator = self._locator(bid)
        target = self._element_state(locator)
        self._prepare_mutation(
            lambda: locator.click(timeout=self._playwright_timeout_ms(), trial=True)
        )
        self._dispatch_mutation(
            lambda: locator.click(timeout=self._playwright_timeout_ms()),
            safe_failure=_is_pre_dispatch_actionability_timeout,
        )
        return {"target": target}

    async def fill(self, bid: str, value: str) -> dict[str, Any]:
        return await self._call(
            self._action_sync,
            "fill",
            lambda: self._fill_op(bid, value),
            bid,
            deadline=time.monotonic() + 11.0,
            operation="fill",
            mutation_aware=True,
        )

    def _fill_op(self, bid: str, value: str) -> dict[str, Any]:
        locator = self._locator(bid)
        before = locator.input_value()
        self._dispatch_mutation(lambda: locator.fill(value, timeout=self.click_timeout_ms))
        is_password = (locator.get_attribute("type") or "").casefold() == "password"
        after = locator.input_value()
        return {
            "before_value": "[REDACTED]" if is_password else before,
            "value": "[REDACTED]" if is_password else after,
            "expected_value": "[REDACTED]" if is_password else value,
            "value_changed": before != after,
        }

    async def select(self, bid: str, values: list[str]) -> dict[str, Any]:
        return await self._call(
            self._action_sync,
            "select",
            lambda: self._select_op(bid, values),
            bid,
            deadline=time.monotonic() + 11.0,
            operation="select",
            mutation_aware=True,
        )

    def _select_op(self, bid: str, values: list[str]) -> dict[str, Any]:
        locator = self._locator(bid)
        before = locator.input_value()
        selected = self._dispatch_mutation(lambda: locator.select_option(values, timeout=self.click_timeout_ms))
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
            self._action_sync,
            "set_checked",
            lambda: self._set_checked_op(bid, checked),
            bid,
            deadline=time.monotonic() + 11.0,
            operation="set_checked",
            mutation_aware=True,
        )

    def _set_checked_op(self, bid: str, checked: bool) -> dict[str, Any]:
        locator = self._locator(bid)
        before = locator.is_checked()
        if before == checked:
            return {
                "before_checked": before,
                "checked": before,
                "expected_checked": checked,
                "value_changed": False,
            }
        self._prepare_mutation(
            lambda: locator.set_checked(checked, timeout=self._playwright_timeout_ms(), trial=True)
        )
        self._dispatch_mutation(
            lambda: locator.set_checked(checked, timeout=self._playwright_timeout_ms()),
            safe_failure=_is_pre_dispatch_actionability_timeout,
        )
        after = locator.is_checked()
        return {
            "before_checked": before,
            "checked": after,
            "expected_checked": checked,
            "value_changed": before != after,
        }

    async def press(self, key: str, bid: str | None = None) -> dict[str, Any]:
        return await self._call(
            self._action_sync,
            "press",
            lambda: self._press_op(key, bid),
            bid,
            deadline=time.monotonic() + 11.0,
            operation="press",
            mutation_aware=True,
        )

    def _press_op(self, key: str, bid: str | None) -> dict[str, Any]:
        if bid:
            locator = self._locator(bid)
            self._dispatch_mutation(lambda: locator.press(key, timeout=self.click_timeout_ms))
        else:
            self._dispatch_mutation(lambda: self._page.keyboard.press(key))
        return {"key": key, "bid": bid}

    async def scroll(self, delta_y: int, bid: str | None = None) -> dict[str, Any]:
        delta_y = min(max(int(delta_y), -4_000), 4_000)
        return await self._call(
            self._action_sync,
            "scroll",
            lambda: self._scroll_op(delta_y, bid),
            bid,
            deadline=time.monotonic() + 11.0,
            operation="scroll",
            mutation_aware=True,
        )

    def _scroll_op(self, delta_y: int, bid: str | None) -> dict[str, Any]:
        if bid:
            locator = self._locator(bid)
            result = self._dispatch_mutation(
                lambda: locator.evaluate(
                    "(el, dy) => { const before=el.scrollTop; el.scrollBy(0,dy); return {before,after:el.scrollTop,height:el.scrollHeight,client:el.clientHeight}; }",
                    delta_y,
                )
            )
        else:
            result = self._dispatch_mutation(
                lambda: self._page.evaluate(
                    "dy => { const before=scrollY; scrollBy(0,dy); return {before,after:scrollY,height:document.documentElement.scrollHeight,client:innerHeight}; }",
                    delta_y,
                )
            )
        return {"delta_y": delta_y, "bid": bid, **result}

    async def wait(self, milliseconds: int) -> dict[str, Any]:
        milliseconds = min(max(int(milliseconds), 0), 8_000)
        return await self._call(
            self._action_sync,
            "wait",
            lambda: self._wait_op(milliseconds),
            None,
            deadline=time.monotonic() + 11.0,
            operation="wait",
            mutation_aware=True,
        )

    def _wait_op(self, milliseconds: int) -> dict[str, Any]:
        self._dispatch_mutation(lambda: self._page.wait_for_timeout(milliseconds))
        return {"milliseconds": milliseconds}

    async def arm_dialog(self, action: str, prompt_text: str | None = None) -> dict[str, Any]:
        return await self._call(
            self._arm_dialog_sync,
            action,
            prompt_text,
            deadline=time.monotonic() + 4.0,
            operation="dialog",
        )

    def _arm_dialog_sync(self, action: str, prompt_text: str | None) -> dict[str, Any]:
        self._assert_thread()
        if action not in {"accept", "dismiss"}:
            raise ValueError("dialog action 仅支持 accept 或 dismiss")
        generation, _ = self._binding()
        self._next_dialog_action = (action, prompt_text, generation)
        self._capture_step("dialog:arm")
        return {"armed": True, "action": action}

    async def tabs(self, action: str, index: int | None = None, url: str | None = None) -> dict[str, Any]:
        return await self._call(
            self._tabs_sync,
            action,
            index,
            url,
            deadline=time.monotonic() + 11.0,
            operation="tabs",
            mutation_aware=action != "list",
        )

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
        try:
            if action == "switch":
                if index is None or index < 0 or index >= len(pages):
                    raise ValueError("tab index 越界")
                target = pages[index]

                def mutate() -> None:
                    self._page = target
                    self._page.bring_to_front()
                    self._refresh_cdp()

                self._dispatch_mutation(mutate)
            elif action == "close":
                if len(pages) == 1:
                    raise ValueError("不能关闭最后一个标签页")
                target = pages[index if index is not None else pages.index(self._page)]

                def mutate() -> None:
                    target.close(run_before_unload=False)
                    remaining = [page for page in self._context.pages if not page.is_closed()]
                    self._page = remaining[-1]
                    self._refresh_cdp()

                self._dispatch_mutation(mutate)
            elif action == "new":
                if not url:
                    raise ValueError("新建标签页必须提供 URL")
                if urlparse(url).scheme not in {"http", "https"}:
                    raise ValueError("新建标签页仅允许 http/https URL")
                target_url = canonical_url(url)
                existing = next((page for page in pages if canonical_url(page.url) == target_url), None)

                def mutate() -> bool:
                    if existing is not None:
                        self._page = existing
                        self._page.bring_to_front()
                        self._refresh_cdp()
                        return True
                    self._page = self._context.new_page()
                    self._page.goto(url, wait_until="domcontentloaded", timeout=8_000)
                    self._refresh_cdp()
                    return False

                reused = self._dispatch_mutation(mutate)
            else:
                raise ValueError("tabs action 仅支持 list/switch/close/new")
            state = self._settle()
            self._capture_step(f"tabs:{action}")
        except Exception as exc:
            if bool(getattr(self._thread_context, "mutation_dispatched", False)):
                generation, _ = self._binding()
                reason = f"ACTOR_POISONED: tabs:{action} 异常且终态不确定"
                self._mark_poisoned(generation, reason)
                raise BrowserActorPoisonedError(reason) from exc
            raise
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
        return await self._call(
            self._action_sync,
            "upload",
            lambda: self._upload_op(bid, paths),
            bid,
            deadline=time.monotonic() + 19.0,
            operation="upload",
            mutation_aware=True,
        )

    def _upload_op(self, bid: str, paths: list[str]) -> dict[str, Any]:
        resolved = [Path(path).expanduser().resolve() for path in paths]
        if not resolved or any(not path.is_file() for path in resolved):
            raise ValueError("上传文件不存在或不是普通文件")
        configured = [value for value in os.getenv("WEBRETRIEVER_UPLOAD_ROOTS", "").split(os.pathsep) if value]
        allowed_roots = [(self.output_dir.parent).resolve(), *(Path(value).expanduser().resolve() for value in configured)]
        if any(not any(path == root or root in path.parents for root in allowed_roots) for path in resolved):
            raise ValueError("上传路径不在任务输出父目录或 WEBRETRIEVER_UPLOAD_ROOTS 白名单中")
        locator = self._locator(bid)
        self._dispatch_mutation(lambda: locator.set_input_files([str(path) for path in resolved]))
        return {"files": [path.name for path in resolved]}

    async def download(self, bid: str) -> dict[str, Any]:
        return await self._call(
            self._action_sync,
            "download",
            lambda: self._download_op(bid),
            bid,
            deadline=time.monotonic() + 19.0,
            operation="download",
            mutation_aware=True,
        )

    def _download_op(self, bid: str) -> dict[str, Any]:
        pdf_cursor = len(self._pdf_responses)
        locator = self._locator(bid)
        href = locator.get_attribute("href")
        expected_url = urljoin(self._page.url, href) if href else None
        self._prepare_mutation(
            lambda: locator.click(timeout=self._playwright_timeout_ms(), trial=True)
        )
        try:
            with self._page.expect_download(timeout=self._playwright_timeout_ms()) as info:
                self._dispatch_mutation(
                    lambda: locator.click(timeout=self._playwright_timeout_ms()),
                    safe_failure=_is_pre_dispatch_actionability_timeout,
                )
            download = info.value
            destination = self.output_dir / "downloads" / download.suggested_filename
            download.save_as(str(destination))
            record = {
                "filename": download.suggested_filename,
                "path": str(destination),
                "url": sanitize_url(download.url),
            }
        except PlaywrightTimeoutError as exc:
            task_generation, attempt = self._binding()
            timed_candidates = [
                entry
                for entry in self._pdf_responses[pdf_cursor:]
                if (entry.get("task_generation"), entry.get("attempt"))
                == (task_generation, attempt)
                and entry.get("mutation_correlated") is True
            ]
            candidates = [
                entry
                for entry in timed_candidates
                if expected_url is not None
                and self._response_matches_target(entry["response"], expected_url)
            ]
            if not candidates:
                raise RuntimeError("点击后既未触发下载，也未收到浏览器内联 PDF 响应") from exc
            response: Any = None
            pdf_body: bytes | None = None
            for candidate in reversed(candidates):
                candidate_response = candidate["response"]
                try:
                    candidate_body = candidate_response.body()
                except Exception:
                    continue
                if b"%PDF-" in candidate_body[:1_024]:
                    response = candidate_response
                    pdf_body = candidate_body
                    break
            if response is None or pdf_body is None:
                candidate_urls = [
                    {
                        "response": sanitize_url(entry["response"].url),
                        "frame": sanitize_url(
                            str(getattr(getattr(entry["response"].request, "frame", None), "url", ""))
                        ),
                    }
                    for entry in timed_candidates
                ]
                raise RuntimeError(
                    f"点击后的关联响应不包含有效 PDF 文件签名；候选 URL: {candidate_urls}"
                ) from exc
            public_url = expected_url or response.url
            parsed_name = Path(urlparse(public_url).path).name
            filename = parsed_name if parsed_name.lower().endswith(".pdf") else "browser-inline.pdf"
            destination = self.output_dir / "downloads" / filename
            destination.write_bytes(pdf_body)
            record = {
                "filename": filename,
                "path": str(destination),
                "url": sanitize_url(public_url),
                "inline_pdf": True,
            }
        task_generation, attempt = self._binding()
        record.update({"task_generation": task_generation, "attempt": attempt})
        if not self._publish_event(task_generation, lambda: self._downloads.append(record), attempt):
            self._assert_publishable(task_generation, attempt)
        return {"download": record}

    async def extract(self, kind: str, bid: str | None = None, limit: int = 1_000) -> dict[str, Any]:
        return await self._call(
            self._extract_sync,
            kind,
            bid,
            min(max(limit, 1), 5_000),
            deadline=time.monotonic() + 19.0,
            operation="extract",
        )

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
        self._assert_publishable()
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
        return await self._call(
            self._extract_many_sync,
            requests,
            deadline=time.monotonic() + 29.0,
            operation="extract_many",
        )

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
        return await self._call(
            self._network_events_sync,
            since_last,
            deadline=time.monotonic() + 19.0,
            operation="network",
        )

    def _network_events_sync(self, since_last: bool) -> dict[str, Any]:
        self._assert_thread()
        start = self._network_cursor if since_last else 0
        records = self._network_records[start:]
        self._network_cursor = len(self._network_records)
        self._assert_publishable()
        evidence = self.evidence_store.add(
            "network",
            self._page.url,
            f"页面触发的 XHR/Fetch 响应 {len(records)} 条",
            {"records": records},
        )
        self._capture_step("network")
        return {"records": records, "evidence_id": evidence.evidence_id}

    async def extract_document(self, path: str) -> dict[str, Any]:
        return await self._call(
            self._extract_document_sync,
            path,
            deadline=time.monotonic() + 29.0,
            operation="document",
        )

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
        self._assert_publishable()
        evidence = self.evidence_store.add(
            "document", self._page.url, f"下载文档 {target.name}，{pages} 页", {"path": str(target), "pages": pages, "text": data}
        )
        self._capture_step("document")
        return {"path": str(target), "pages": pages, "text": data, "evidence_id": evidence.evidence_id}

    async def visual_crop(self, bid: str, question: str) -> dict[str, Any]:
        return await self._call(
            self._visual_crop_sync,
            bid,
            question,
            deadline=time.monotonic() + 60.0,
            operation="visual_crop",
        )

    def _visual_crop_sync(self, bid: str, question: str) -> dict[str, Any]:
        self._assert_thread()
        locator = self._locator(bid)
        path = self.output_dir / "trajectory_visual" / f"visual-{self._visual_counter:03d}.png"
        self._visual_counter += 1
        locator.screenshot(path=str(path), timeout=30_000)
        self._capture_step("visual:element")
        self._assert_publishable()
        evidence = self.evidence_store.add(
            "visual", self._page.url, f"局部视觉检查: {question}", {"path": str(path), "question": question, "analysis": None}
        )
        return {"path": str(path), "question": question, "evidence_id": evidence.evidence_id}

    async def render_document_page(self, path: str, page_number: int, question: str) -> dict[str, Any]:
        return await self._call(
            self._render_document_page_sync,
            path,
            page_number,
            question,
            deadline=time.monotonic() + 60.0,
            operation="visual_document",
        )

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
        self._assert_publishable()
        evidence = self.evidence_store.add(
            "visual",
            self._page.url,
            f"扫描 PDF 第 {page_number} 页局部视觉检查: {question}",
            {"path": str(output), "document": str(target), "page": page_number, "question": question, "analysis": None},
        )
        self._capture_step("visual:document")
        return {"path": str(output), "question": question, "evidence_id": evidence.evidence_id}

    async def audit_step(self, label: str) -> str:
        return await self._call(
            self._capture_step,
            label,
            label in {"finish", "record_coverage"},
            deadline=time.monotonic() + 4.0,
            operation="audit_step",
        )

    def _element_state(self, locator: Any) -> dict[str, Any]:
        return locator.evaluate(
            "el => ({bid:el.getAttribute('bid'),tag:el.tagName.toLowerCase(),type:el.type||'',text:(el.innerText||el.textContent||'').trim().slice(0,300),value:'value' in el?(el.type==='password'?'[REDACTED]':String(el.value||'')):'',checked:'checked' in el?Boolean(el.checked):null,selected:'selected' in el?Boolean(el.selected):null})"
        )

    def _prepare_mutation(self, operation: Callable[[], T]) -> T:
        mutation_state = getattr(self._thread_context, "mutation_state", None)
        deadline = getattr(self._thread_context, "deadline", None)
        operation_name = str(getattr(self._thread_context, "operation", "action"))
        generation, attempt = self._binding()
        if mutation_state is not None and not mutation_state.can_prepare(deadline):
            raise ActorCallDeadlineExceeded(
                operation_name,
                dispatched=False,
                task_generation=generation,
                attempt=attempt,
            )
        return operation()

    def _playwright_timeout_ms(self) -> int:
        deadline = getattr(self._thread_context, "deadline", None)
        if deadline is None:
            return self.click_timeout_ms
        remaining_ms = int((deadline - time.monotonic()) * 1_000) - 1_000
        return max(1, min(self.click_timeout_ms, remaining_ms))

    def _dispatch_mutation(
        self,
        operation: Callable[[], T],
        *,
        safe_failure: Callable[[Exception], bool] | None = None,
    ) -> T:
        mutation_state = getattr(self._thread_context, "mutation_state", None)
        deadline = getattr(self._thread_context, "deadline", None)
        operation_name = str(getattr(self._thread_context, "operation", "action"))
        generation, attempt = self._binding()
        if mutation_state is not None and not mutation_state.begin(deadline):
            raise ActorCallDeadlineExceeded(
                operation_name,
                dispatched=False,
                task_generation=generation,
                attempt=attempt,
            )
        dispatched_ms = time.time() * 1_000
        with self._state_lock:
            window = self._attempt_windows.get((generation, attempt))
            if window is not None:
                window["mutation_dispatched_ms"] = dispatched_ms
        self._thread_context.mutation_dispatched = True
        try:
            return operation()
        except Exception as exc:
            if safe_failure is None or not safe_failure(exc):
                raise
            if mutation_state is not None:
                mutation_state.retract()
            with self._state_lock:
                window = self._attempt_windows.get((generation, attempt))
                if window is not None:
                    window["mutation_dispatched_ms"] = None
            self._thread_context.mutation_dispatched = False
            raise

    def _action_sync(self, action: str, operation: Callable[[], dict[str, Any]], bid: str | None) -> dict[str, Any]:
        self._assert_thread()
        self._action_counter += 1
        action_id = f"act-{self._action_counter:04d}"
        task_generation, attempt = self._binding()
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
                task_generation=task_generation,
                attempt=attempt,
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
        self._thread_context.mutation_dispatched = False
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
            if bool(getattr(self._thread_context, "mutation_dispatched", False)):
                reason = f"ACTOR_POISONED: {action} 异常且终态不确定"
                self._mark_poisoned(task_generation, reason)
                raise BrowserActorPoisonedError(reason) from exc
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
            if bool(getattr(self._thread_context, "mutation_dispatched", False)):
                reason = f"ACTOR_POISONED: {action} 回执不完整且终态不确定"
                self._mark_poisoned(task_generation, reason)
                raise BrowserActorPoisonedError(reason) from exc
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
        self._assert_publishable(task_generation)
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
            task_generation=task_generation,
            attempt=attempt,
        )
        try:
            self._capture_step(action)
        except Exception:
            pass
        result = receipt.to_dict()
        if not success:
            result.update({"terminal_uncertain": False, "safe_to_retry": True})
        return result

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
            await self._call(
                self._close_sync,
                deadline=time.monotonic() + 5.0,
                operation="close",
                allow_poisoned=True,
            )
        except (ActorCallDeadlineExceeded, BrowserActorPoisonedError):
            pass
        finally:
            self._closed = True
            self._executor.shutdown(wait=False, cancel_futures=True)

    async def retire(self) -> None:
        """Drain prior calls and disconnect this actor without closing the external CDP browser."""
        if self._closed:
            return
        completed = False
        try:
            await self._call(
                self._retire_sync,
                operation="retire",
                allow_poisoned=True,
            )
            completed = True
        finally:
            self._closed = True
            self._executor.shutdown(wait=completed, cancel_futures=True)

    async def flush_artifacts(self) -> None:
        await self._call(
            self._flush_artifacts_sync,
            deadline=time.monotonic() + 10.0,
            operation="flush_artifacts",
            allow_poisoned=True,
        )

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

    def _retire_sync(self) -> None:
        self._assert_thread()
        try:
            self.output_dir.mkdir(parents=True, exist_ok=True)
            self._flush_artifacts_sync()
        except Exception:
            pass
        try:
            if self._page and not self._page.is_closed():
                self._page.close(run_before_unload=False)
        except Exception:
            pass
        try:
            if self._playwright:
                self._playwright.stop()
        except Exception:
            pass
        self._connected = False
