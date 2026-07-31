from __future__ import annotations

import asyncio
import hashlib
import threading
from types import SimpleNamespace

import pytest

from web_agent.browser_actor import (
    BrowserActor,
    _redact,
    _safe_post_data,
    _sanitize_headers,
    canonical_url,
    semantic_page_fingerprint,
)
from web_agent.evidence import EvidenceStore
from web_agent.sanitization import sanitize_error_text, sanitize_url


def test_sensitive_network_material_is_removed() -> None:
    assert _sanitize_headers(
        {
            "content-type": "application/json",
            "Authorization": "Bearer private",
            "Cookie": "session=private",
            "x-api-key": "private",
        }
    ) == {"content-type": "application/json"}
    assert _redact({"user": "alice", "password": "private", "nested": {"token": "private"}}) == {
        "user": "alice",
        "password": "[REDACTED]",
        "nested": {"token": "[REDACTED]"},
    }
    assert _redact({"message": "Bearer abc.def-123", "session": "private"}) == {
        "message": "[REDACTED]",
        "session": "[REDACTED]",
    }
    sanitized_url = sanitize_url(
        "https://example.test/data?query=public&sess=private&token=secret#authorization=Bearer%20private"
    )
    assert "query=public" in sanitized_url
    assert "private" not in sanitized_url
    assert "secret" not in sanitized_url
    alias_url = sanitize_url(
        "https://user:password@example.test/data?code=private&sig=private&key=private&auth=private"
    )
    assert "user" not in alias_url
    assert "password" not in alias_url
    assert "private" not in alias_url
    sanitized_error = sanitize_error_text(
        "connect http://127.0.0.1:9?access_token=private "
        "org-privatevalue proj-privatevalue ak-private-private"
    )
    assert "private" not in sanitized_error


def test_binary_post_data_is_fingerprinted_without_decoding() -> None:
    raw = b"\x1f\x8b\x08\x00\xff\x00\x80"
    value = _safe_post_data(raw, 4)
    assert value == {
        "binary_or_unstructured": True,
        "byte_length": len(raw),
        "sha256": hashlib.sha256(raw).hexdigest(),
        "truncated": True,
    }


def test_unstructured_and_truncated_post_data_never_leaks_secrets() -> None:
    form = _safe_post_data(
        b"query=browser-event&password=super-secret",
        100,
        "application/x-www-form-urlencoded",
    )
    assert form == {"query": ["browser-event"], "password": "[REDACTED]"}

    for raw, content_type, limit in (
        (b"password=super-secret", "text/plain", 100),
        (b'{"password":"super-secret"}', "application/json", 8),
        (b"query=public&note=super-secret", "application/x-www-form-urlencoded", 12),
        (b"--boundary\r\npassword=super-secret", "multipart/form-data; boundary=boundary", 100),
    ):
        value = _safe_post_data(raw, limit, content_type)
        assert value["binary_or_unstructured"] is True
        assert "super-secret" not in str(value)

    response = _safe_post_data(
        b'{"token":"super-secret","payload":"too long"}',
        20,
        "application/json",
    )
    assert response["binary_or_unstructured"] is True
    assert "super-secret" not in str(response)


@pytest.mark.asyncio
async def test_response_capture_errors_do_not_escape_event_listener(tmp_path) -> None:
    actor = BrowserActor("http://127.0.0.1:9", tmp_path, EvidenceStore())

    class BrokenResponse:
        @property
        def headers(self):
            raise RuntimeError("target closed")

    await actor._call(actor._on_response, BrokenResponse())
    assert actor._network_records == [{"capture_error": "RuntimeError"}]
    tmp_path.mkdir(exist_ok=True)
    await actor.close()


@pytest.mark.asyncio
async def test_oversized_response_is_not_read(tmp_path) -> None:
    actor = BrowserActor("http://127.0.0.1:9", tmp_path, EvidenceStore(), response_body_limit=16)

    class OversizedResponse:
        url = "https://example.test/api"
        status = 200
        headers = {"content-type": "application/json", "content-length": "17"}
        request = SimpleNamespace(
            resource_type="fetch",
            method="GET",
            headers={},
            post_data_buffer=None,
        )

        def body(self):
            raise AssertionError("oversized body must not be read")

    await actor._call(actor._on_response, OversizedResponse())
    assert actor._network_records[0]["body_truncated"] is True
    assert "超过采集上限" in actor._network_records[0]["body_skipped"]
    tmp_path.mkdir(exist_ok=True)
    await actor.close()


@pytest.mark.asyncio
async def test_oversized_request_body_is_not_read_and_url_is_sanitized(tmp_path) -> None:
    actor = BrowserActor("http://127.0.0.1:9", tmp_path, EvidenceStore(), response_body_limit=16)

    class OversizedRequest:
        resource_type = "fetch"
        method = "POST"
        headers = {"content-type": "application/json", "content-length": "17"}

        @property
        def post_data_buffer(self):
            raise AssertionError("oversized request body must not be read")

    response = SimpleNamespace(
        url="https://example.test/api?query=public&security=private",
        status=200,
        headers={"content-type": "application/json", "content-length": "0"},
        request=OversizedRequest(),
        body=lambda: b"",
    )

    await actor._call(actor._on_response, response)
    record = actor._network_records[0]
    assert record["post_data_truncated"] is True
    assert "超过采集上限" in record["post_data_skipped"]
    assert "query=public" in record["url"]
    assert "private" not in record["url"]
    tmp_path.mkdir(exist_ok=True)
    await actor.close()


@pytest.mark.asyncio
async def test_request_size_metadata_allows_bounded_structured_body(tmp_path) -> None:
    actor = BrowserActor("http://127.0.0.1:9", tmp_path, EvidenceStore(), response_body_limit=64)
    raw = b'{"query":"browser-event"}'
    request = SimpleNamespace(
        resource_type="fetch",
        method="POST",
        headers={"content-type": "application/json"},
        sizes=lambda: {"requestBodySize": len(raw)},
        post_data_buffer=raw,
    )
    response = SimpleNamespace(
        url="https://example.test/api",
        status=204,
        headers={},
        request=request,
    )

    await actor._call(actor._on_response, response)
    assert actor._network_records[0]["post_data"] == {"query": "browser-event"}
    tmp_path.mkdir(exist_ok=True)
    await actor.close()


def test_form_confirmation_requires_submit_intent_and_real_change() -> None:
    submit = {"target": {"tag": "button", "type": "submit", "text": "Submit"}}
    ordinary = {"target": {"tag": "button", "type": "button", "text": "Next"}}
    assert BrowserActor._is_confirmed_submission("click", True, True, submit, "Submitted successfully")
    assert not BrowserActor._is_confirmed_submission("fill", True, True, submit, "Submitted successfully")
    assert not BrowserActor._is_confirmed_submission("click", True, True, ordinary, "Success stories")
    assert not BrowserActor._is_confirmed_submission("click", True, False, submit, "Submitted successfully")


@pytest.mark.asyncio
async def test_eight_browser_actors_have_isolated_single_threads(tmp_path) -> None:
    actors = [
        BrowserActor(f"http://127.0.0.1:{9000 + index}", tmp_path / str(index), EvidenceStore())
        for index in range(8)
    ]

    def owner(actor: BrowserActor) -> int:
        actor._assert_thread()
        return threading.get_ident()

    thread_ids = await asyncio.gather(*(actor._call(owner, actor) for actor in actors))
    assert len(set(thread_ids)) == 8
    for index, actor in enumerate(actors):
        assert actor.output_dir == tmp_path / str(index)
        actor.output_dir.mkdir(parents=True)
    await asyncio.gather(*(actor.close() for actor in actors))


@pytest.mark.asyncio
async def test_cross_thread_access_fails_fast(tmp_path) -> None:
    actor = BrowserActor("http://127.0.0.1:9", tmp_path, EvidenceStore())
    await actor._call(actor._assert_thread)
    with pytest.raises(RuntimeError, match="线程边界"):
        actor._assert_thread()
    tmp_path.mkdir(exist_ok=True)
    await actor.close()


def test_semantic_fingerprint_ignores_volatile_time_but_tracks_real_state() -> None:
    base = [
        {
            "bid": "volatile-bid",
            "tag": "p",
            "text": "Updated 2026-07-31T17:10:11.123Z",
            "rect": [0, 0, 10, 10],
        },
        {"bid": "answer", "tag": "span", "text": "Result 7"},
    ]
    changed_clock = [
        {**base[0], "bid": "new-bid", "text": "Updated 2026-07-31T17:59:59.999Z"},
        base[1],
    ]
    real_change = [changed_clock[0], {**base[1], "text": "Result 8"}]
    first = semantic_page_fingerprint("https://EXAMPLE.test:443/x?b=2&a=1#clock", "Report", base)
    assert first == semantic_page_fingerprint(
        "https://example.test/x?a=1&b=2#clock", "Report", changed_clock
    )
    assert first != semantic_page_fingerprint(
        "https://example.test/x?a=1&b=2#clock", "Report", real_change
    )
    assert canonical_url("https://EXAMPLE.test:443/x?b=2&a=1#fragment") == (
        "https://example.test/x?a=1&b=2#fragment"
    )
    assert canonical_url("https://example.test/#/route-a") != canonical_url(
        "https://example.test/#/route-b"
    )


def test_semantic_fingerprint_tracks_later_iframe_after_large_main_frame() -> None:
    elements = [
        {"frame_url": "https://example.test", "tag": "p", "text": f"main-{index}"}
        for index in range(3_000)
    ]
    elements.extend(
        {"frame_url": "https://example.test/frame-1", "tag": "p", "text": f"first-{index}"}
        for index in range(1_000)
    )
    elements.extend(
        {"frame_url": "https://example.test/frame-2", "tag": "p", "text": f"second-{index}"}
        for index in range(1_000)
    )
    changed = [*elements[:-1], {**elements[-1], "text": "second-changed"}]
    assert semantic_page_fingerprint("https://example.test", "Frames", elements) != (
        semantic_page_fingerprint("https://example.test", "Frames", changed)
    )


def test_screenshot_failure_does_not_publish_broken_artifact(tmp_path) -> None:
    actor = BrowserActor("http://127.0.0.1:9", tmp_path, EvidenceStore())

    class BrokenPage:
        url = "https://example.test"

        def is_closed(self) -> bool:
            return False

        def screenshot(self, **kwargs) -> None:  # noqa: ANN003
            raise RuntimeError("screenshot failed")

    actor._connected = True
    actor._page = BrokenPage()
    actor._last_semantic_state = {"semantic_page_fingerprint": "same"}
    for dirname in ("trajectory", "trajectory_visual"):
        (tmp_path / dirname).mkdir(parents=True)
    with pytest.raises(RuntimeError, match="screenshot failed"):
        actor._capture_step("test", force=True)
    assert actor._last_capture_path is None
    assert not list((tmp_path / "trajectory").iterdir())
    actor._closed = True
    actor._executor.shutdown(wait=True, cancel_futures=True)


def test_raw_dom_hash_records_frame_failure_without_raising(tmp_path) -> None:
    actor = BrowserActor("http://127.0.0.1:9", tmp_path, EvidenceStore())

    class BrokenFrame:
        url = "https://example.test/frame"

        def evaluate(self, script: str) -> str:
            raise RuntimeError("frame detached")

    actor._page = SimpleNamespace(frames=[BrokenFrame()])
    value = actor._dom_hash()
    assert len(value) == 16
    assert actor._last_dom_hash == value
    actor._closed = True
    actor._executor.shutdown(wait=True, cancel_futures=True)


def test_action_after_state_failure_returns_receipt_instead_of_crashing(tmp_path) -> None:
    actor = BrowserActor("http://127.0.0.1:9", tmp_path, EvidenceStore())

    class Page:
        url = "https://example.test"

        def is_closed(self) -> bool:
            return False

    page = Page()
    actor._owner_thread = threading.get_ident()
    actor._connected = True
    actor._page = page
    actor._context = SimpleNamespace(pages=[page])
    actor._dom_hash = lambda: "raw"  # type: ignore[method-assign]
    calls = 0

    def semantic_state():
        nonlocal calls
        calls += 1
        if calls > 1:
            raise RuntimeError("frame detached")
        return {
            "url": page.url,
            "title": "before",
            "semantic_page_fingerprint": "before-state",
            "semantic_element_count": 1,
        }

    actor._semantic_page_state = semantic_state  # type: ignore[method-assign]
    receipt = actor._action_sync("click", lambda: {}, None)
    assert receipt["success"] is False
    assert receipt["postconditions"]["after_semantic_page_fingerprint"] == "before-state"
    actor._closed = True
    actor._executor.shutdown(wait=True, cancel_futures=True)


def test_unchanged_actions_reuse_screenshot_and_skip_raw_dom_hash(tmp_path) -> None:
    actor = BrowserActor("http://127.0.0.1:9", tmp_path, EvidenceStore())

    class Page:
        url = "https://example.test"

        def __init__(self) -> None:
            self.screenshot_calls = 0

        def is_closed(self) -> bool:
            return False

        def screenshot(self, *, path: str, **kwargs) -> None:  # noqa: ANN003
            from pathlib import Path

            self.screenshot_calls += 1
            Path(path).write_bytes(b"png")

    page = Page()
    state = {
        "url": page.url,
        "title": "unchanged",
        "semantic_page_fingerprint": "same",
        "semantic_element_count": 1,
    }
    actor._owner_thread = threading.get_ident()
    actor._connected = True
    actor._page = page
    actor._context = SimpleNamespace(pages=[page])
    actor._last_semantic_state = state
    actor._semantic_page_state = lambda: state  # type: ignore[method-assign]
    actor._settle = lambda: state  # type: ignore[method-assign]
    dom_hash_calls = 0

    def dom_hash() -> str:
        nonlocal dom_hash_calls
        dom_hash_calls += 1
        return "raw"

    actor._dom_hash = dom_hash  # type: ignore[method-assign]
    for dirname in ("trajectory", "trajectory_visual"):
        (tmp_path / dirname).mkdir(parents=True)
    first = actor._action_sync("wait", lambda: {"milliseconds": 0}, None)
    second = actor._action_sync("wait", lambda: {"milliseconds": 0}, None)
    assert first["before_dom_hash"] == first["after_dom_hash"] == ""
    assert second["before_dom_hash"] == second["after_dom_hash"] == ""
    assert dom_hash_calls == 0
    assert page.screenshot_calls == 1
    actor._closed = True
    actor._executor.shutdown(wait=True, cancel_futures=True)


def test_screenshot_hardlink_falls_back_to_copy(tmp_path, monkeypatch) -> None:
    actor = BrowserActor("http://127.0.0.1:9", tmp_path, EvidenceStore())

    class Page:
        url = "https://example.test"

        def is_closed(self) -> bool:
            return False

        def screenshot(self, *, path: str, **kwargs) -> None:  # noqa: ANN003
            from pathlib import Path

            Path(path).write_bytes(b"png")

    actor._connected = True
    actor._page = Page()
    actor._last_semantic_state = {"semantic_page_fingerprint": "same"}
    for dirname in ("trajectory", "trajectory_visual"):
        (tmp_path / dirname).mkdir(parents=True)
    monkeypatch.setattr("web_agent.browser_actor.os.link", lambda source, target: (_ for _ in ()).throw(OSError()))
    path = actor._capture_step("test", force=True)
    assert (tmp_path / "trajectory_visual" / "000.png").read_bytes() == b"png"
    assert path.endswith("trajectory/000.png")
    actor._closed = True
    actor._executor.shutdown(wait=True, cancel_futures=True)
