from __future__ import annotations

import asyncio
import hashlib
import threading
import time
from types import SimpleNamespace

import pytest

from web_agent.browser_actor import (
    ActorCallDeadlineExceeded,
    BrowserActor,
    BrowserActorPoisonedError,
    _CallDispatchState,
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
    websocket_error = sanitize_error_text(
        "connect ws://user:password@127.0.0.1:9222/devtools?access_token=private"
    )
    assert "user" not in websocket_error
    assert "password" not in websocket_error
    assert "private" not in websocket_error
    path_secret = sanitize_url("wss://cdp.test/access_token=supersecret123/api_key=anothersecret123")
    assert "supersecret123" not in path_secret
    assert "anothersecret123" not in path_secret


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
    assert actor._network_records == [
        {
            "capture_error": "RuntimeError",
            "task_generation": 0,
            "attempt": 1,
        }
    ]
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


def test_observation_change_tracking_uses_stable_bid_and_semantic_signature(tmp_path) -> None:
    actor = BrowserActor("http://127.0.0.1:9", tmp_path, EvidenceStore())
    baseline = [
        {"bid": "10", "frame": 0, "frame_url": "https://example.test/app#/one", "document_id": "doc-1", "tag": "p", "text": "Waiting"},
        {"bid": "11", "frame": 1, "frame_url": "https://example.test/frame", "document_id": "frame-doc", "tag": "button", "text": "Open"},
    ]
    actor._annotate_observation_changes(baseline, document_scope="doc-1")
    assert not any(item.get("new") or item.get("changed") for item in baseline)

    updated = [
        {"bid": "10", "frame": 0, "frame_url": "https://example.test/app#/two", "document_id": "doc-1", "tag": "p", "text": "Ready"},
        {"bid": "11", "frame": 2, "frame_url": "https://example.test/frame", "document_id": "frame-doc", "tag": "button", "text": "Open"},
        {"bid": "12", "frame": 0, "frame_url": "https://example.test/app#/two", "document_id": "doc-1", "role": "option", "text": "Paris"},
    ]
    actor._annotate_observation_changes(updated, document_scope="doc-1")
    assert updated[0]["changed"] is True
    assert "changed" not in updated[1] and "new" not in updated[1]
    assert updated[2]["new"] is True

    partial = [updated[0]]
    actor._annotate_observation_changes(
        partial,
        document_scope="doc-1",
        incomplete_document_scopes={"frame-doc"},
    )
    recovered = [
        {"bid": "10", "frame": 0, "frame_url": "https://example.test/app#/two", "document_id": "doc-1", "tag": "p", "text": "Ready"},
        {"bid": "11", "frame": 5, "frame_url": "https://example.test/frame", "document_id": "frame-doc", "tag": "button", "text": "Open"},
    ]
    actor._annotate_observation_changes(recovered, document_scope="doc-1")
    assert "new" not in recovered[1]
    actor._closed = True
    actor._executor.shutdown(wait=True, cancel_futures=True)


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


@pytest.mark.asyncio
async def test_deadline_before_dispatch_is_safe_and_actor_remains_usable(tmp_path) -> None:
    actor = BrowserActor("http://127.0.0.1:9", tmp_path, EvidenceStore())
    started = threading.Event()
    release = threading.Event()

    def occupy_owner_thread() -> str:
        started.set()
        release.wait(timeout=2)
        return "released"

    first = asyncio.create_task(actor._call(occupy_owner_thread))
    assert await asyncio.to_thread(started.wait, 1)
    with pytest.raises(ActorCallDeadlineExceeded) as caught:
        await actor._call(lambda: "must-not-run", deadline=time.monotonic() + 0.02, operation="queued")
    assert caught.value.dispatched is False
    assert actor.poisoned is False
    release.set()
    assert await first == "released"
    assert await actor._call(lambda: "still-usable") == "still-usable"
    tmp_path.mkdir(exist_ok=True)
    await actor.close()


@pytest.mark.asyncio
async def test_deadline_during_precondition_never_dispatches_mutation(tmp_path) -> None:
    actor = BrowserActor("http://127.0.0.1:9", tmp_path, EvidenceStore())
    precondition_started = threading.Event()
    release_precondition = threading.Event()
    finished = threading.Event()
    click_count = 0

    class Locator:
        def click(self, timeout: int, trial: bool = False) -> None:  # noqa: ARG002
            nonlocal click_count
            click_count += 1

    page = SimpleNamespace(url="https://example.test", is_closed=lambda: False)
    actor._page = page
    actor._context = SimpleNamespace(pages=[page])
    actor._connected = True
    actor._ensure_live = lambda: None  # type: ignore[method-assign]
    actor._locator = lambda bid: Locator()  # type: ignore[method-assign]
    actor._semantic_page_state = lambda: {  # type: ignore[method-assign]
        "url": page.url,
        "title": "page",
        "semantic_page_fingerprint": "same",
        "semantic_element_count": 1,
    }

    def blocked_element_state(locator):  # noqa: ANN001, ARG001
        precondition_started.set()
        release_precondition.wait(timeout=2)
        return {"type": "button", "text": "go"}

    actor._element_state = blocked_element_state  # type: ignore[method-assign]

    def invoke() -> dict:
        try:
            return actor._action_sync("click", lambda: actor._click_op("abc"), "abc")
        finally:
            finished.set()

    call = asyncio.create_task(
        actor._call(
            invoke,
            deadline=time.monotonic() + 0.2,
            operation="click",
            mutation_aware=True,
        )
    )
    assert await asyncio.to_thread(precondition_started.wait, 1)
    with pytest.raises(ActorCallDeadlineExceeded) as caught:
        await call
    assert caught.value.dispatched is False
    assert actor.poisoned is False
    release_precondition.set()
    assert await asyncio.to_thread(finished.wait, 1)
    assert click_count == 0
    assert await actor._call(lambda: "still-usable") == "still-usable"
    tmp_path.mkdir(exist_ok=True)
    await actor.close()


@pytest.mark.asyncio
async def test_deadline_during_actionability_trial_is_safe_and_late_result_is_consumed(tmp_path) -> None:
    actor = BrowserActor("http://127.0.0.1:9", tmp_path, EvidenceStore())
    trial_started = threading.Event()
    release_trial = threading.Event()
    finished = threading.Event()
    actual_clicks = 0

    class Locator:
        def evaluate(self, expression: str) -> dict[str, object]:  # noqa: ARG002
            return {"bid": "abc", "tag": "button", "type": "button", "text": "Open"}

        def click(self, timeout: int, trial: bool = False) -> None:  # noqa: ARG002
            nonlocal actual_clicks
            if trial:
                trial_started.set()
                release_trial.wait(timeout=2)
            else:
                actual_clicks += 1

    page = SimpleNamespace(url="https://example.test", is_closed=lambda: False)
    state = {
        "url": page.url,
        "title": "page",
        "semantic_page_fingerprint": "same",
        "semantic_element_count": 1,
    }
    actor._page = page
    actor._context = SimpleNamespace(pages=[page])
    actor._connected = True
    actor._locator = lambda bid: Locator()  # type: ignore[method-assign]
    actor._semantic_page_state = lambda: state  # type: ignore[method-assign]

    def invoke() -> dict[str, object]:
        try:
            return actor._action_sync("click", lambda: actor._click_op("abc"), "abc")
        finally:
            finished.set()

    call = asyncio.create_task(
        actor._call(
            invoke,
            deadline=time.monotonic() + 0.2,
            operation="click",
            mutation_aware=True,
        )
    )
    assert await asyncio.to_thread(trial_started.wait, 1)
    with pytest.raises(ActorCallDeadlineExceeded) as caught:
        await call
    assert caught.value.dispatched is False
    assert actor.poisoned is False

    release_trial.set()
    assert await asyncio.to_thread(finished.wait, 1)
    await asyncio.sleep(0)
    assert actual_clicks == 0
    assert await actor._call(lambda: "still-usable") == "still-usable"
    tmp_path.mkdir(exist_ok=True)
    await actor.close()


@pytest.mark.asyncio
async def test_actionability_failure_before_click_dispatch_keeps_actor_usable(tmp_path) -> None:
    from playwright.sync_api import TimeoutError as PlaywrightTimeoutError

    actor = BrowserActor("http://127.0.0.1:9", tmp_path, EvidenceStore())
    fail_actionability = True
    calls: list[bool] = []

    class Locator:
        def evaluate(self, expression: str) -> dict[str, object]:  # noqa: ARG002
            return {"bid": "abc", "tag": "button", "type": "button", "text": "Open"}

        def click(self, timeout: int, trial: bool = False) -> None:  # noqa: ARG002
            calls.append(trial)
            if trial and fail_actionability:
                raise PlaywrightTimeoutError("element is not visible")

    page = SimpleNamespace(url="https://example.test", is_closed=lambda: False)
    state = {
        "url": page.url,
        "title": "page",
        "semantic_page_fingerprint": "same",
        "semantic_element_count": 1,
    }
    actor._page = page
    actor._context = SimpleNamespace(pages=[page])
    actor._connected = True
    actor._locator = lambda bid: Locator()  # type: ignore[method-assign]
    actor._semantic_page_state = lambda: state  # type: ignore[method-assign]
    actor._settle = lambda: state  # type: ignore[method-assign]

    failed = await actor.click("abc")
    assert failed["success"] is False
    assert failed["safe_to_retry"] is True
    assert failed["terminal_uncertain"] is False
    assert actor.poisoned is False
    assert calls == [True]

    fail_actionability = False
    succeeded = await actor.click("abc")
    assert succeeded["success"] is True
    assert calls == [True, True, False]
    assert actor.poisoned is False
    tmp_path.mkdir(exist_ok=True)
    await actor.close()


@pytest.mark.asyncio
async def test_actionability_failure_between_trial_and_real_click_keeps_actor_usable(tmp_path) -> None:
    from playwright.sync_api import TimeoutError as PlaywrightTimeoutError

    actor = BrowserActor("http://127.0.0.1:9", tmp_path, EvidenceStore())
    fail_real_actionability = True
    calls: list[bool] = []

    class Locator:
        def evaluate(self, expression: str) -> dict[str, object]:  # noqa: ARG002
            return {"bid": "abc", "tag": "button", "type": "button", "text": "Open"}

        def click(self, timeout: int, trial: bool = False) -> None:  # noqa: ARG002
            calls.append(trial)
            if not trial and fail_real_actionability:
                raise PlaywrightTimeoutError(
                    "Locator.click: Timeout exceeded.\n"
                    "- attempting click action\n"
                    "- waiting for element to be visible, enabled and stable\n"
                    "- element is not visible"
                )

    page = SimpleNamespace(url="https://example.test", is_closed=lambda: False)
    state = {
        "url": page.url,
        "title": "page",
        "semantic_page_fingerprint": "same",
        "semantic_element_count": 1,
    }
    actor._page = page
    actor._context = SimpleNamespace(pages=[page])
    actor._connected = True
    actor._locator = lambda bid: Locator()  # type: ignore[method-assign]
    actor._semantic_page_state = lambda: state  # type: ignore[method-assign]
    actor._settle = lambda: state  # type: ignore[method-assign]

    failed = await actor.click("abc")
    assert failed["success"] is False
    assert failed["safe_to_retry"] is True
    assert failed["terminal_uncertain"] is False
    assert actor.poisoned is False
    assert calls == [True, False]

    fail_real_actionability = False
    assert (await actor.click("abc"))["success"] is True
    assert calls == [True, False, True, False]
    assert actor.poisoned is False
    tmp_path.mkdir(exist_ok=True)
    await actor.close()


@pytest.mark.asyncio
async def test_set_checked_noop_does_not_dispatch_mutation(tmp_path) -> None:
    actor = BrowserActor("http://127.0.0.1:9", tmp_path, EvidenceStore())
    set_checked_calls = 0

    class Locator:
        def is_checked(self) -> bool:
            return True

        def set_checked(self, checked: bool, timeout: int, trial: bool = False) -> None:  # noqa: ARG002
            nonlocal set_checked_calls
            set_checked_calls += 1

    page = SimpleNamespace(url="https://example.test", is_closed=lambda: False)
    state = {
        "url": page.url,
        "title": "page",
        "semantic_page_fingerprint": "same",
        "semantic_element_count": 1,
    }
    actor._page = page
    actor._context = SimpleNamespace(pages=[page])
    actor._connected = True
    actor._locator = lambda bid: Locator()  # type: ignore[method-assign]
    actor._semantic_page_state = lambda: state  # type: ignore[method-assign]
    actor._settle = lambda: state  # type: ignore[method-assign]

    result = await actor.set_checked("abc", True)
    assert result["success"] is True
    assert result["postconditions"]["value_changed"] is False
    assert set_checked_calls == 0
    assert actor.poisoned is False
    tmp_path.mkdir(exist_ok=True)
    await actor.close()


@pytest.mark.asyncio
async def test_begin_task_waits_for_old_action_before_switching_generation(tmp_path) -> None:
    actor = BrowserActor("http://127.0.0.1:9", tmp_path, EvidenceStore())
    action_started = threading.Event()
    release_action = threading.Event()
    mutation_generations: list[int] = []
    old_generation = actor.task_generation

    def old_action() -> str:
        action_started.set()
        release_action.wait(timeout=2)
        return actor._dispatch_mutation(
            lambda: mutation_generations.append(actor.task_generation) or "done"
        )

    actor._begin_task_sync = lambda initial_url, output_dir, store: {  # type: ignore[method-assign]
        "url": initial_url,
        "connected": True,
    }
    action = asyncio.create_task(
        actor._call(old_action, operation="click", mutation_aware=True)
    )
    assert await asyncio.to_thread(action_started.wait, 1)
    transition = asyncio.create_task(
        actor.begin_task("https://new.test", tmp_path / "new", EvidenceStore())
    )
    await asyncio.sleep(0)

    assert actor.task_generation == old_generation
    with pytest.raises(BrowserActorPoisonedError, match="任务切换期间"):
        await actor._call(lambda: "old task must not queue")

    release_action.set()
    assert await action == "done"
    result = await transition
    assert mutation_generations == [old_generation]
    assert result["task_generation"] == old_generation + 1
    assert actor.task_generation == old_generation + 1
    await actor.close()


def test_dispatch_cancellation_and_worker_start_are_atomic() -> None:
    state = _CallDispatchState()
    assert state.cancel_if_queued() is True
    assert state.begin(time.monotonic() + 1) is False


def test_poison_and_evidence_publication_share_one_commit_boundary(tmp_path) -> None:
    store = EvidenceStore()
    actor = BrowserActor("http://127.0.0.1:9", tmp_path, store)
    actor._mark_poisoned(actor.task_generation, "ACTOR_POISONED: test")

    with pytest.raises(BrowserActorPoisonedError, match="ACTOR_POISONED"):
        store.add("dom", "https://example.test", "late", {"value": 1})

    assert store.values() == []
    actor._closed = True
    actor._executor.shutdown(wait=True, cancel_futures=True)


@pytest.mark.asyncio
async def test_timed_out_attempt_rolls_back_evidence_written_before_tail_block(tmp_path) -> None:
    store = EvidenceStore()
    actor = BrowserActor("http://127.0.0.1:9", tmp_path, store)
    evidence_added = threading.Event()
    release = threading.Event()

    def write_then_block() -> None:
        store.add("dom", "https://example.test", "temporary", {"data": "late"})
        evidence_added.set()
        release.wait(timeout=2)

    call = asyncio.create_task(
        actor._call(
            write_then_block,
            deadline=time.monotonic() + 0.1,
            operation="observe",
        )
    )
    assert await asyncio.to_thread(evidence_added.wait, 1)
    with pytest.raises(ActorCallDeadlineExceeded):
        await call
    assert store.values() == []
    release.set()
    tmp_path.mkdir(exist_ok=True)
    await actor.close()


def test_old_evidence_store_is_rejected_after_generation_switch(tmp_path) -> None:
    old_store = EvidenceStore()
    actor = BrowserActor("http://127.0.0.1:9", tmp_path, old_store)
    first_generation = actor._reserve_generation()
    actor._bind_evidence_store(old_store, first_generation)
    new_store = EvidenceStore()
    second_generation = actor._reserve_generation()
    actor._bind_evidence_store(new_store, second_generation)

    with pytest.raises(BrowserActorPoisonedError, match="STALE_TASK_GENERATION"):
        old_store.add("dom", "https://example.test", "old", {"data": 1})
    current = new_store.add("dom", "https://example.test", "new", {"data": 2})

    assert old_store.values() == []
    assert current.payload["task_generation"] == second_generation
    actor._closed = True
    actor._executor.shutdown(wait=True, cancel_futures=True)


@pytest.mark.parametrize("action", ["accept", "dismiss"])
def test_dialog_resolution_failure_poisons_actor(tmp_path, action: str) -> None:
    actor = BrowserActor("http://127.0.0.1:9", tmp_path, EvidenceStore())
    actor._page = SimpleNamespace(url="https://example.test")
    actor._next_dialog_action = (action, None, actor.task_generation)

    class Dialog:
        type = "confirm"
        message = "continue?"

        def accept(self, prompt_text=None) -> None:  # noqa: ANN001, ARG002
            raise RuntimeError("accept failed")

        def dismiss(self) -> None:
            raise RuntimeError("dismiss failed")

    with pytest.raises(BrowserActorPoisonedError, match=f"dialog {action}"):
        actor._on_dialog(Dialog(), actor.task_generation)

    assert actor.poisoned is True
    assert actor._dialogs[0]["error"] == "RuntimeError"
    actor._closed = True
    actor._executor.shutdown(wait=True, cancel_futures=True)


def test_begin_task_clears_unconsumed_dialog_arm(tmp_path) -> None:
    actor = BrowserActor("http://127.0.0.1:9", tmp_path, EvidenceStore())
    old_generation = actor.task_generation
    actor._next_dialog_action = ("accept", "stale secret", old_generation)
    generation = actor._reserve_generation()
    actor._owner_thread = threading.get_ident()
    actor._playwright = object()
    actor._context = None
    actor._prepare_task = lambda initial_url: {"url": initial_url}  # type: ignore[method-assign]

    actor._begin_task_sync("https://example.test", tmp_path, EvidenceStore())

    assert actor._next_dialog_action == ("dismiss", None, generation)
    actor._closed = True
    actor._executor.shutdown(wait=True, cancel_futures=True)


def test_late_pdf_response_cannot_satisfy_next_download_attempt(tmp_path) -> None:
    from playwright.sync_api import TimeoutError as PlaywrightTimeoutError

    actor = BrowserActor("http://127.0.0.1:9", tmp_path, EvidenceStore())
    (tmp_path / "downloads").mkdir()
    actor._owner_thread = threading.get_ident()
    actor._thread_context.generation = actor.task_generation
    actor._thread_context.attempt = 2
    base_ms = time.time() * 1_000 - 1_000
    actor._attempt_windows[(actor.task_generation, 1)] = {
        "started_ms": base_ms,
        "ended_ms": base_ms + 500,
        "mutation_dispatched_ms": base_ms + 100,
    }
    actor._attempt_windows[(actor.task_generation, 2)] = {
        "started_ms": base_ms + 800,
        "ended_ms": None,
        "mutation_dispatched_ms": None,
    }

    request = SimpleNamespace(
        resource_type="document",
        timing={"startTime": base_ms + 200},
        url="https://old.test/stale.pdf",
        redirected_from=None,
        is_navigation_request=lambda: True,
    )
    response = SimpleNamespace(
        headers={"content-type": "application/pdf"},
        request=request,
        url="https://old.test/stale.pdf",
    )

    class ExpectDownload:
        def __enter__(self):
            return SimpleNamespace()

        def __exit__(self, exc_type, exc, traceback):  # noqa: ANN001, ARG002
            raise PlaywrightTimeoutError("no download")

    class Locator:
        def get_attribute(self, name: str) -> str | None:
            return "/wanted.pdf" if name == "href" else None

        def click(self, timeout: int, trial: bool = False) -> None:  # noqa: ARG002
            if not trial:
                actor._capture_response(response, actor.task_generation)

    actor._page = SimpleNamespace(
        url="https://current.test/page",
        expect_download=lambda timeout: ExpectDownload(),
    )
    actor._locator = lambda bid: Locator()  # type: ignore[method-assign]

    with pytest.raises(RuntimeError, match="既未触发下载"):
        actor._download_op("pdf")

    assert actor._pdf_responses[0]["attempt"] == 1
    assert list((tmp_path / "downloads").iterdir()) == []
    actor._closed = True
    actor._executor.shutdown(wait=True, cancel_futures=True)


def test_current_non_navigation_pdf_matching_link_can_satisfy_download_fallback(tmp_path) -> None:
    from playwright.sync_api import TimeoutError as PlaywrightTimeoutError

    actor = BrowserActor("http://127.0.0.1:9", tmp_path, EvidenceStore())
    (tmp_path / "downloads").mkdir()
    actor._owner_thread = threading.get_ident()
    actor._thread_context.generation = actor.task_generation
    actor._thread_context.attempt = 1
    actor._attempt_windows[(actor.task_generation, 1)] = {
        "started_ms": time.time() * 1_000 - 100,
        "ended_ms": None,
        "mutation_dispatched_ms": None,
    }

    class Request:
        resource_type = "document"
        url = "chrome-extension://pdf-viewer/stream-id"
        redirected_from = None
        frame = SimpleNamespace(url="https://example.test/current.pdf")

        @property
        def timing(self) -> dict[str, float]:
            return {"startTime": time.time() * 1_000}

        def is_navigation_request(self) -> bool:
            return False

    response = SimpleNamespace(
        headers={"content-type": "application/pdf"},
        request=Request(),
        url="chrome-extension://pdf-viewer/stream-id",
        body=lambda: b"%PDF-current",
    )

    class ExpectDownload:
        def __enter__(self):
            return SimpleNamespace()

        def __exit__(self, exc_type, exc, traceback):  # noqa: ANN001, ARG002
            raise PlaywrightTimeoutError("inline PDF")

    class Locator:
        def get_attribute(self, name: str) -> str | None:
            return "/current.pdf" if name == "href" else None

        def click(self, timeout: int, trial: bool = False) -> None:  # noqa: ARG002
            if not trial:
                actor._capture_response(response, actor.task_generation)

    actor._page = SimpleNamespace(
        url="https://example.test/page",
        expect_download=lambda timeout: ExpectDownload(),
    )
    actor._locator = lambda bid: Locator()  # type: ignore[method-assign]

    result = actor._download_op("pdf")

    assert result["download"]["inline_pdf"] is True
    assert (tmp_path / "downloads" / "current.pdf").read_bytes() == b"%PDF-current"
    actor._closed = True
    actor._executor.shutdown(wait=True, cancel_futures=True)


def test_unrelated_non_navigation_pdf_cannot_satisfy_download_fallback(tmp_path) -> None:
    from playwright.sync_api import TimeoutError as PlaywrightTimeoutError

    actor = BrowserActor("http://127.0.0.1:9", tmp_path, EvidenceStore())
    (tmp_path / "downloads").mkdir()
    actor._owner_thread = threading.get_ident()
    actor._thread_context.generation = actor.task_generation
    actor._thread_context.attempt = 1
    actor._attempt_windows[(actor.task_generation, 1)] = {
        "started_ms": time.time() * 1_000 - 100,
        "ended_ms": None,
        "mutation_dispatched_ms": None,
    }
    request = SimpleNamespace(
        resource_type="document",
        timing={"startTime": time.time() * 1_000},
        url="https://example.test/unrelated.pdf",
        redirected_from=None,
        is_navigation_request=lambda: False,
    )
    response = SimpleNamespace(
        headers={"content-type": "application/pdf"},
        request=request,
        url=request.url,
        body=lambda: b"%PDF-unrelated",
    )

    class ExpectDownload:
        def __enter__(self):
            return SimpleNamespace()

        def __exit__(self, exc_type, exc, traceback):  # noqa: ANN001, ARG002
            raise PlaywrightTimeoutError("no download")

    class Locator:
        def get_attribute(self, name: str) -> str | None:
            return "/expected.pdf" if name == "href" else None

        def click(self, timeout: int, trial: bool = False) -> None:  # noqa: ARG002
            if not trial:
                actor._capture_response(response, actor.task_generation)

    actor._page = SimpleNamespace(
        url="https://example.test/page",
        expect_download=lambda timeout: ExpectDownload(),
    )
    actor._locator = lambda bid: Locator()  # type: ignore[method-assign]

    with pytest.raises(RuntimeError, match="既未触发下载"):
        actor._download_op("pdf")

    assert list((tmp_path / "downloads").iterdir()) == []
    actor._closed = True
    actor._executor.shutdown(wait=True, cancel_futures=True)


def test_download_trial_failure_is_safe_but_post_click_timeout_poisons(tmp_path) -> None:
    from playwright.sync_api import TimeoutError as PlaywrightTimeoutError

    actor = BrowserActor("http://127.0.0.1:9", tmp_path, EvidenceStore())
    actor._owner_thread = threading.get_ident()
    actor._connected = True
    fail_actionability = True
    download_waits = 0

    class ExpectDownload:
        def __enter__(self):
            nonlocal download_waits
            download_waits += 1
            return SimpleNamespace()

        def __exit__(self, exc_type, exc, traceback):  # noqa: ANN001, ARG002
            raise PlaywrightTimeoutError("no download")

    class Locator:
        def get_attribute(self, name: str) -> str | None:
            return "/report" if name == "href" else None

        def click(self, timeout: int, trial: bool = False) -> None:  # noqa: ARG002
            if trial and fail_actionability:
                raise PlaywrightTimeoutError("element is not visible")

    page = SimpleNamespace(
        url="https://example.test/page",
        is_closed=lambda: False,
        expect_download=lambda timeout: ExpectDownload(),
    )
    state = {
        "url": page.url,
        "title": "page",
        "semantic_page_fingerprint": "same",
        "semantic_element_count": 1,
    }
    actor._page = page
    actor._context = SimpleNamespace(pages=[page])
    actor._locator = lambda bid: Locator()  # type: ignore[method-assign]
    actor._semantic_page_state = lambda: state  # type: ignore[method-assign]

    safe_failure = actor._action_sync(
        "download",
        lambda: actor._download_op("report"),
        "report",
    )
    assert safe_failure["success"] is False
    assert safe_failure["safe_to_retry"] is True
    assert actor.poisoned is False
    assert download_waits == 0

    fail_actionability = False
    with pytest.raises(BrowserActorPoisonedError, match="终态不确定"):
        actor._action_sync(
            "download",
            lambda: actor._download_op("report"),
            "report",
        )
    assert download_waits == 1
    assert actor.poisoned is True
    actor._closed = True
    actor._executor.shutdown(wait=True, cancel_futures=True)


def test_internal_failure_after_mutation_dispatch_poisons_actor(tmp_path) -> None:
    actor = BrowserActor("http://127.0.0.1:9", tmp_path, EvidenceStore())
    page = SimpleNamespace(url="https://example.test", is_closed=lambda: False)
    actor._owner_thread = threading.get_ident()
    actor._connected = True
    actor._page = page
    actor._context = SimpleNamespace(pages=[page])
    actor._semantic_page_state = lambda: {  # type: ignore[method-assign]
        "url": page.url,
        "title": "before",
        "semantic_page_fingerprint": "before-state",
        "semantic_element_count": 1,
    }

    def fail_after_dispatch() -> dict:
        return actor._dispatch_mutation(lambda: (_ for _ in ()).throw(TimeoutError("uncertain")))

    with pytest.raises(BrowserActorPoisonedError, match="终态不确定"):
        actor._action_sync("click", fail_after_dispatch, None)
    assert actor.poisoned is True
    actor._closed = True
    actor._executor.shutdown(wait=True, cancel_futures=True)


def test_failure_after_actionability_trial_still_poisons_actor(tmp_path) -> None:
    from playwright.sync_api import TimeoutError as PlaywrightTimeoutError

    actor = BrowserActor("http://127.0.0.1:9", tmp_path, EvidenceStore())
    calls: list[bool] = []

    class Locator:
        def evaluate(self, expression: str) -> dict[str, object]:  # noqa: ARG002
            return {"bid": "abc", "tag": "button", "type": "button", "text": "Open"}

        def click(self, timeout: int, trial: bool = False) -> None:  # noqa: ARG002
            calls.append(trial)
            if not trial:
                raise PlaywrightTimeoutError(
                    "Locator.click: Timeout exceeded.\n"
                    "- element is not visible\n"
                    "- performing click action\n"
                    "- waiting for scheduled navigations to finish"
                )

    page = SimpleNamespace(url="https://example.test", is_closed=lambda: False)
    actor._owner_thread = threading.get_ident()
    actor._connected = True
    actor._page = page
    actor._context = SimpleNamespace(pages=[page])
    actor._locator = lambda bid: Locator()  # type: ignore[method-assign]
    actor._semantic_page_state = lambda: {  # type: ignore[method-assign]
        "url": page.url,
        "title": "page",
        "semantic_page_fingerprint": "same",
        "semantic_element_count": 1,
    }

    with pytest.raises(BrowserActorPoisonedError, match="终态不确定"):
        actor._action_sync("click", lambda: actor._click_op("abc"), "abc")
    assert calls == [True, False]
    assert actor.poisoned is True
    actor._closed = True
    actor._executor.shutdown(wait=True, cancel_futures=True)


@pytest.mark.asyncio
async def test_dispatched_timeout_poisons_actor_and_drops_late_event(tmp_path) -> None:
    actor = BrowserActor("http://127.0.0.1:9", tmp_path, EvidenceStore())
    finished = threading.Event()

    class BrokenResponse:
        @property
        def headers(self):
            raise RuntimeError("late response")

    def dispatched_then_late() -> str:
        time.sleep(0.08)
        actor._on_response(BrokenResponse())
        finished.set()
        return "side effect may have happened"

    with pytest.raises(ActorCallDeadlineExceeded) as caught:
        await actor._call(
            dispatched_then_late,
            deadline=time.monotonic() + 0.02,
            operation="click",
        )
    assert caught.value.dispatched is True
    assert actor.poisoned is True
    assert await asyncio.to_thread(finished.wait, 1)
    assert actor._network_records == []
    with pytest.raises(BrowserActorPoisonedError, match="ACTOR_POISONED"):
        await actor._call(lambda: "must-not-run")
    tmp_path.mkdir(exist_ok=True)
    await actor.close()


@pytest.mark.asyncio
async def test_outer_cancellation_after_dispatch_also_poisons_actor(tmp_path) -> None:
    actor = BrowserActor("http://127.0.0.1:9", tmp_path, EvidenceStore())
    release = threading.Event()
    started = threading.Event()

    def blocking_action() -> None:
        started.set()
        release.wait(timeout=1)

    call = asyncio.create_task(actor._call(blocking_action, operation="upload"))
    assert await asyncio.to_thread(started.wait, 1)
    call.cancel()
    with pytest.raises(asyncio.CancelledError):
        await call
    assert actor.poisoned is True
    release.set()
    tmp_path.mkdir(exist_ok=True)
    await actor.close()


@pytest.mark.asyncio
async def test_retire_waits_for_late_poisoned_call_before_disconnecting(tmp_path) -> None:
    actor = BrowserActor("http://127.0.0.1:9", tmp_path, EvidenceStore())
    started = threading.Event()
    release = threading.Event()

    def blocking_action() -> None:
        started.set()
        release.wait(timeout=1)

    call = asyncio.create_task(actor._call(blocking_action, operation="click"))
    assert await asyncio.to_thread(started.wait, 1)
    call.cancel()
    with pytest.raises(asyncio.CancelledError):
        await call
    assert actor.poisoned is True

    retiring = asyncio.create_task(actor.retire())
    await asyncio.sleep(0.02)
    assert retiring.done() is False
    release.set()
    await asyncio.wait_for(retiring, timeout=1)
    assert actor._closed is True


@pytest.mark.asyncio
async def test_old_generation_event_is_ignored_after_task_switch(tmp_path) -> None:
    actor = BrowserActor("http://127.0.0.1:9", tmp_path, EvidenceStore())
    old_generation = actor.task_generation
    new_generation = actor._reserve_generation()
    assert new_generation != old_generation

    class BrokenResponse:
        @property
        def headers(self):
            raise RuntimeError("old task response")

    await actor._call(actor._on_response, BrokenResponse(), old_generation)
    assert actor._network_records == []
    tmp_path.mkdir(exist_ok=True)
    await actor.close()


@pytest.mark.asyncio
async def test_actor_evidence_is_bound_to_generation_and_attempt(tmp_path) -> None:
    store = EvidenceStore()
    actor = BrowserActor("http://127.0.0.1:9", tmp_path, store)

    def add_evidence():
        return store.add("dom", "https://example.test", "bound", {"value": 1})

    evidence = await actor._call(add_evidence)
    assert evidence.payload["task_generation"] == actor.task_generation
    assert evidence.payload["attempt"] == 1
    tmp_path.mkdir(exist_ok=True)
    await actor.close()


@pytest.mark.parametrize("event", ["crash", "disconnect"])
def test_page_crash_and_cdp_disconnect_poison_actor(tmp_path, event: str) -> None:
    actor = BrowserActor("http://127.0.0.1:9", tmp_path, EvidenceStore())
    generation = actor.task_generation
    if event == "crash":
        actor._on_page_crash(generation)
        assert "PAGE_CRASHED" in (actor.poisoned_reason or "")
    else:
        actor._connected = True
        actor._on_disconnected(generation)
        assert "CDP_DISCONNECTED" in (actor.poisoned_reason or "")
    assert actor.poisoned is True
    actor._closed = True
    actor._executor.shutdown(wait=True, cancel_futures=True)
