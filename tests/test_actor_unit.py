from __future__ import annotations

import asyncio
import hashlib
import threading
from types import SimpleNamespace

import pytest

from web_agent.browser_actor import BrowserActor, _redact, _safe_post_data, _sanitize_headers
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
