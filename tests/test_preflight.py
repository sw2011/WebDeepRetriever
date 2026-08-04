from __future__ import annotations

import json
import threading
from functools import partial
from types import SimpleNamespace

import pytest

from web_agent.preflight import (
    PreflightProbeError,
    _probe_cdp,
    _probe_existing_pages,
    run_preflight,
    validate_cdp_urls,
)


def healthy_worker(url: str, worker_id: int) -> dict:
    return {
        "worker_id": worker_id,
        "status": "ok",
        "code": "PREFLIGHT_OK",
        "endpoint": url,
        "browser": "chromium",
        "chrome": "Chrome/130.0.0.0",
        "chrome_major": "130",
        "transport": "playwright_cdp",
        "context_mode": "shared_existing",
        "context_count": 1,
        "page_count": 1,
        "capabilities": {
            "browser": True,
            "context": True,
            "page": True,
            "dom": True,
            "accessibility": True,
            "cdp": True,
        },
    }


def blocking_worker(url: str, worker_id: int) -> dict:  # noqa: ARG001
    threading.Event().wait()
    raise AssertionError("unreachable")


def failing_worker(url: str, worker_id: int, *, code: str) -> dict:  # noqa: ARG001
    raise PreflightProbeError(code, "probe failed")


def leaking_failure(url: str, worker_id: int) -> dict:  # noqa: ARG001
    raise RuntimeError(f"connection refused: {url}")


def partial_worker(url: str, worker_id: int) -> dict:
    if worker_id == 1:
        raise PreflightProbeError("PREFLIGHT_CDP_CONNECT_FAILED", "refused")
    return healthy_worker(url, worker_id)


def mismatched_worker(url: str, worker_id: int) -> dict:
    worker = healthy_worker(url, worker_id)
    if worker_id == 1:
        worker["chrome_major"] = "131"
        worker["chrome"] = "Chrome/131.0.0.0"
    return worker


def test_preflight_validates_url_shape_and_browser_identity() -> None:
    _, websocket_error = validate_cdp_urls(["ws://localhost:9222/devtools/browser/id"])
    assert websocket_error is None
    _, invalid = validate_cdp_urls(["not-a-url"])
    assert invalid and invalid["code"] == "PREFLIGHT_INVALID_CDP_URL"
    _, duplicate = validate_cdp_urls(
        [
            "http://localhost:9222?access_token=first",
            "http://LOCALHOST:9222/?access_token=second",
        ]
    )
    assert duplicate and duplicate["code"] == "PREFLIGHT_DUPLICATE_CDP_URL"


def test_invalid_url_still_writes_audit_report(tmp_path) -> None:
    report = run_preflight(["not-a-url"], tmp_path)
    persisted = json.loads((tmp_path / "logs" / "preflight.json").read_text(encoding="utf-8"))
    assert report["code"] == persisted["code"] == "PREFLIGHT_INVALID_CDP_URL"


@pytest.mark.parametrize("timeout", [float("nan"), float("inf"), float("-inf"), 0.0, -1.0])
def test_preflight_rejects_non_finite_or_non_positive_probe_timeout(tmp_path, timeout: float) -> None:
    report = run_preflight(
        ["http://localhost:9222"],
        tmp_path,
        probe=healthy_worker,
        probe_timeout_seconds=timeout,
    )
    assert report["status"] == "error"
    assert report["code"] == "PREFLIGHT_INVALID_PROBE_TIMEOUT"
    persisted = json.loads((tmp_path / "logs" / "preflight.json").read_text(encoding="utf-8"))
    assert persisted["code"] == "PREFLIGHT_INVALID_PROBE_TIMEOUT"


def test_preflight_success_is_auditable_and_writes_atomically(tmp_path) -> None:
    report = run_preflight(
        ["http://localhost:9222", "http://localhost:9223"],
        tmp_path,
        probe=healthy_worker,
    )
    assert report["status"] == "ok"
    assert report["code"] == "PREFLIGHT_OK"
    assert report["model_requests"] == 0
    assert report["target_navigation"] is False
    assert report["runtime"]["python"]
    assert report["runtime"]["playwright"]
    assert report["capability_signature"]["transport"] == "playwright_cdp"
    persisted = json.loads((tmp_path / "logs" / "preflight.json").read_text(encoding="utf-8"))
    assert persisted["status"] == "ok"


@pytest.mark.parametrize(
    "worker_code",
    [
        "PREFLIGHT_CDP_CONNECT_FAILED",
        "PREFLIGHT_CDP_DISCONNECTED",
        "PREFLIGHT_MISSING_CAPABILITY",
        "PREFLIGHT_NO_CONTEXT",
        "PREFLIGHT_NO_PAGE",
    ],
)
def test_preflight_preserves_stable_worker_failure_codes(tmp_path, worker_code: str) -> None:
    report = run_preflight(
        ["http://localhost:9222"],
        tmp_path,
        probe=partial(failing_worker, code=worker_code),
    )
    assert report["status"] == "error"
    assert report["code"] == "PREFLIGHT_WORKER_FAILED"
    assert report["workers"][0]["code"] == worker_code


def test_preflight_rejects_read_only_output_directory(tmp_path) -> None:
    output = tmp_path / "readonly"
    output.mkdir()
    output.chmod(0o555)
    try:
        report = run_preflight(["http://localhost:9222"], output, probe=healthy_worker)
    finally:
        output.chmod(0o755)
    assert report["status"] == "error"
    assert report["code"] == "PREFLIGHT_OUTPUT_NOT_WRITABLE"


def test_preflight_rejects_logs_path_that_is_not_a_directory(tmp_path) -> None:
    output = tmp_path / "output"
    output.mkdir()
    (output / "logs").write_text("not a directory", encoding="utf-8")

    report = run_preflight(["http://localhost:9222"], output, probe=healthy_worker)

    assert report["status"] == "error"
    assert report["code"] == "PREFLIGHT_OUTPUT_NOT_WRITABLE"


def test_preflight_rejects_read_only_logs_directory(tmp_path) -> None:
    logs = tmp_path / "output" / "logs"
    logs.mkdir(parents=True)
    logs.chmod(0o555)
    try:
        report = run_preflight(["http://localhost:9222"], tmp_path / "output", probe=healthy_worker)
    finally:
        logs.chmod(0o755)

    assert report["status"] == "error"
    assert report["code"] == "PREFLIGHT_OUTPUT_NOT_WRITABLE"


def test_preflight_rejects_read_only_output_even_when_logs_is_writable(tmp_path) -> None:
    output = tmp_path / "output"
    logs = output / "logs"
    logs.mkdir(parents=True)
    output.chmod(0o555)
    try:
        report = run_preflight(["http://localhost:9222"], output, probe=healthy_worker)
    finally:
        output.chmod(0o755)

    assert report["status"] == "error"
    assert report["code"] == "PREFLIGHT_OUTPUT_NOT_WRITABLE"


def test_preflight_redacts_access_tokens_from_urls_and_diagnostics(tmp_path) -> None:
    secret = "preflight-super-secret"

    report = run_preflight(
        [f"http://localhost:9222/json?access_token={secret}"],
        tmp_path,
        probe=leaking_failure,
    )
    assert secret not in json.dumps(report)
    assert "%5BREDACTED%5D" in json.dumps(report)


def test_preflight_reports_partial_worker_failure(tmp_path) -> None:
    report = run_preflight(
        ["http://localhost:9222", "http://localhost:9223"],
        tmp_path,
        probe=partial_worker,
    )
    assert report["code"] == "PREFLIGHT_WORKER_FAILED"
    assert [worker["status"] for worker in report["workers"]] == ["ok", "error"]


def test_preflight_hard_deadline_terminates_blocked_probe(tmp_path) -> None:
    report = run_preflight(
        ["http://localhost:9222"],
        tmp_path,
        probe=blocking_worker,
        probe_timeout_seconds=0.05,
    )

    assert report["status"] == "error"
    assert report["code"] == "PREFLIGHT_WORKER_FAILED"
    assert report["workers"][0]["code"] == "PREFLIGHT_PROBE_TIMEOUT"


def test_preflight_rejects_worker_capability_mismatch(tmp_path) -> None:
    report = run_preflight(
        ["http://localhost:9222", "http://localhost:9223"],
        tmp_path,
        probe=mismatched_worker,
    )
    assert report["status"] == "error"
    assert report["code"] == "PREFLIGHT_CAPABILITY_MISMATCH"


def test_preflight_uses_temporary_page_in_first_context_only() -> None:
    class FakePage:
        def __init__(self, healthy: bool) -> None:
            self.healthy = healthy
            self.closed = False

        def is_closed(self) -> bool:
            return self.closed

        def evaluate(self, script: str) -> dict:  # noqa: ARG002
            if not self.healthy:
                raise RuntimeError("page unavailable")
            return {"readyState": "complete", "hasDocument": True}

        def close(self, run_before_unload: bool = False) -> None:  # noqa: ARG002
            self.closed = True

    class FakeSession:
        def __init__(self, healthy: bool) -> None:
            self.healthy = healthy
            self.detached = False

        def send(self, method: str, params=None) -> dict:  # noqa: ANN001
            if not self.healthy:
                raise RuntimeError("capability unavailable")
            return {
                "DOM.getDocument": {"root": {"nodeId": 1}},
                "Accessibility.getFullAXTree": {"nodes": []},
                "Browser.getVersion": {"product": "Chrome/130.0.0.0"},
            }[method]

        def detach(self) -> None:
            self.detached = True

    existing = FakePage(True)
    temporary = FakePage(True)
    sessions: list[FakeSession] = []

    def new_session(page: FakePage) -> FakeSession:
        session = FakeSession(page.healthy)
        sessions.append(session)
        return session

    contexts = [
        SimpleNamespace(pages=[existing], new_page=lambda: temporary, new_cdp_session=new_session),
        SimpleNamespace(pages=[FakePage(True)], new_page=lambda: FakePage(True), new_cdp_session=new_session),
    ]
    browser = SimpleNamespace(contexts=contexts, version="130.0.0.0", is_connected=lambda: True)

    probe = _probe_existing_pages(browser)

    assert probe["context_count"] == 2
    assert probe["page_count"] == 2
    assert probe["selected_context_index"] == 0
    assert probe["selected_page"] == "temporary_about_blank"
    assert temporary.closed is True
    assert all(session.detached for session in sessions)


def test_preflight_fails_when_first_context_cannot_create_page_even_if_later_context_is_healthy() -> None:
    def fail_new_page():
        raise RuntimeError("first context unavailable")

    first = SimpleNamespace(pages=[], new_page=fail_new_page)
    later = SimpleNamespace(pages=[], new_page=lambda: SimpleNamespace())
    browser = SimpleNamespace(contexts=[first, later], version="130", is_connected=lambda: True)

    with pytest.raises(PreflightProbeError) as caught:
        _probe_existing_pages(browser)

    assert caught.value.code == "PREFLIGHT_NO_PAGE"


def test_probe_cdp_maps_connection_refusal_to_stable_code(monkeypatch) -> None:
    class Chromium:
        def connect_over_cdp(self, url: str, **kwargs):  # noqa: ANN003, ANN201, ARG002
            raise ConnectionRefusedError("refused access_token=private")

    playwright = SimpleNamespace(chromium=Chromium(), stop=lambda: None)
    monkeypatch.setattr("playwright.sync_api.sync_playwright", lambda: SimpleNamespace(start=lambda: playwright))

    with pytest.raises(PreflightProbeError) as caught:
        _probe_cdp("http://localhost:9222?access_token=private", 0)

    assert caught.value.code == "PREFLIGHT_CDP_CONNECT_FAILED"
    assert "private" not in str(caught.value)


def test_page_probe_maps_disconnect_during_capability_check() -> None:
    class Session:
        def send(self, method: str, params=None) -> dict:  # noqa: ANN001
            return {
                "DOM.getDocument": {"root": {"nodeId": 1}},
                "Accessibility.getFullAXTree": {"nodes": []},
                "Browser.getVersion": {"product": "Chrome/130.0.0.0"},
            }[method]

        def detach(self) -> None:
            pass

    page = SimpleNamespace(
        is_closed=lambda: False,
        evaluate=lambda script: {"readyState": "complete", "hasDocument": True},
        close=lambda run_before_unload=False: None,
    )
    context = SimpleNamespace(pages=[], new_page=lambda: page, new_cdp_session=lambda candidate: Session())
    browser = SimpleNamespace(contexts=[context], version="130", is_connected=lambda: False)

    with pytest.raises(PreflightProbeError) as caught:
        _probe_existing_pages(browser)

    assert caught.value.code == "PREFLIGHT_CDP_DISCONNECTED"
