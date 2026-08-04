from __future__ import annotations

import json
import sys

import pytest

from web_agent import cli


def test_healthcheck_is_real_preflight_alias_and_returns_nonzero_on_failure(
    monkeypatch,
    capsys,
) -> None:
    monkeypatch.setattr(
        cli,
        "run_preflight",
        lambda urls, output: {
            "status": "error",
            "code": "PREFLIGHT_CDP_CONNECT_FAILED",
            "model_requests": 0,
            "target_navigation": False,
        },
    )
    monkeypatch.setattr(
        sys,
        "argv",
        ["webdeepretriever", "--healthcheck", "--cdp_url", "http://localhost:9222"],
    )
    with pytest.raises(SystemExit) as caught:
        cli.main()
    assert caught.value.code == 2
    assert json.loads(capsys.readouterr().out)["code"] == "PREFLIGHT_CDP_CONNECT_FAILED"


def test_preflight_without_cdp_url_returns_stable_audited_failure(tmp_path, monkeypatch, capsys) -> None:
    output = tmp_path / "output"
    monkeypatch.setattr(
        sys,
        "argv",
        ["webdeepretriever", "--preflight", "--output", str(output)],
    )

    with pytest.raises(SystemExit) as caught:
        cli.main()

    assert caught.value.code == 2
    printed = json.loads(capsys.readouterr().out)
    persisted = json.loads((output / "logs" / "preflight.json").read_text(encoding="utf-8"))
    assert printed["code"] == persisted["code"] == "PREFLIGHT_CDP_WORKER_COUNT"


def test_normal_cli_runs_preflight_before_runner_and_forwards_rerun_controls(
    tmp_path,
    monkeypatch,
    capsys,
) -> None:
    calls: list[str] = []

    def preflight(urls, output):  # noqa: ANN001
        calls.append("preflight")
        return {"status": "ok", "code": "PREFLIGHT_OK"}

    def runner(*args, **kwargs):  # noqa: ANN002, ANN003
        calls.append("runner")
        assert kwargs["force_rerun"] is True
        assert kwargs["worker_watchdog_seconds"] == 12.5
        return {"total": 0, "completed": 0}

    monkeypatch.setattr(cli, "run_preflight", preflight)
    monkeypatch.setattr(cli, "run_tasks", runner)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "webdeepretriever",
            "--input",
            str(tmp_path / "tasks.json"),
            "--output",
            str(tmp_path / "out"),
            "--cdp_url",
            "http://localhost:9222",
            "--api_key",
            "test-key",
            "--force_rerun",
            "--worker_watchdog_seconds",
            "12.5",
        ],
    )
    cli.main()
    assert calls == ["preflight", "runner"]
    assert json.loads(capsys.readouterr().out)["total"] == 0
