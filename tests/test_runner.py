from __future__ import annotations

import json

import pytest

from web_agent.runner import (
    _aggregate_usage,
    _distribution,
    _kimi_tpm_budget,
    atomic_write_json,
    is_completed,
    task_directory,
)


def test_result_resume_requires_verified_answer(tmp_path) -> None:
    item = {"task_idx": 7, "task_id": "abc"}
    directory = task_directory(tmp_path, item)
    directory.mkdir()
    atomic_write_json(directory / "result.json", {"status": "SUCCESS", "agent_answer": None})
    assert is_completed(directory) is False
    atomic_write_json(directory / "result.json", {"status": "SUCCESS", "agent_answer": "answer"})
    assert is_completed(directory) is True


def test_task_directory_is_stable(tmp_path) -> None:
    assert task_directory(tmp_path, {"task_idx": 2, "task_id": "id"}) == tmp_path / "2_id"


def test_worker_limit_rejected_before_browser_connection(tmp_path) -> None:
    from web_agent.runner import run_tasks

    input_path = tmp_path / "tasks.json"
    input_path.write_text(json.dumps([]), encoding="utf-8")
    with pytest.raises(ValueError, match="1 到 8"):
        run_tasks(input_path, tmp_path / "out", ["x"] * 9, "m", "http://model", "key")


def test_duplicate_cdp_urls_are_rejected_for_worker_isolation(tmp_path) -> None:
    from web_agent.runner import run_tasks

    input_path = tmp_path / "tasks.json"
    input_path.write_text(json.dumps([]), encoding="utf-8")
    with pytest.raises(ValueError, match="独立的 CDP URL"):
        run_tasks(input_path, tmp_path / "out", ["same", "same"], "m", "http://model", "key")


def test_duplicate_tasks_are_rejected_before_worker_start(tmp_path) -> None:
    from web_agent.runner import run_tasks

    input_path = tmp_path / "tasks.json"
    input_path.write_text(json.dumps([{"task_idx": 1, "task_id": "x"}] * 2), encoding="utf-8")
    with pytest.raises(ValueError, match="重复"):
        run_tasks(input_path, tmp_path / "out", ["cdp"], "m", "http://model", "key")


def test_kimi_tpm_safety_budget_is_configurable_and_validated(monkeypatch) -> None:
    assert _kimi_tpm_budget("gpt-4.1-mini") is None
    monkeypatch.delenv("MOONSHOT_TPM_LIMIT", raising=False)
    monkeypatch.delenv("MOONSHOT_TPM_SAFETY_RATIO", raising=False)
    assert _kimi_tpm_budget("kimi-k2.6") == 2_400_000
    monkeypatch.setenv("MOONSHOT_TPM_LIMIT", "1000")
    monkeypatch.setenv("MOONSHOT_TPM_SAFETY_RATIO", "0.5")
    assert _kimi_tpm_budget("provider/kimi-k2.6") == 500
    monkeypatch.setenv("MOONSHOT_TPM_SAFETY_RATIO", "broken")
    with pytest.raises(ValueError, match="合法数字"):
        _kimi_tpm_budget("kimi-k2.6")


def test_usage_aggregation_preserves_task_worker_totals_and_throttle_reasons() -> None:
    summary = _aggregate_usage(
        [
            {
                "model_usage": {
                    "request_count": 2,
                    "usage_available_count": 1,
                    "usage_unavailable_count": 1,
                    "input_tokens": 100,
                    "output_tokens": 10,
                    "total_tokens": 110,
                    "estimated_input_tokens": 300,
                    "throttle_wait_seconds": 2.5,
                    "throttle_reasons": {"pre_send_tpm_capacity": 1},
                }
            },
            {
                "model_usage": {
                    "request_count": 1,
                    "usage_available_count": 1,
                    "usage_unavailable_count": 0,
                    "input_tokens": 50,
                    "output_tokens": 5,
                    "total_tokens": 55,
                    "estimated_input_tokens": 100,
                    "throttle_wait_seconds": 0,
                    "throttle_reasons": {},
                }
            },
        ]
    )
    assert summary["task_count"] == 2
    assert summary["request_count"] == 3
    assert summary["input_tokens"] == 150
    assert summary["throttle_reasons"] == {"pre_send_tpm_capacity": 1}
    assert summary["request_count_per_task"] == {"p50": 1, "p95": 2, "max": 2}


def test_summary_distribution_reports_p50_p95_and_max() -> None:
    assert _distribution(list(range(1, 101))) == {"p50": 50, "p95": 95, "max": 100}
    assert _distribution([1, 100]) == {"p50": 1, "p95": 100, "max": 100}
    assert _distribution(list(range(1, 9))) == {"p50": 4, "p95": 8, "max": 8}
    assert _distribution([]) == {"p50": None, "p95": None, "max": None}


def test_worker_start_failure_cleans_up_started_process_and_queue(tmp_path, monkeypatch) -> None:
    from web_agent.runner import run_tasks

    class FakeQueue:
        def __init__(self) -> None:
            self.closed = False
            self.joined = False

        def close(self) -> None:
            self.closed = True

        def join_thread(self) -> None:
            self.joined = True

    class FakeProcess:
        def __init__(self, fail_start: bool = False) -> None:
            self.fail_start = fail_start
            self.alive = False
            self.terminated = False
            self.joined = False

        def start(self) -> None:
            if self.fail_start:
                raise RuntimeError("start failed")
            self.alive = True

        def is_alive(self) -> bool:
            return self.alive

        def terminate(self) -> None:
            self.terminated = True
            self.alive = False

        def join(self) -> None:
            self.joined = True

    queue = FakeQueue()
    first = FakeProcess()
    second = FakeProcess(fail_start=True)

    class FakeContext:
        def __init__(self) -> None:
            self.processes = iter((first, second))

        def Queue(self) -> FakeQueue:  # noqa: N802
            return queue

        def Process(self, **kwargs) -> FakeProcess:  # noqa: N802, ARG002
            return next(self.processes)

    monkeypatch.setattr("web_agent.runner.mp.get_context", lambda method: FakeContext())
    input_path = tmp_path / "tasks.json"
    input_path.write_text(
        json.dumps(
            [
                {"task_idx": 1, "task_id": "a"},
                {"task_idx": 2, "task_id": "b"},
            ]
        ),
        encoding="utf-8",
    )
    with pytest.raises(RuntimeError, match="start failed"):
        run_tasks(
            input_path,
            tmp_path / "out",
            ["cdp-1", "cdp-2"],
            "test-model",
            "http://model",
            "key",
        )
    assert first.terminated is True
    assert first.joined is True
    assert queue.closed is True and queue.joined is True
