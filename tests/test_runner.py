from __future__ import annotations

import json
from queue import Empty

import pytest

from web_agent.runner import (
    _aggregate_usage,
    _distribution,
    _kimi_tpm_budget,
    atomic_write_json,
    is_completed,
    task_directory,
)
from web_agent.run_manifest import build_run_manifest, load_manifest, manifest_matches


def test_result_resume_requires_verified_answer(tmp_path) -> None:
    item = {"task_idx": 7, "task_id": "abc"}
    directory = task_directory(tmp_path, item)
    directory.mkdir()
    atomic_write_json(directory / "result.json", {"status": "SUCCESS", "agent_answer": None})
    assert is_completed(directory, "fp", "run-a", manifest_valid=True) is False
    atomic_write_json(
        directory / "result.json",
        {"status": "SUCCESS", "agent_answer": "answer", "run_fingerprint": "fp", "run_id": "run-a"},
    )
    assert is_completed(directory) is False
    assert is_completed(directory, "fp", "run-a", manifest_valid=False) is False
    assert is_completed(directory, "other", "run-a", manifest_valid=True) is False
    assert is_completed(directory, "fp", "run-b", manifest_valid=True) is False
    assert is_completed(directory, "fp", "run-a", manifest_valid=True) is True
    atomic_write_json(
        directory / "result.json",
        {"status": "SUCCESS", "agent_answer": "   ", "run_fingerprint": "fp", "run_id": "run-a"},
    )
    assert is_completed(directory, "fp", "run-a", manifest_valid=True) is False
    atomic_write_json(directory / "result.json", [])
    assert is_completed(directory, "fp", "run-a", manifest_valid=True) is False
    atomic_write_json(
        directory / "result.json",
        {"status": "SUCCESS", "agent_answer": 0, "run_fingerprint": "fp", "run_id": "run-a"},
    )
    assert is_completed(directory, "fp", "run-a", manifest_valid=True) is True


def test_missing_or_corrupt_manifest_never_authorizes_resume(tmp_path) -> None:
    path = tmp_path / "run_manifest.json"
    assert load_manifest(path) is None
    path.write_text("{broken", encoding="utf-8")
    assert load_manifest(path) is None
    path.write_text(json.dumps({"schema_version": 1}), encoding="utf-8")
    assert load_manifest(path) is None
    assert manifest_matches(None, {"fingerprint": "fp"}) is False
    assert manifest_matches({"fingerprint": "fp", "schema_version": 1}, {"fingerprint": "fp"}) is False

    dataset = tmp_path / "tasks.json"
    dataset.write_text("[]", encoding="utf-8")
    valid = build_run_manifest(dataset, ["http://localhost:9222"], "m", "https://api.test/v1", 10, 0, 60)
    atomic_write_json(path, valid)
    assert load_manifest(path) is not None
    valid["model"]["name"] = "tampered"
    atomic_write_json(path, valid)
    assert load_manifest(path) is None


def test_run_manifest_fingerprint_tracks_reproducibility_inputs(tmp_path, monkeypatch) -> None:
    import web_agent.run_manifest as manifest_module

    dataset = tmp_path / "tasks.json"
    dataset.write_text('[{"task_idx":1}]', encoding="utf-8")
    original_prompt = manifest_module.SYSTEM_INSTRUCTIONS
    original_tools = manifest_module.TOOLS

    def build(**overrides):
        values = {
            "input_path": dataset,
            "cdp_urls": ["http://127.0.0.1:9222?access_token=manifest-secret"],
            "model": "model-a",
            "api_base": "https://api.example.test/v1?access_token=manifest-secret",
            "max_steps": 50,
            "worker_count": 1,
            "worker_watchdog_seconds": 900.0,
        }
        values.update(overrides)
        return build_run_manifest(**values)

    baseline = build()
    repeated = build()
    assert baseline["fingerprint"] == repeated["fingerprint"]
    assert baseline["created_at"] != ""
    assert "sha" in baseline["git"] and "dirty" in baseline["git"]
    assert "manifest-secret" not in json.dumps(baseline)

    assert build(model="model-b")["fingerprint"] != baseline["fingerprint"]
    assert build(max_steps=51)["fingerprint"] != baseline["fingerprint"]
    assert build(worker_watchdog_seconds=901.0)["fingerprint"] != baseline["fingerprint"]

    dataset.write_text('[{"task_idx":2}]', encoding="utf-8")
    assert build()["fingerprint"] != baseline["fingerprint"]
    dataset.write_text('[{"task_idx":1}]', encoding="utf-8")

    monkeypatch.setattr(manifest_module, "SYSTEM_INSTRUCTIONS", "changed prompt")
    assert build()["fingerprint"] != baseline["fingerprint"]
    monkeypatch.setattr(manifest_module, "SYSTEM_INSTRUCTIONS", original_prompt)

    class FakeTool:
        name = "changed_tool"
        params_json_schema = {"type": "object", "properties": {"new": {"type": "string"}}}

    monkeypatch.setattr(manifest_module, "TOOLS", [FakeTool()])
    assert build()["fingerprint"] != baseline["fingerprint"]
    monkeypatch.setattr(manifest_module, "TOOLS", original_tools)


@pytest.mark.parametrize("timeout", [float("nan"), float("inf"), float("-inf"), 0.0, -1.0])
def test_run_manifest_rejects_non_finite_or_non_positive_watchdog(tmp_path, timeout: float) -> None:
    dataset = tmp_path / "tasks.json"
    dataset.write_text("[]", encoding="utf-8")
    with pytest.raises(ValueError, match="有限正数"):
        build_run_manifest(
            dataset,
            ["http://localhost:9222"],
            "model",
            "https://api.test/v1",
            10,
            0,
            timeout,
        )


def test_run_manifest_tracks_source_changes(tmp_path, monkeypatch) -> None:
    import web_agent.run_manifest as manifest_module

    dataset = tmp_path / "tasks.json"
    dataset.write_text("[]", encoding="utf-8")
    monkeypatch.setattr(manifest_module, "_source_hash", lambda: "source-a")
    first = build_run_manifest(dataset, ["http://localhost:9222"], "m", "https://api.test/v1", 10, 0, 60)
    monkeypatch.setattr(manifest_module, "_source_hash", lambda: "source-b")
    second = build_run_manifest(dataset, ["http://localhost:9222"], "m", "https://api.test/v1", 10, 0, 60)
    assert first["fingerprint"] != second["fingerprint"]


def test_source_hash_uses_loaded_package_locations_without_checkout(tmp_path, monkeypatch) -> None:
    import web_agent.run_manifest as manifest_module

    web_agent_root = tmp_path / "site-packages" / "web_agent"
    browsergym_root = tmp_path / "site-packages" / "browsergym"
    web_agent_root.mkdir(parents=True)
    browsergym_root.mkdir(parents=True)
    (web_agent_root / "runner.py").write_text("VALUE = 1\n", encoding="utf-8")
    (browsergym_root / "observation.py").write_text("VALUE = 2\n", encoding="utf-8")
    monkeypatch.setattr(manifest_module, "_PACKAGE_ROOT", web_agent_root)
    monkeypatch.setattr(
        manifest_module.importlib.util,
        "find_spec",
        lambda name: type("Spec", (), {"submodule_search_locations": [str(browsergym_root)]})(),
    )

    first = manifest_module._source_hash()
    (browsergym_root / "observation.py").write_text("VALUE = 3\n", encoding="utf-8")
    second = manifest_module._source_hash()

    assert first != second
    assert first != "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"


class ImmediateQueue:
    def get(self, timeout=None):  # noqa: ANN001, ARG002
        raise Empty

    def get_nowait(self):
        raise Empty

    def close(self) -> None:
        pass

    def join_thread(self) -> None:
        pass


class ImmediateProcess:
    exitcode = 0

    def __init__(self, kwargs: dict, on_start=None) -> None:  # noqa: ANN001
        self.kwargs = kwargs
        self.on_start = on_start

    def start(self) -> None:
        if self.on_start is not None:
            self.on_start(self.kwargs)

    def is_alive(self) -> bool:
        return False

    def join(self, timeout=None) -> None:  # noqa: ANN001, ARG002
        pass


class ImmediateContext:
    def __init__(self, on_start=None) -> None:  # noqa: ANN001
        self.on_start = on_start
        self.process_count = 0

    def Queue(self) -> ImmediateQueue:  # noqa: N802
        return ImmediateQueue()

    def Process(self, **kwargs) -> ImmediateProcess:  # noqa: N802, ANN003
        self.process_count += 1
        return ImmediateProcess(kwargs, self.on_start)


def _write_resumable_result(output, item, fingerprint: str, run_id: str) -> None:  # noqa: ANN001
    directory = task_directory(output, item)
    directory.mkdir(parents=True, exist_ok=True)
    atomic_write_json(
        directory / "result.json",
        {
            "task_idx": item["task_idx"],
            "task_id": item["task_id"],
            "status": "SUCCESS",
            "agent_answer": "answer",
            "run_fingerprint": fingerprint,
            "run_id": run_id,
        },
    )


def test_run_tasks_reuses_matching_manifest_without_starting_worker(tmp_path, monkeypatch) -> None:
    import web_agent.run_manifest as manifest_module
    from web_agent.runner import run_tasks

    monkeypatch.setattr(manifest_module, "_source_hash", lambda: "source-a")
    item = {"task_idx": 1, "task_id": "a", "task": "query"}
    dataset = tmp_path / "tasks.json"
    dataset.write_text(json.dumps([item]), encoding="utf-8")
    output = tmp_path / "output"
    output.mkdir()
    manifest = build_run_manifest(
        dataset,
        ["http://localhost:9222"],
        "model-a",
        "https://api.test/v1",
        100,
        1,
        900,
    )
    atomic_write_json(output / "run_manifest.json", manifest)
    _write_resumable_result(output, item, str(manifest["fingerprint"]), str(manifest["run_id"]))
    monkeypatch.setattr(
        "web_agent.runner.mp.get_context",
        lambda method: (_ for _ in ()).throw(AssertionError("不应启动 worker")),
    )

    summary = run_tasks(
        dataset,
        output,
        ["http://localhost:9222"],
        "model-a",
        "https://api.test/v1",
        "unused",
    )

    assert summary["pending"] == 0
    assert summary["reused"] == 1
    assert summary["run_id"] == manifest["run_id"]


@pytest.mark.parametrize(
    "change",
    ["code", "prompt", "model", "dataset", "config", "corrupt_manifest", "force_rerun"],
)
def test_run_tasks_reruns_when_reproducibility_identity_changes(tmp_path, monkeypatch, change: str) -> None:
    import web_agent.run_manifest as manifest_module
    from web_agent.runner import run_tasks

    monkeypatch.setattr(manifest_module, "_source_hash", lambda: "source-a")
    item = {"task_idx": 1, "task_id": "a", "task": "query"}
    dataset = tmp_path / "tasks.json"
    dataset.write_text(json.dumps([item]), encoding="utf-8")
    output = tmp_path / "output"
    output.mkdir()
    manifest = build_run_manifest(
        dataset,
        ["http://localhost:9222"],
        "model-a",
        "https://api.test/v1",
        100,
        1,
        900,
    )
    atomic_write_json(output / "run_manifest.json", manifest)
    _write_resumable_result(output, item, str(manifest["fingerprint"]), str(manifest["run_id"]))
    options = {"model": "model-a", "max_steps": 100, "force_rerun": False}
    if change == "code":
        monkeypatch.setattr(manifest_module, "_source_hash", lambda: "source-b")
    elif change == "prompt":
        monkeypatch.setattr(manifest_module, "SYSTEM_INSTRUCTIONS", "changed prompt")
    elif change == "model":
        options["model"] = "model-b"
    elif change == "dataset":
        dataset.write_text(json.dumps([{**item, "task": "changed query"}]), encoding="utf-8")
    elif change == "config":
        options["max_steps"] = 99
    elif change == "corrupt_manifest":
        (output / "run_manifest.json").write_text("{broken", encoding="utf-8")
    elif change == "force_rerun":
        options["force_rerun"] = True
    context = ImmediateContext()
    monkeypatch.setattr("web_agent.runner.mp.get_context", lambda method: context)

    summary = run_tasks(
        dataset,
        output,
        ["http://localhost:9222"],
        str(options["model"]),
        "https://api.test/v1",
        "unused",
        max_steps=int(options["max_steps"]),
        force_rerun=bool(options["force_rerun"]),
    )

    assert context.process_count == 1
    assert summary["pending"] == 1
    assert summary["reused"] == 0


def test_interrupted_force_run_does_not_reauthorize_previous_run_result(tmp_path, monkeypatch) -> None:
    import web_agent.run_manifest as manifest_module
    from web_agent.runner import run_tasks

    monkeypatch.setattr(manifest_module, "_source_hash", lambda: "source-a")
    item = {"task_idx": 1, "task_id": "a", "task": "query"}
    dataset = tmp_path / "tasks.json"
    dataset.write_text(json.dumps([item]), encoding="utf-8")
    output = tmp_path / "output"
    output.mkdir()
    original = build_run_manifest(
        dataset,
        ["http://localhost:9222"],
        "model-a",
        "https://api.test/v1",
        100,
        1,
        900,
    )
    atomic_write_json(output / "run_manifest.json", original)
    _write_resumable_result(output, item, str(original["fingerprint"]), str(original["run_id"]))

    forced_context = ImmediateContext()
    monkeypatch.setattr("web_agent.runner.mp.get_context", lambda method: forced_context)
    forced = run_tasks(
        dataset,
        output,
        ["http://localhost:9222"],
        "model-a",
        "https://api.test/v1",
        "unused",
        force_rerun=True,
    )
    assert forced["run_id"] != original["run_id"]
    assert forced_context.process_count == 1

    resumed_context = ImmediateContext()
    monkeypatch.setattr("web_agent.runner.mp.get_context", lambda method: resumed_context)
    resumed = run_tasks(
        dataset,
        output,
        ["http://localhost:9222"],
        "model-a",
        "https://api.test/v1",
        "unused",
    )

    assert resumed["run_id"] == forced["run_id"]
    assert resumed["reused"] == 0
    assert resumed_context.process_count == 1


def test_parent_preserves_current_run_artifact_when_completion_message_is_lost(tmp_path, monkeypatch) -> None:
    import web_agent.run_manifest as manifest_module
    from web_agent.runner import run_tasks

    monkeypatch.setattr(manifest_module, "_source_hash", lambda: "source-a")
    item = {"task_idx": 1, "task_id": "a", "task": "query"}
    dataset = tmp_path / "tasks.json"
    dataset.write_text(json.dumps([item]), encoding="utf-8")
    output = tmp_path / "output"

    def write_artifact(process_kwargs: dict) -> None:
        config, shard, _queue, _rate_limit_state = process_kwargs["args"]
        directory = task_directory(output, shard[0])
        atomic_write_json(directory / "evidence.json", [{"evidence_id": "ev-00001"}])
        atomic_write_json(
            directory / "result.json",
            {
                "task_idx": 1,
                "task_id": "a",
                "status": "SUCCESS",
                "agent_answer": "landed",
                "model_usage": {},
                "duration_seconds": 1.0,
                "run_fingerprint": config["run_fingerprint"],
                "run_id": config["run_id"],
            },
        )

    context = ImmediateContext(on_start=write_artifact)
    monkeypatch.setattr("web_agent.runner.mp.get_context", lambda method: context)

    summary = run_tasks(
        dataset,
        output,
        ["http://localhost:9222"],
        "model-a",
        "https://api.test/v1",
        "unused",
    )

    assert summary["success"] == 1
    assert json.loads((output / "1_a" / "result.json").read_text(encoding="utf-8"))["status"] == "SUCCESS"
    assert json.loads((output / "1_a" / "evidence.json").read_text(encoding="utf-8")) == [
        {"evidence_id": "ev-00001"}
    ]


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


def test_worker_watchdog_terminates_hung_process_and_writes_failures(tmp_path, monkeypatch) -> None:
    from queue import Empty

    from web_agent.runner import run_tasks

    class FakeQueue:
        def __init__(self) -> None:
            self.closed = False

        def get(self, timeout=None):  # noqa: ANN001, ARG002
            raise Empty

        def get_nowait(self):
            raise Empty

        def close(self) -> None:
            self.closed = True

        def join_thread(self) -> None:
            pass

    class HungProcess:
        exitcode = None

        def __init__(self) -> None:
            self.alive = False
            self.terminated = False
            self.joined = False

        def start(self) -> None:
            self.alive = True

        def is_alive(self) -> bool:
            return self.alive

        def terminate(self) -> None:
            self.terminated = True
            self.alive = False
            self.exitcode = -15

        def join(self) -> None:
            self.joined = True

    queue = FakeQueue()
    process = HungProcess()

    class FakeContext:
        def Queue(self) -> FakeQueue:  # noqa: N802
            return queue

        def Process(self, **kwargs) -> HungProcess:  # noqa: N802, ARG002
            return process

    monkeypatch.setattr("web_agent.runner.mp.get_context", lambda method: FakeContext())
    input_path = tmp_path / "tasks.json"
    input_path.write_text(
        json.dumps(
            [
                {"task_idx": 1, "task_id": "a", "task": "one"},
                {"task_idx": 2, "task_id": "b", "task": "two"},
            ]
        ),
        encoding="utf-8",
    )
    output = tmp_path / "out"
    summary = run_tasks(
        input_path,
        output,
        ["http://localhost:9222"],
        "test-model",
        "http://model",
        "key",
        worker_watchdog_seconds=0.01,
    )
    assert process.terminated is True and process.joined is True
    assert summary["completed"] == 2
    assert summary["failed"] == 2
    assert summary["workers"][0]["watchdog_triggered"] is True
    for directory in (output / "1_a", output / "2_b"):
        result = json.loads((directory / "result.json").read_text(encoding="utf-8"))
        assert result["status"] == "FAIL_WORKER_WATCHDOG"
        assert result["run_fingerprint"] == summary["run_fingerprint"]
        assert json.loads((directory / "evidence.json").read_text(encoding="utf-8")) == []


def test_worker_watchdog_returns_when_process_refuses_terminate_and_kill(tmp_path, monkeypatch) -> None:
    from web_agent.runner import run_tasks

    class StubbornProcess:
        exitcode = None

        def start(self) -> None:
            pass

        def is_alive(self) -> bool:
            return True

        def terminate(self) -> None:
            pass

        def kill(self) -> None:
            pass

        def join(self, timeout=None) -> None:  # noqa: ANN001, ARG002
            pass

    queue = ImmediateQueue()
    process = StubbornProcess()

    class StubbornContext:
        def Queue(self) -> ImmediateQueue:  # noqa: N802
            return queue

        def Process(self, **kwargs) -> StubbornProcess:  # noqa: N802, ANN003
            return process

    monkeypatch.setattr("web_agent.runner.mp.get_context", lambda method: StubbornContext())
    input_path = tmp_path / "tasks.json"
    input_path.write_text(
        json.dumps([{"task_idx": 1, "task_id": "a", "task": "one"}]),
        encoding="utf-8",
    )

    summary = run_tasks(
        input_path,
        tmp_path / "out",
        ["http://localhost:9222"],
        "test-model",
        "http://model",
        "key",
        worker_watchdog_seconds=0.01,
    )

    assert summary["failed"] == 1
    assert summary["workers"][0]["watchdog_triggered"] is True
    assert summary["workers"][0]["kill_failed"] is True
    assert "WORKER_KILL_FAILED" in summary["workers"][0]["error"]
    task_dir = tmp_path / "out" / "1_a"
    failure_path = task_dir / "watchdog_failure.json"
    failure = json.loads(failure_path.read_text(encoding="utf-8"))
    assert failure["status"] == "FAIL_WORKER_WATCHDOG"
    assert not (task_dir / "result.json").exists()

    atomic_write_json(
        task_dir / "result.json",
        {
            "status": "SUCCESS",
            "agent_answer": "late",
            "run_fingerprint": summary["run_fingerprint"],
            "run_id": summary["run_id"],
        },
    )
    assert json.loads(failure_path.read_text(encoding="utf-8"))["status"] == "FAIL_WORKER_WATCHDOG"
    assert is_completed(
        task_dir,
        summary["run_fingerprint"],
        summary["run_id"],
        manifest_valid=True,
    ) is False


@pytest.mark.parametrize("timeout", [float("nan"), float("inf"), float("-inf"), 0.0, -1.0])
def test_run_tasks_rejects_non_finite_or_non_positive_watchdog(tmp_path, timeout: float) -> None:
    from web_agent.runner import run_tasks

    input_path = tmp_path / "tasks.json"
    input_path.write_text(
        json.dumps([{"task_idx": 1, "task_id": "a", "task": "one"}]),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="有限正数"):
        run_tasks(
            input_path,
            tmp_path / "out",
            ["http://localhost:9222"],
            "test-model",
            "http://model",
            "key",
            worker_watchdog_seconds=timeout,
        )
