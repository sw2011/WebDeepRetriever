from __future__ import annotations

import json

import pytest

from web_agent.runner import atomic_write_json, is_completed, task_directory


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
