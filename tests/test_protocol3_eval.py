from __future__ import annotations

import json
from pathlib import Path

import pytest

from web_agent.protocol3_eval import (
    compare_results,
    load_protocol3,
    normalize_answer,
    profile_dataset,
    stratified_sample,
    write_comparison_csv,
)


def _tasks() -> list[dict[str, object]]:
    return [
        {
            "task_idx": 0,
            "task_id": "lookup",
            "website": "https://example.test/a",
            "task": "查询产品名称",
            "answer": "Alpha",
        },
        {
            "task_idx": 1,
            "task_id": "analytical",
            "website": "https://example.test/b",
            "task": "排名第二的产品是什么",
            "answer": "Beta",
        },
        {
            "task_idx": 2,
            "task_id": "exhaustive",
            "website": "https://other.test/list",
            "task": "完整列出所有产品",
            "answer": "Alpha\nBeta",
        },
        {
            "task_idx": 3,
            "task_id": "structured",
            "website": "https://other.test/object",
            "task": "查询产品详情",
            "answer": {"name": "Gamma", "count": 3},
        },
        {
            "task_idx": 4,
            "task_id": "lookup-2",
            "website": "https://third.test/",
            "task": "查询发布日期",
            "answer": "2026-01-01",
        },
        {
            "task_idx": 5,
            "task_id": "analytical-2",
            "website": "https://third.test/rank",
            "task": "找出增长最快的年份",
            "answer": "2025年",
        },
    ]


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")


def test_load_and_profile_protocol3(tmp_path: Path) -> None:
    source = tmp_path / "protocol3.json"
    _write_json(source, _tasks())

    items = load_protocol3(source)
    profile = profile_dataset(items)

    assert profile["task_count"] == 6
    assert profile["unique_websites"] == 6
    assert profile["unique_hosts"] == 3
    assert profile["field_profile"]["answer"]["types"] == {"object": 1, "string": 5}
    assert profile["task_type_counts"] == {"analytical": 2, "exhaustive": 1, "lookup": 3}
    assert profile["answer_shape_counts"] == {"multiline": 1, "scalar": 4, "structured": 1}
    assert len(profile["dataset_sha256"]) == 64
    assert "Alpha" not in json.dumps(profile, ensure_ascii=False)


def test_load_rejects_missing_and_duplicate_identifiers(tmp_path: Path) -> None:
    missing = tmp_path / "missing.json"
    _write_json(missing, [{"task_idx": 0}])
    with pytest.raises(ValueError, match="缺少字段"):
        load_protocol3(missing)

    duplicate = tmp_path / "duplicate.json"
    tasks = _tasks()
    tasks[1]["task_id"] = tasks[0]["task_id"]
    _write_json(duplicate, tasks)
    with pytest.raises(ValueError, match="task_id 重复"):
        load_protocol3(duplicate)


def test_stratified_sample_is_deterministic_and_preserves_source_order() -> None:
    tasks = _tasks()
    first, first_report = stratified_sample(tasks, size=4, seed=42)
    second, second_report = stratified_sample(tasks, size=4, seed=42)

    assert first == second
    assert first_report == second_report
    assert len(first) == 4
    assert [item["task_idx"] for item in first] == sorted(item["task_idx"] for item in first)
    assert sum(first_report["allocation"].values()) == 4

    with pytest.raises(ValueError, match="超过"):
        stratified_sample(tasks, size=7)
    with pytest.raises(ValueError, match="大于 0"):
        stratified_sample(tasks, size=0)


def test_compare_results_separates_execution_success_and_scores(tmp_path: Path) -> None:
    tasks = _tasks()[:4]
    results = tmp_path / "results"
    _write_json(
        results / "0_lookup" / "result.json",
        {"task_idx": 0, "task_id": "lookup", "status": "SUCCESS", "agent_answer": "Alpha"},
    )
    _write_json(
        results / "1_analytical" / "result.json",
        {"task_idx": 1, "task_id": "analytical", "status": "SUCCESS", "agent_answer": "  ALPHA　BETA  "},
    )
    tasks[1]["answer"] = "alpha beta"
    _write_json(
        results / "2_exhaustive" / "result.json",
        {"task_idx": 2, "task_id": "exhaustive", "status": "FAIL_BROWSER_ERROR", "error": "CDP disconnected"},
    )
    _write_json(
        results / "orphan" / "result.json",
        {"task_idx": 99, "task_id": "orphan", "status": "SUCCESS", "agent_answer": "unused"},
    )
    _write_json(results / "broken" / "result.json", ["not", "an", "object"])
    reasons = tmp_path / "unrun.json"
    _write_json(reasons, [{"task_id": "structured", "reason": "站点需要未提供的账号"}])

    report = compare_results(tasks, results, unrun_reasons_path=reasons)

    assert report["summary"] == {
        "total_tasks": 4,
        "executed_tasks": 3,
        "not_run_tasks": 1,
        "successful_results": 2,
        "failed_results": 1,
        "comparable_tasks": 2,
        "exact_matches": 1,
        "normalized_matches": 2,
        "exact_accuracy_all": 0.25,
        "normalized_accuracy_all": 0.5,
        "exact_accuracy_comparable": 0.5,
        "normalized_accuracy_comparable": 1.0,
        "not_run_reason_counts": {"站点需要未提供的账号": 1},
    }
    assert report["tasks"][2]["execution"] == "RUN_NO_ANSWER"
    assert report["tasks"][2]["reason"] == "CDP disconnected"
    assert report["tasks"][3]["execution"] == "NOT_RUN"
    assert len(report["orphan_results"]) == 1
    assert len(report["malformed_result_files"]) == 1
    assert "NOT_RUN" in report["score_scope_warning"]

    csv_path = tmp_path / "report.csv"
    write_comparison_csv(csv_path, report["tasks"])
    assert "normalized_match" in csv_path.read_text(encoding="utf-8")


def test_compare_uses_task_idx_fallback_and_rejects_duplicate_results(tmp_path: Path) -> None:
    tasks = _tasks()[:1]
    results = tmp_path / "results"
    _write_json(results / "first" / "result.json", {"task_idx": 0, "status": "SUCCESS", "agent_answer": "Alpha"})
    report = compare_results(tasks, results)
    assert report["summary"]["exact_matches"] == 1

    _write_json(results / "second" / "result.json", {"task_idx": 0, "status": "SUCCESS", "agent_answer": "Alpha"})
    with pytest.raises(ValueError, match="多个 result.json"):
        compare_results(tasks, results)


def test_compare_does_not_fallback_from_wrong_task_id(tmp_path: Path) -> None:
    tasks = _tasks()[:1]
    results = tmp_path / "results"
    _write_json(
        results / "wrong-id" / "result.json",
        {"task_idx": 0, "task_id": "not-the-task", "status": "SUCCESS", "agent_answer": "Alpha"},
    )

    report = compare_results(tasks, results)

    assert report["summary"]["executed_tasks"] == 0
    assert report["tasks"][0]["execution"] == "NOT_RUN"
    assert len(report["orphan_results"]) == 1


def test_normalize_answer_is_conservative() -> None:
    assert normalize_answer(" Ａ\n B  ") == "a b"
    assert normalize_answer(["A", "B"]) == ["a", "b"]
    assert normalize_answer(["A", "B"]) != normalize_answer(["B", "A"])
    assert normalize_answer(1) == 1


def test_compare_only_scores_nonempty_success_answers_and_sanitizes_errors(tmp_path: Path) -> None:
    tasks = _tasks()[:3]
    results = tmp_path / "results"
    _write_json(
        results / "0_lookup" / "result.json",
        {"task_idx": 0, "task_id": "lookup", "status": "FAIL_AGENT_ERROR", "agent_answer": "Alpha"},
    )
    _write_json(
        results / "1_analytical" / "result.json",
        {"task_idx": 1, "task_id": "analytical", "status": "SUCCESS", "agent_answer": ""},
    )
    _write_json(
        results / "2_exhaustive" / "result.json",
        {
            "task_idx": 2,
            "task_id": "exhaustive",
            "status": "FAIL_AGENT_ERROR",
            "error": "RateLimitError 429 TPM org-private ak-private-private",
        },
    )

    report = compare_results(tasks, results)

    assert report["summary"]["successful_results"] == 0
    assert report["summary"]["failed_results"] == 3
    assert report["summary"]["comparable_tasks"] == 0
    assert report["tasks"][0]["agent_answer"] is None
    assert report["tasks"][1]["reason"] == "SUCCESS 结果缺少非空 agent_answer"
    assert report["tasks"][2]["reason"] == "模型 API 429：组织级 TPM 限额"
