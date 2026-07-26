"""Protocol III 数据剖析、确定性分层抽样和离线答案对比。"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import statistics
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence
from urllib.parse import urlparse


REQUIRED_FIELDS = ("task_idx", "task_id", "website", "task", "answer")
_EXHAUSTIVE_PATTERN = re.compile(
    r"(?:哪些|所有|全部|完整|列举|列出|分别|各(?:个|项)|每(?:个|项)|前\s*\d+|top\s*\d+)",
    re.IGNORECASE,
)
_ANALYTICAL_PATTERN = re.compile(
    r"(?:最高|最低|最多|最少|排名|第[一二三四五六七八九十\d]+|增速|增长|变化|比较|占比|总数|多少|合计|平均|同比|环比)",
    re.IGNORECASE,
)


def _json_type(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "boolean"
    if isinstance(value, str):
        return "string"
    if isinstance(value, (int, float)):
        return "number"
    if isinstance(value, list):
        return "array"
    if isinstance(value, dict):
        return "object"
    return type(value).__name__


def _stable_rank(seed: int, value: str) -> str:
    return hashlib.sha256(f"{seed}\0{value}".encode("utf-8")).hexdigest()


def _dataset_digest(items: Sequence[Mapping[str, Any]]) -> str:
    encoded = json.dumps(items, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def load_protocol3(path: str | Path) -> list[dict[str, Any]]:
    """读取并严格校验 Protocol III JSON 数组。"""

    source = Path(path)
    try:
        value = json.loads(source.read_text(encoding="utf-8"))
    except OSError as exc:
        raise ValueError(f"无法读取数据集 {source}: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"数据集不是合法 JSON: {exc}") from exc
    if not isinstance(value, list):
        raise ValueError("Protocol III 数据集根节点必须是 JSON 数组")

    items: list[dict[str, Any]] = []
    task_ids: set[str] = set()
    task_indexes: set[str] = set()
    for position, raw_item in enumerate(value):
        if not isinstance(raw_item, dict):
            raise ValueError(f"第 {position} 条任务必须是 JSON 对象")
        missing = [field for field in REQUIRED_FIELDS if field not in raw_item]
        if missing:
            raise ValueError(f"第 {position} 条任务缺少字段: {missing}")
        task_id = str(raw_item["task_id"]).strip()
        task_idx = str(raw_item["task_idx"])
        if not task_id:
            raise ValueError(f"第 {position} 条任务的 task_id 为空")
        if task_id in task_ids:
            raise ValueError(f"task_id 重复: {task_id}")
        if task_idx in task_indexes:
            raise ValueError(f"task_idx 重复: {task_idx}")
        task_ids.add(task_id)
        task_indexes.add(task_idx)
        items.append(dict(raw_item))
    return items


def classify_task(item: Mapping[str, Any]) -> dict[str, str]:
    task = str(item.get("task", ""))
    answer = item.get("answer")
    if _EXHAUSTIVE_PATTERN.search(task):
        task_type = "exhaustive"
    elif _ANALYTICAL_PATTERN.search(task):
        task_type = "analytical"
    else:
        task_type = "lookup"

    if isinstance(answer, (list, dict)):
        answer_shape = "structured"
    elif isinstance(answer, str) and "\n" in answer.strip():
        answer_shape = "multiline"
    elif isinstance(answer, str) and len(answer.strip()) >= 120:
        answer_shape = "long_text"
    else:
        answer_shape = "scalar"
    return {
        "task_type": task_type,
        "answer_shape": answer_shape,
        "stratum": f"{task_type}:{answer_shape}",
    }


def profile_dataset(items: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """生成不包含答案正文的数据字段与分层统计。"""

    fields = sorted(set(REQUIRED_FIELDS).union(*(item.keys() for item in items))) if items else list(REQUIRED_FIELDS)
    field_profile: dict[str, Any] = {}
    for field in fields:
        present = [item[field] for item in items if field in item]
        field_profile[field] = {
            "present": len(present),
            "missing": len(items) - len(present),
            "null": sum(value is None for value in present),
            "empty_string": sum(isinstance(value, str) and not value.strip() for value in present),
            "types": dict(sorted(Counter(_json_type(value) for value in present).items())),
        }

    classifications = [classify_task(item) for item in items]
    websites = [str(item.get("website", "")) for item in items]
    hosts = [urlparse(website).hostname or "" for website in websites]
    answer_lengths = [
        len(value) if isinstance(value, str) else len(json.dumps(value, ensure_ascii=False, sort_keys=True))
        for value in (item.get("answer") for item in items)
    ]
    length_stats = {
        "min": min(answer_lengths, default=0),
        "max": max(answer_lengths, default=0),
        "mean": round(statistics.fmean(answer_lengths), 3) if answer_lengths else 0,
        "median": statistics.median(answer_lengths) if answer_lengths else 0,
    }
    return {
        "schema": "webretriever.protocol3.profile/v1",
        "task_count": len(items),
        "dataset_sha256": _dataset_digest(items),
        "required_fields": list(REQUIRED_FIELDS),
        "field_profile": field_profile,
        "unique_websites": len(set(websites)),
        "unique_hosts": len(set(hosts)),
        "website_counts": dict(sorted(Counter(websites).items())),
        "task_type_counts": dict(sorted(Counter(value["task_type"] for value in classifications).items())),
        "answer_shape_counts": dict(sorted(Counter(value["answer_shape"] for value in classifications).items())),
        "stratum_counts": dict(sorted(Counter(value["stratum"] for value in classifications).items())),
        "answer_length": length_stats,
    }


def _allocate_strata(groups: Mapping[str, Sequence[Mapping[str, Any]]], size: int, seed: int) -> dict[str, int]:
    names = sorted(groups)
    if size < len(names):
        chosen = sorted(names, key=lambda name: (-len(groups[name]), _stable_rank(seed, name)))[:size]
        return {name: int(name in chosen) for name in names}

    total = sum(len(group) for group in groups.values())
    quotas = {name: size * len(groups[name]) / total for name in names}
    allocation = {name: 1 for name in names}
    remaining = size - len(names)
    while remaining:
        candidates = [name for name in names if allocation[name] < len(groups[name])]
        if not candidates:
            break
        selected = min(
            candidates,
            key=lambda name: (allocation[name] - quotas[name], _stable_rank(seed, name)),
        )
        allocation[selected] += 1
        remaining -= 1
    return allocation


def stratified_sample(
    items: Sequence[Mapping[str, Any]], size: int, seed: int = 20260726
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """按任务类型和答案形态分层，使用哈希排序产生可复现样本。"""

    if size < 1:
        raise ValueError("样本数必须大于 0")
    if size > len(items):
        raise ValueError(f"样本数 {size} 超过数据集任务数 {len(items)}")
    groups: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    positions: dict[tuple[str, str], int] = {}
    for position, item in enumerate(items):
        stratum = classify_task(item)["stratum"]
        groups[stratum].append(item)
        positions[(str(item["task_id"]), str(item["task_idx"]))] = position

    allocation = _allocate_strata(groups, size, seed)
    selected: list[Mapping[str, Any]] = []
    for name, group in groups.items():
        ranked = sorted(
            group,
            key=lambda item: _stable_rank(seed, f"{name}\0{item['task_id']}\0{item['task_idx']}"),
        )
        selected.extend(ranked[: allocation[name]])
    selected.sort(key=lambda item: positions[(str(item["task_id"]), str(item["task_idx"]))])
    sample = [dict(item) for item in selected]
    report = {
        "schema": "webretriever.protocol3.sample/v1",
        "source_count": len(items),
        "sample_count": len(sample),
        "seed": seed,
        "source_sha256": _dataset_digest(items),
        "sample_sha256": _dataset_digest(sample),
        "allocation": dict(sorted(allocation.items())),
        "task_ids": [str(item["task_id"]) for item in sample],
    }
    return sample, report


def normalize_answer(value: Any) -> Any:
    """保守规范化：Unicode NFKC、大小写和空白，不改变列表顺序或数值类型。"""

    if isinstance(value, str):
        return re.sub(r"\s+", " ", unicodedata.normalize("NFKC", value)).strip().casefold()
    if isinstance(value, list):
        return [normalize_answer(item) for item in value]
    if isinstance(value, dict):
        return {str(key): normalize_answer(value[key]) for key in sorted(value, key=str)}
    return value


def _strict_equal(actual: Any, expected: Any) -> bool:
    return type(actual) is type(expected) and actual == expected


def _load_unrun_reasons(path: str | Path | None) -> dict[str, str]:
    if path is None:
        return {}
    source = Path(path)
    try:
        value = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"无法读取未运行原因文件 {source}: {exc}") from exc
    if isinstance(value, dict) and "reasons" in value:
        value = value["reasons"]
    reasons: dict[str, str] = {}
    if isinstance(value, dict):
        for key, reason in value.items():
            reasons[str(key)] = str(reason).strip()
        return reasons
    if not isinstance(value, list):
        raise ValueError("未运行原因必须是键值对象，或包含 task_id/task_idx/reason 的数组")
    for position, item in enumerate(value):
        if not isinstance(item, dict) or "reason" not in item:
            raise ValueError(f"未运行原因第 {position} 项格式错误")
        identifiers = [item.get("task_id"), item.get("task_idx")]
        if not any(identifier is not None for identifier in identifiers):
            raise ValueError(f"未运行原因第 {position} 项缺少 task_id 或 task_idx")
        for identifier in identifiers:
            if identifier is not None:
                reasons[str(identifier)] = str(item["reason"]).strip()
    return reasons


def _discover_results(results_dir: str | Path) -> tuple[list[dict[str, Any]], list[dict[str, str]]]:
    root = Path(results_dir)
    candidates: list[dict[str, Any]] = []
    malformed: list[dict[str, str]] = []
    for path in sorted(root.rglob("result.json")) if root.is_dir() else []:
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(value, dict):
                raise ValueError("根节点不是 JSON 对象")
        except (OSError, json.JSONDecodeError, ValueError) as exc:
            malformed.append({"path": str(path), "error": str(exc)})
            continue
        candidates.append({"path": path, "value": value})
    return candidates, malformed


def compare_results(
    items: Sequence[Mapping[str, Any]],
    results_dir: str | Path,
    *,
    unrun_reasons_path: str | Path | None = None,
    default_not_run_reason: str = "未找到对应的 result.json",
) -> dict[str, Any]:
    """递归读取 result.json，并按 task_id（后备 task_idx）生成离线对比报告。"""

    candidates, malformed = _discover_results(results_dir)
    by_id: dict[str, dict[str, Any]] = {}
    by_idx: dict[str, dict[str, Any]] = {}
    for candidate in candidates:
        value = candidate["value"]
        task_id = value.get("task_id")
        if task_id is not None:
            identifier = str(task_id)
            if identifier in by_id and by_id[identifier]["path"] != candidate["path"]:
                raise ValueError(f"多个 result.json 使用相同 task_id={identifier}")
            by_id[identifier] = candidate
        elif value.get("task_idx") is not None:
            # 只有旧结果完全没有 task_id 时才后备 task_idx，避免错误 task_id 被误计分。
            identifier = str(value["task_idx"])
            if identifier in by_idx and by_idx[identifier]["path"] != candidate["path"]:
                raise ValueError(f"多个 result.json 使用相同 task_idx={identifier}")
            by_idx[identifier] = candidate

    unrun_reasons = _load_unrun_reasons(unrun_reasons_path)
    rows: list[dict[str, Any]] = []
    used_paths: set[Path] = set()
    for item in items:
        task_id = str(item["task_id"])
        task_idx = str(item["task_idx"])
        candidate = by_id.get(task_id) or by_idx.get(task_idx)
        if candidate is None:
            reason = unrun_reasons.get(task_id) or unrun_reasons.get(task_idx) or default_not_run_reason
            rows.append(
                {
                    "task_idx": item["task_idx"],
                    "task_id": task_id,
                    "website": item["website"],
                    "execution": "NOT_RUN",
                    "result_status": None,
                    "reason": reason,
                    "agent_answer": None,
                    "expected_answer": item["answer"],
                    "exact_match": None,
                    "normalized_match": None,
                    "result_path": None,
                }
            )
            continue

        used_paths.add(candidate["path"])
        result = candidate["value"]
        has_answer = "agent_answer" in result and result["agent_answer"] is not None
        actual = result.get("agent_answer")
        expected = item["answer"]
        result_status = str(result.get("status", "UNKNOWN"))
        reason = result.get("error")
        if result_status != "SUCCESS" and not reason:
            reason = f"任务已运行但状态为 {result_status}"
        rows.append(
            {
                "task_idx": item["task_idx"],
                "task_id": task_id,
                "website": item["website"],
                "execution": "RUN_WITH_ANSWER" if has_answer else "RUN_NO_ANSWER",
                "result_status": result_status,
                "reason": str(reason) if reason else None,
                "agent_answer": actual,
                "expected_answer": expected,
                "exact_match": _strict_equal(actual, expected) if has_answer else None,
                "normalized_match": _strict_equal(normalize_answer(actual), normalize_answer(expected))
                if has_answer
                else None,
                "result_path": str(candidate["path"]),
            }
        )

    comparable = [row for row in rows if row["execution"] == "RUN_WITH_ANSWER"]
    exact_matches = sum(row["exact_match"] is True for row in comparable)
    normalized_matches = sum(row["normalized_match"] is True for row in comparable)
    total = len(rows)
    executed = sum(row["execution"] != "NOT_RUN" for row in rows)

    def rate(numerator: int, denominator: int) -> float | None:
        return round(numerator / denominator, 6) if denominator else None

    orphan_results = [str(candidate["path"]) for candidate in candidates if candidate["path"] not in used_paths]
    summary = {
        "total_tasks": total,
        "executed_tasks": executed,
        "not_run_tasks": total - executed,
        "successful_results": sum(row["result_status"] == "SUCCESS" for row in rows),
        "failed_results": sum(
            row["execution"] != "NOT_RUN" and row["result_status"] != "SUCCESS" for row in rows
        ),
        "comparable_tasks": len(comparable),
        "exact_matches": exact_matches,
        "normalized_matches": normalized_matches,
        "exact_accuracy_all": rate(exact_matches, total),
        "normalized_accuracy_all": rate(normalized_matches, total),
        "exact_accuracy_comparable": rate(exact_matches, len(comparable)),
        "normalized_accuracy_comparable": rate(normalized_matches, len(comparable)),
        "not_run_reason_counts": dict(
            sorted(Counter(row["reason"] for row in rows if row["execution"] == "NOT_RUN").items())
        ),
    }
    return {
        "schema": "webretriever.protocol3.comparison/v1",
        "dataset_sha256": _dataset_digest(items),
        "results_dir": str(Path(results_dir)),
        "summary": summary,
        "tasks": rows,
        "orphan_results": orphan_results,
        "malformed_result_files": malformed,
        "score_scope_warning": "报告仅统计提供目录中实际存在的结果；NOT_RUN 不代表网站任务已执行。",
    }


def write_json(path: str | Path, value: Any) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_comparison_csv(path: str | Path, rows: Iterable[Mapping[str, Any]]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    fields = (
        "task_idx",
        "task_id",
        "website",
        "execution",
        "result_status",
        "reason",
        "agent_answer",
        "expected_answer",
        "exact_match",
        "normalized_match",
        "result_path",
    )
    with destination.open("w", encoding="utf-8", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            serialized = dict(row)
            for field in ("agent_answer", "expected_answer"):
                if not isinstance(serialized.get(field), (str, int, float, bool, type(None))):
                    serialized[field] = json.dumps(serialized[field], ensure_ascii=False, sort_keys=True)
            writer.writerow({field: serialized.get(field) for field in fields})


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="WebRetriever Protocol III 离线评测工具")
    subparsers = parser.add_subparsers(dest="command", required=True)

    profile = subparsers.add_parser("profile", help="剖析字段、网站和任务分层")
    profile.add_argument("--input", required=True, help="protocol3.json 路径")
    profile.add_argument("--output", help="JSON 报告路径；省略时输出到 stdout")

    sample = subparsers.add_parser("sample", help="生成确定性分层样本")
    sample.add_argument("--input", required=True, help="protocol3.json 路径")
    sample.add_argument("--output", required=True, help="样本 JSON 路径")
    sample.add_argument("--size", required=True, type=int, help="样本任务数")
    sample.add_argument("--seed", type=int, default=20260726, help="确定性抽样种子")
    sample.add_argument("--report", help="抽样元数据 JSON 路径")

    compare = subparsers.add_parser("compare", help="对比 result.json 中的 agent_answer")
    compare.add_argument("--input", required=True, help="包含标准 answer 的 protocol3.json")
    compare.add_argument("--results", required=True, help="递归搜索 result.json 的输出目录")
    compare.add_argument("--output", required=True, help="JSON 对比报告路径")
    compare.add_argument("--csv", help="可选逐任务 CSV 报告路径")
    compare.add_argument("--unrun-reasons", help="可选的未运行原因 JSON")
    compare.add_argument("--default-not-run-reason", default="未找到对应的 result.json")
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    items = load_protocol3(args.input)
    if args.command == "profile":
        report = profile_dataset(items)
        if args.output:
            write_json(args.output, report)
            print(json.dumps({"output": args.output, "task_count": report["task_count"]}, ensure_ascii=False))
        else:
            print(json.dumps(report, ensure_ascii=False, indent=2))
        return
    if args.command == "sample":
        sample, report = stratified_sample(items, args.size, args.seed)
        write_json(args.output, sample)
        if args.report:
            write_json(args.report, report)
        print(json.dumps({"output": args.output, **report}, ensure_ascii=False, indent=2))
        return
    if args.command == "compare":
        report = compare_results(
            items,
            args.results,
            unrun_reasons_path=args.unrun_reasons,
            default_not_run_reason=args.default_not_run_reason,
        )
        write_json(args.output, report)
        if args.csv:
            write_comparison_csv(args.csv, report["tasks"])
        print(json.dumps({"output": args.output, "summary": report["summary"]}, ensure_ascii=False, indent=2))
        return
    raise AssertionError(f"未知命令: {args.command}")


if __name__ == "__main__":
    main()
