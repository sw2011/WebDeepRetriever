from __future__ import annotations

import asyncio
import json
import logging
import multiprocessing as mp
import os
import queue as queue_module
import tempfile
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .browser_actor import BrowserActor
from .contracts import TaskContract
from .evidence import EvidenceStore
from .runtime import ProtocolIIIAgent
from .sanitization import sanitize_exception
from .token_control import SharedTPMLimiter


@dataclass(frozen=True)
class WorkerConfig:
    worker_id: int
    cdp_url: str
    output_dir: str
    model: str
    api_base: str
    api_key: str
    max_steps: int


def atomic_write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=path.parent, prefix=f".{path.name}.", suffix=".tmp", delete=False
    ) as output:
        json.dump(value, output, ensure_ascii=False, indent=2)
        output.flush()
        os.fsync(output.fileno())
        temporary = Path(output.name)
    temporary.replace(path)


def task_directory(output_dir: Path, item: dict[str, Any]) -> Path:
    task_idx = item.get("task_idx", item.get("task_index", ""))
    task_id = str(item.get("task_id", task_idx))
    return output_dir / f"{task_idx}_{task_id}"


def is_completed(path: Path) -> bool:
    result_path = path / "result.json"
    if not result_path.is_file():
        return False
    try:
        result = json.loads(result_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return result.get("status") == "SUCCESS" and bool(result.get("agent_answer"))


def _aggregate_usage(items: list[dict[str, Any]]) -> dict[str, Any]:
    usages = [item.get("model_usage", {}) for item in items]
    reasons: Counter[str] = Counter()
    for usage in usages:
        reasons.update(usage.get("throttle_reasons", {}))
    return {
        "task_count": len(usages),
        "request_count": sum(int(usage.get("request_count", 0)) for usage in usages),
        "usage_available_count": sum(int(usage.get("usage_available_count", 0)) for usage in usages),
        "usage_unavailable_count": sum(int(usage.get("usage_unavailable_count", 0)) for usage in usages),
        "input_tokens": sum(int(usage.get("input_tokens", 0)) for usage in usages),
        "output_tokens": sum(int(usage.get("output_tokens", 0)) for usage in usages),
        "total_tokens": sum(int(usage.get("total_tokens", 0)) for usage in usages),
        "estimated_input_tokens": sum(int(usage.get("estimated_input_tokens", 0)) for usage in usages),
        "throttle_wait_seconds": round(
            sum(float(usage.get("throttle_wait_seconds", 0.0)) for usage in usages), 3
        ),
        "throttle_reasons": dict(sorted(reasons.items())),
    }


def _kimi_tpm_budget(model: str) -> int | None:
    if model.strip().lower().rsplit("/", 1)[-1] != "kimi-k2.6":
        return None
    try:
        limit = int(os.environ.get("MOONSHOT_TPM_LIMIT", "3000000"))
        safety_ratio = float(os.environ.get("MOONSHOT_TPM_SAFETY_RATIO", "0.8"))
    except ValueError as exc:
        raise ValueError("Moonshot TPM 环境变量必须是合法数字") from exc
    if limit < 1 or not 0.1 <= safety_ratio <= 1.0:
        raise ValueError("Moonshot TPM 上限必须大于 0，安全比例必须在 0.1 到 1.0 之间")
    return max(1, int(limit * safety_ratio))


async def _worker_async(
    config: WorkerConfig,
    items: list[dict[str, Any]],
    rate_limit_state: tuple[Any, Any, int, float] | None = None,
) -> list[dict[str, Any]]:
    output_root = Path(config.output_dir)
    placeholder_store = EvidenceStore()
    actor = BrowserActor(config.cdp_url, output_root / f".worker-{config.worker_id}", placeholder_store)
    limiter = (
        SharedTPMLimiter(
            rate_limit_state[0],
            rate_limit_state[1],
            token_budget=rate_limit_state[2],
            window_seconds=rate_limit_state[3],
        )
        if rate_limit_state is not None
        else None
    )
    agent = ProtocolIIIAgent(
        config.model,
        config.api_base,
        config.api_key,
        rate_limiter=limiter,
        worker_id=config.worker_id,
    )
    summaries: list[dict[str, Any]] = []
    try:
        for item in items:
            contract = TaskContract.from_item(item, config.max_steps)
            task_dir = task_directory(output_root, item)
            task_dir.mkdir(parents=True, exist_ok=True)
            evidence_store = EvidenceStore()
            try:
                await actor.begin_task(contract.website, task_dir, evidence_store)
                run_result = await agent.run(actor, contract, evidence_store)
            except Exception as exc:
                run_result = {
                    "status": "FAIL_BROWSER_ERROR",
                    "agent_answer": None,
                    "evidence_ids": [],
                    "evidence_bindings": {},
                    "coverage": None,
                    "actions": [],
                    "thoughts": [],
                    "urls": [],
                    "receipts": [],
                    "predict_length": 0,
                    "model_usage": {
                        "worker_id": config.worker_id,
                        "task_idx": contract.task_idx,
                        "task_id": contract.task_id,
                        "request_count": 0,
                        "usage_available_count": 0,
                        "usage_unavailable_count": 0,
                        "input_tokens": 0,
                        "output_tokens": 0,
                        "total_tokens": 0,
                        "estimated_input_tokens": 0,
                        "throttle_wait_seconds": 0.0,
                        "throttle_reasons": {},
                        "requests": [],
                    },
                    "error": sanitize_exception(exc),
                }
            try:
                await actor.flush_artifacts()
            except Exception as exc:
                run_result["artifact_error"] = sanitize_exception(exc)
            evidence_store.save(task_dir / "evidence.json")
            result = {
                "task_idx": contract.task_idx,
                "task_id": contract.task_id,
                "task": contract.task,
                "website": contract.website,
                **run_result,
                "final_result_response": run_result.get("agent_answer") or "",
                "worker_id": config.worker_id,
            }
            atomic_write_json(task_dir / "result.json", result)
            summaries.append(
                {
                    "task_idx": contract.task_idx,
                    "task_id": contract.task_id,
                    "status": result["status"],
                    "agent_answer": result.get("agent_answer"),
                    "error": result.get("error"),
                    "model_usage": result.get("model_usage", {}),
                }
            )
    finally:
        await actor.close()
    return summaries


def worker_entry(
    config_dict: dict[str, Any],
    items: list[dict[str, Any]],
    queue: Any,
    rate_limit_state: tuple[Any, Any, int, float] | None = None,
) -> None:
    config = WorkerConfig(**config_dict)
    log_dir = Path(config.output_dir) / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format=f"%(asctime)s %(levelname)s [worker {config.worker_id}] %(message)s",
        handlers=[logging.FileHandler(log_dir / f"worker_{config.worker_id}.log", encoding="utf-8")],
    )
    try:
        summaries = asyncio.run(_worker_async(config, items, rate_limit_state))
        queue.put(
            {
                "worker_id": config.worker_id,
                "summaries": summaries,
                "model_usage": _aggregate_usage(summaries),
                "error": None,
            }
        )
    except BaseException as exc:
        queue.put(
            {
                "worker_id": config.worker_id,
                "summaries": [],
                "model_usage": _aggregate_usage([]),
                "error": sanitize_exception(exc),
            }
        )


def run_tasks(
    input_path: Path,
    output_dir: Path,
    cdp_urls: list[str],
    model: str,
    api_base: str,
    api_key: str,
    max_steps: int = 100,
) -> dict[str, Any]:
    if not 1 <= len(cdp_urls) <= 8:
        raise ValueError("Protocol III 要求 CDP worker 数为 1 到 8")
    if len(set(cdp_urls)) != len(cdp_urls):
        raise ValueError("每个 worker 必须使用独立的 CDP URL，不能复用同一浏览器")
    items = json.loads(Path(input_path).read_text(encoding="utf-8"))
    if not isinstance(items, list):
        raise ValueError("输入必须是 JSON 数组")
    if any(not isinstance(item, dict) for item in items):
        raise ValueError("输入数组的每一项都必须是 JSON 对象")
    task_keys = [task_directory(Path("."), item).name for item in items]
    if len(set(task_keys)) != len(task_keys):
        raise ValueError("输入中存在重复的 task_idx/task_id，无法保证 worker 输出隔离")
    output_dir.mkdir(parents=True, exist_ok=True)
    pending = [item for item in items if not is_completed(task_directory(output_dir, item))]
    if not pending:
        summary = {
            "total": len(items),
            "pending": 0,
            "completed": 0,
            "model_usage": _aggregate_usage([]),
            "workers": [],
        }
        atomic_write_json(output_dir / "logs" / "summary.json", summary)
        return summary

    worker_count = min(len(cdp_urls), len(pending), 8)
    shards = [pending[index::worker_count] for index in range(worker_count)]
    context = mp.get_context("spawn")
    tpm_budget = _kimi_tpm_budget(model)
    queue: Any | None = None
    manager: Any | None = None
    processes: list[mp.Process] = []
    worker_results: list[dict[str, Any]] = []
    try:
        queue = context.Queue()
        manager = context.Manager() if tpm_budget is not None else None
        rate_limit_state = (
            (manager.list(), manager.RLock(), tpm_budget, 60.0)
            if manager is not None and tpm_budget is not None
            else None
        )
        for worker_id, shard in enumerate(shards):
            config = WorkerConfig(
                worker_id=worker_id,
                cdp_url=cdp_urls[worker_id],
                output_dir=str(output_dir),
                model=model,
                api_base=api_base,
                api_key=api_key,
                max_steps=min(max(max_steps, 1), 100),
            )
            process = context.Process(
                target=worker_entry,
                args=(asdict(config), shard, queue, rate_limit_state),
            )
            process.start()
            processes.append(process)

        while any(process.is_alive() for process in processes):
            try:
                worker_results.append(queue.get(timeout=0.25))
            except queue_module.Empty:
                pass
        for process in processes:
            process.join()
        while len(worker_results) < len(processes):
            try:
                worker_results.append(queue.get(timeout=0.25))
            except queue_module.Empty:
                break
        reported_workers = {item["worker_id"] for item in worker_results}
        for worker_id, process in enumerate(processes):
            if worker_id not in reported_workers:
                worker_results.append(
                    {
                        "worker_id": worker_id,
                        "summaries": [],
                        "model_usage": _aggregate_usage([]),
                        "error": f"worker 未返回结果，进程退出码 {process.exitcode}",
                    }
                )
    except BaseException:
        for process in processes:
            try:
                if process.is_alive():
                    process.terminate()
            except Exception:
                pass
        for process in processes:
            try:
                process.join()
            except Exception:
                pass
        raise
    finally:
        if queue is not None:
            try:
                queue.close()
                queue.join_thread()
            except Exception:
                pass
        if manager is not None:
            try:
                manager.shutdown()
            except Exception:
                pass
    summaries = [entry for worker in worker_results for entry in worker["summaries"]]
    summary = {
        "total": len(items),
        "pending": len(pending),
        "completed": len(summaries),
        "success": sum(item["status"] == "SUCCESS" for item in summaries),
        "failed": sum(item["status"] != "SUCCESS" for item in summaries),
        "tpm_safety_budget": tpm_budget,
        "model_usage": _aggregate_usage(summaries),
        "workers": worker_results,
    }
    atomic_write_json(output_dir / "logs" / "summary.json", summary)
    return summary
