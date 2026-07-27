from __future__ import annotations

import asyncio
import json
import logging
import multiprocessing as mp
import os
import queue as queue_module
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .browser_actor import BrowserActor
from .contracts import TaskContract
from .evidence import EvidenceStore
from .runtime import ProtocolIIIAgent
from .sanitization import sanitize_exception


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


async def _worker_async(config: WorkerConfig, items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output_root = Path(config.output_dir)
    placeholder_store = EvidenceStore()
    actor = BrowserActor(config.cdp_url, output_root / f".worker-{config.worker_id}", placeholder_store)
    agent = ProtocolIIIAgent(config.model, config.api_base, config.api_key)
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
                }
            )
    finally:
        await actor.close()
    return summaries


def worker_entry(config_dict: dict[str, Any], items: list[dict[str, Any]], queue: Any) -> None:
    config = WorkerConfig(**config_dict)
    log_dir = Path(config.output_dir) / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format=f"%(asctime)s %(levelname)s [worker {config.worker_id}] %(message)s",
        handlers=[logging.FileHandler(log_dir / f"worker_{config.worker_id}.log", encoding="utf-8")],
    )
    try:
        summaries = asyncio.run(_worker_async(config, items))
        queue.put({"worker_id": config.worker_id, "summaries": summaries, "error": None})
    except BaseException as exc:
        queue.put({"worker_id": config.worker_id, "summaries": [], "error": sanitize_exception(exc)})


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
        summary = {"total": len(items), "pending": 0, "completed": 0, "workers": []}
        atomic_write_json(output_dir / "logs" / "summary.json", summary)
        return summary

    worker_count = min(len(cdp_urls), len(pending), 8)
    shards = [pending[index::worker_count] for index in range(worker_count)]
    context = mp.get_context("spawn")
    queue = context.Queue()
    processes: list[mp.Process] = []
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
        process = context.Process(target=worker_entry, args=(asdict(config), shard, queue))
        process.start()
        processes.append(process)

    for process in processes:
        process.join()
    worker_results: list[dict[str, Any]] = []
    for _ in processes:
        try:
            worker_results.append(queue.get(timeout=1))
        except queue_module.Empty:
            break
    reported_workers = {item["worker_id"] for item in worker_results}
    for worker_id, process in enumerate(processes):
        if worker_id not in reported_workers:
            worker_results.append(
                {
                    "worker_id": worker_id,
                    "summaries": [],
                    "error": f"worker 未返回结果，进程退出码 {process.exitcode}",
                }
            )
    summaries = [entry for worker in worker_results for entry in worker["summaries"]]
    summary = {
        "total": len(items),
        "pending": len(pending),
        "completed": len(summaries),
        "success": sum(item["status"] == "SUCCESS" for item in summaries),
        "failed": sum(item["status"] != "SUCCESS" for item in summaries),
        "workers": worker_results,
    }
    atomic_write_json(output_dir / "logs" / "summary.json", summary)
    return summary
