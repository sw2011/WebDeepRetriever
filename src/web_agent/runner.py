from __future__ import annotations

import asyncio
import json
import logging
import math
import multiprocessing as mp
import os
import queue as queue_module
import tempfile
import time
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .browser_actor import BrowserActor
from .contracts import TaskContract
from .evidence import EvidenceStore
from .run_manifest import MANIFEST_SCHEMA_VERSION, build_run_manifest, load_manifest, manifest_matches
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
    run_fingerprint: str
    run_id: str


def atomic_write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=path.parent, prefix=f".{path.name}.", suffix=".tmp", delete=False
    ) as output:
        output.write(encoded)
        output.flush()
        os.fsync(output.fileno())
        temporary = Path(output.name)
    temporary.replace(path)


def task_directory(output_dir: Path, item: dict[str, Any]) -> Path:
    task_idx = item.get("task_idx", item.get("task_index", ""))
    task_id = str(item.get("task_id", task_idx))
    return output_dir / f"{task_idx}_{task_id}"


def is_completed(
    path: Path,
    run_fingerprint: str | None = None,
    run_id: str | None = None,
    *,
    manifest_valid: bool = False,
) -> bool:
    if not run_fingerprint or not run_id or not manifest_valid:
        return False
    if _has_watchdog_failure(path, run_fingerprint, run_id):
        return False
    result_path = path / "result.json"
    if not result_path.is_file():
        return False
    try:
        result = json.loads(result_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    if not isinstance(result, dict):
        return False
    answer = result.get("agent_answer")
    if isinstance(answer, str):
        has_answer = bool(answer.strip())
    elif isinstance(answer, (list, dict, tuple, set)):
        has_answer = bool(answer)
    else:
        has_answer = answer is not None
    return (
        result.get("status") == "SUCCESS"
        and has_answer
        and result.get("run_fingerprint") == run_fingerprint
        and result.get("run_id") == run_id
    )


def _has_watchdog_failure(path: Path, run_fingerprint: str, run_id: str) -> bool:
    try:
        failure = json.loads((path / "watchdog_failure.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return bool(
        isinstance(failure, dict)
        and failure.get("run_fingerprint") == run_fingerprint
        and failure.get("run_id") == run_id
    )


def _distribution(values: list[float | int]) -> dict[str, float | int | None]:
    if not values:
        return {"p50": None, "p95": None, "max": None}
    ordered = sorted(values)

    def percentile(fraction: float) -> float | int:
        return ordered[max(0, min(len(ordered) - 1, math.ceil(len(ordered) * fraction) - 1))]

    return {"p50": percentile(0.50), "p95": percentile(0.95), "max": ordered[-1]}


def _aggregate_usage(items: list[dict[str, Any]]) -> dict[str, Any]:
    usages = [item.get("model_usage", {}) for item in items]
    reasons: Counter[str] = Counter()
    for usage in usages:
        reasons.update(usage.get("throttle_reasons", {}))
    requests = [request for usage in usages for request in usage.get("requests", [])]
    available_inputs = [int(request["input_tokens"]) for request in requests if request.get("input_tokens") is not None]
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
        "request_count_per_task": _distribution([int(usage.get("request_count", 0)) for usage in usages]),
        "input_tokens_per_request": _distribution(available_inputs),
        "serialized_context_bytes_per_request": _distribution(
            [int(request.get("serialized_context_bytes", 0)) for request in requests]
        ),
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


def _empty_model_usage(worker_id: int, contract: TaskContract) -> dict[str, Any]:
    return {
        "worker_id": worker_id,
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
    }


def _result_summary(result: dict[str, Any]) -> dict[str, Any]:
    return {
        "task_idx": result.get("task_idx"),
        "task_id": result.get("task_id"),
        "status": result.get("status", "FAIL_UNKNOWN"),
        "agent_answer": result.get("agent_answer"),
        "error": result.get("error"),
        "model_usage": result.get("model_usage", {}),
        "duration_seconds": result.get("duration_seconds", 0.0),
    }


def _load_current_run_artifact(path: Path, run_fingerprint: str, run_id: str) -> dict[str, Any] | None:
    if _has_watchdog_failure(path, run_fingerprint, run_id):
        return None
    try:
        result = json.loads((path / "result.json").read_text(encoding="utf-8"))
        evidence = json.loads((path / "evidence.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if (
        not isinstance(result, dict)
        or not isinstance(evidence, list)
        or result.get("run_fingerprint") != run_fingerprint
        or result.get("run_id") != run_id
        or not isinstance(result.get("status"), str)
    ):
        return None
    return _result_summary(result)


def _write_stable_worker_failure(
    output_root: Path,
    item: dict[str, Any],
    worker_id: int,
    run_fingerprint: str,
    run_id: str,
    status: str,
    error: str,
    *,
    parent_only: bool = False,
) -> dict[str, Any]:
    contract = TaskContract.from_item(item)
    task_dir = task_directory(output_root, item)
    result = {
        "task_idx": contract.task_idx,
        "task_id": contract.task_id,
        "task": contract.task,
        "website": contract.website,
        "status": status,
        "agent_answer": None,
        "evidence_ids": [],
        "evidence_bindings": {},
        "coverage": None,
        "actions": [],
        "thoughts": [],
        "urls": [],
        "receipts": [],
        "predict_length": 0,
        "model_usage": _empty_model_usage(worker_id, contract),
        "error": error,
        "duration_seconds": 0.0,
        "final_result_response": "",
        "worker_id": worker_id,
        "run_fingerprint": run_fingerprint,
        "run_id": run_id,
        "run_manifest_version": MANIFEST_SCHEMA_VERSION,
    }
    if parent_only:
        atomic_write_json(task_dir / "watchdog_failure.json", result)
    else:
        atomic_write_json(task_dir / "evidence.json", [])
        atomic_write_json(task_dir / "result.json", result)
    return _result_summary(result)


def _queue_message(queue: Any | None, value: dict[str, Any]) -> None:
    if queue is None:
        return
    try:
        queue.put(value)
    except Exception:
        pass


async def _worker_async(
    config: WorkerConfig,
    items: list[dict[str, Any]],
    rate_limit_state: tuple[Any, Any, int, float, Any] | None = None,
    progress_queue: Any | None = None,
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
            calibration_samples=rate_limit_state[4],
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
    actor_reset_required = False
    try:
        for item in items:
            task_started = time.monotonic()
            contract = TaskContract.from_item(item, config.max_steps)
            task_key = task_directory(Path("."), item).name
            _queue_message(
                progress_queue,
                {
                    "kind": "task_started",
                    "worker_id": config.worker_id,
                    "task_key": task_key,
                    "wall_time": time.time(),
                },
            )
            task_dir = task_directory(output_root, item)
            task_dir.mkdir(parents=True, exist_ok=True)
            evidence_store = EvidenceStore()
            try:
                if actor.poisoned or actor_reset_required:
                    await actor.retire()
                    actor = BrowserActor(
                        config.cdp_url,
                        output_root / f".worker-{config.worker_id}",
                        EvidenceStore(),
                    )
                    actor_reset_required = False
                await actor.begin_task(contract.website, task_dir, evidence_store)
                run_result = await agent.run(
                    actor,
                    contract,
                    evidence_store,
                    progress_callback=lambda phase, key=task_key: _queue_message(
                        progress_queue,
                        {
                            "kind": "heartbeat",
                            "worker_id": config.worker_id,
                            "task_key": key,
                            "phase": phase,
                        },
                    ),
                )
            except Exception as exc:
                actor_reset_required = True
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
                    "model_usage": _empty_model_usage(config.worker_id, contract),
                    "error": sanitize_exception(exc),
                }
            run_result["duration_seconds"] = round(time.monotonic() - task_started, 3)
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
                "run_fingerprint": config.run_fingerprint,
                "run_id": config.run_id,
                "run_manifest_version": MANIFEST_SCHEMA_VERSION,
            }
            atomic_write_json(task_dir / "result.json", result)
            summary = _result_summary(result)
            summaries.append(summary)
            _queue_message(
                progress_queue,
                {
                    "kind": "task_completed",
                    "worker_id": config.worker_id,
                    "task_key": task_key,
                    "summary": summary,
                },
            )
    finally:
        await actor.close()
    return summaries


def worker_entry(
    config_dict: dict[str, Any],
    items: list[dict[str, Any]],
    queue: Any,
    rate_limit_state: tuple[Any, Any, int, float, Any] | None = None,
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
        summaries = asyncio.run(_worker_async(config, items, rate_limit_state, queue))
        queue.put(
            {
                "kind": "worker_done",
                "worker_id": config.worker_id,
                "completed_count": len(summaries),
                "error": None,
            }
        )
    except BaseException as exc:
        queue.put(
            {
                "kind": "worker_done",
                "worker_id": config.worker_id,
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
    *,
    force_rerun: bool = False,
    worker_watchdog_seconds: float = 900.0,
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
    for item in items:
        TaskContract.from_item(item, max_steps)
    task_keys = [task_directory(Path("."), item).name for item in items]
    if len(set(task_keys)) != len(task_keys):
        raise ValueError("输入中存在重复的 task_idx/task_id，无法保证 worker 输出隔离")
    output_dir.mkdir(parents=True, exist_ok=True)
    if not math.isfinite(worker_watchdog_seconds) or worker_watchdog_seconds <= 0:
        raise ValueError("worker watchdog 超时必须是有限正数")
    configured_worker_count = min(len(cdp_urls), len(items), 8)
    manifest_path = output_dir / "run_manifest.json"
    existing_manifest = load_manifest(manifest_path)
    manifest = build_run_manifest(
        input_path,
        cdp_urls,
        model,
        api_base,
        max_steps,
        configured_worker_count,
        worker_watchdog_seconds,
    )
    can_resume = not force_rerun and manifest_matches(existing_manifest, manifest)
    if can_resume and existing_manifest is not None:
        manifest["run_id"] = existing_manifest["run_id"]
        manifest["created_at"] = existing_manifest.get("created_at", manifest["created_at"])
    atomic_write_json(manifest_path, manifest)
    fingerprint = str(manifest["fingerprint"])
    run_id = str(manifest["run_id"])
    pending = [
        item
        for item in items
        if not is_completed(
            task_directory(output_dir, item),
            fingerprint,
            run_id,
            manifest_valid=can_resume,
        )
    ]
    reused = len(items) - len(pending)
    if not pending:
        summary = {
            "total": len(items),
            "pending": 0,
            "completed": 0,
            "reused": reused,
            "run_fingerprint": fingerprint,
            "run_id": run_id,
            "model_usage": _aggregate_usage([]),
            "task_duration_seconds": _distribution([]),
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
    worker_states: dict[int, dict[str, Any]] = {}

    def handle_message(message: dict[str, Any]) -> None:
        worker_id = int(message.get("worker_id", -1))
        state = worker_states.get(worker_id)
        if state is None or state.get("kill_failed"):
            return
        state["last_progress"] = time.monotonic()
        kind = message.get("kind")
        if kind == "task_started":
            state["current_task"] = message.get("task_key")
            state["current_task_wall_time"] = message.get("wall_time")
        elif kind == "task_completed":
            task_key = str(message.get("task_key", ""))
            if task_key and task_key not in state["completed_keys"]:
                state["completed_keys"].add(task_key)
                state["summaries"].append(message["summary"])
            state["current_task"] = None
        elif kind == "worker_done":
            state["done"] = True
            state["done_at"] = time.monotonic()
            if not state["watchdog"]:
                state["error"] = message.get("error")

    def join_process(process: Any, timeout: float = 2.0) -> None:
        try:
            process.join(timeout=timeout)
        except TypeError:
            process.join()

    def stop_process(process: Any) -> bool:
        try:
            if process.is_alive():
                process.terminate()
        except Exception:
            pass
        try:
            join_process(process)
        except Exception:
            pass
        try:
            if process.is_alive() and hasattr(process, "kill"):
                process.kill()
                join_process(process)
        except Exception:
            pass
        try:
            return not process.is_alive()
        except Exception:
            return False

    try:
        queue = context.Queue()
        manager = context.Manager() if tpm_budget is not None else None
        rate_limit_state = (
            (manager.list(), manager.RLock(), tpm_budget, 60.0, manager.list())
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
                run_fingerprint=fingerprint,
                run_id=run_id,
            )
            process = context.Process(
                target=worker_entry,
                args=(asdict(config), shard, queue, rate_limit_state),
            )
            process.start()
            processes.append(process)
            worker_states[worker_id] = {
                "worker_id": worker_id,
                "items": shard,
                "summaries": [],
                "completed_keys": set(),
                "last_progress": time.monotonic(),
                "current_task": None,
                "current_task_wall_time": None,
                "done": False,
                "done_at": None,
                "error": None,
                "watchdog": False,
                "cleanup_forced": False,
                "monitoring_done": False,
                "kill_failed": False,
            }

        cleanup_grace = min(5.0, max(0.1, worker_watchdog_seconds))
        while any(
            not worker_states[worker_id]["monitoring_done"] and process.is_alive()
            for worker_id, process in enumerate(processes)
        ):
            try:
                handle_message(queue.get(timeout=min(0.25, worker_watchdog_seconds)))
            except queue_module.Empty:
                pass
            while True:
                try:
                    handle_message(queue.get_nowait())
                except (queue_module.Empty, AttributeError):
                    break
            now = time.monotonic()
            for worker_id, process in enumerate(processes):
                if not process.is_alive():
                    continue
                state = worker_states[worker_id]
                if state["monitoring_done"]:
                    continue
                if state["done"] and now - float(state["done_at"]) >= cleanup_grace:
                    state["cleanup_forced"] = True
                    state["monitoring_done"] = True
                    if not stop_process(process):
                        state["kill_failed"] = True
                        state["error"] = f"WORKER_KILL_FAILED: worker {worker_id} 清理后仍存活"
                    continue
                if not state["done"] and now - float(state["last_progress"]) >= worker_watchdog_seconds:
                    state["watchdog"] = True
                    state["error"] = (
                        f"WORKER_WATCHDOG_TIMEOUT: worker {worker_id} "
                        f"连续 {worker_watchdog_seconds:g} 秒无进度"
                    )
                    state["monitoring_done"] = True
                    if not stop_process(process):
                        state["kill_failed"] = True
                        state["error"] = f"WORKER_KILL_FAILED: worker {worker_id} 超时且无法终止"
        for process in processes:
            join_process(process)
        while True:
            try:
                handle_message(queue.get(timeout=0.1))
            except queue_module.Empty:
                break
    except BaseException:
        for process in processes:
            try:
                if process.is_alive():
                    process.terminate()
            except Exception:
                pass
        for process in processes:
            try:
                join_process(process)
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
    worker_results: list[dict[str, Any]] = []
    for worker_id, process in enumerate(processes):
        state = worker_states[worker_id]
        if not state["done"] and not state["error"]:
            state["error"] = f"WORKER_EXITED_WITHOUT_RESULT: 进程退出码 {process.exitcode}"
        missing_items: list[dict[str, Any]] = []
        for item in state["items"]:
            task_key = task_directory(Path("."), item).name
            if task_key in state["completed_keys"]:
                continue
            landed = (
                None
                if state["watchdog"]
                else _load_current_run_artifact(task_directory(output_dir, item), fingerprint, run_id)
            )
            if landed is not None:
                state["completed_keys"].add(task_key)
                state["summaries"].append(landed)
            else:
                missing_items.append(item)
        if state["watchdog"]:
            failure_status = "FAIL_WORKER_WATCHDOG"
        elif state["error"]:
            failure_status = "FAIL_WORKER_EXIT"
        else:
            failure_status = "FAIL_WORKER_INCOMPLETE"
        failure_error = str(state["error"] or "WORKER_INCOMPLETE: worker 未返回任务完成事件")
        for item in missing_items:
            summary_item = _write_stable_worker_failure(
                output_dir,
                item,
                worker_id,
                fingerprint,
                run_id,
                failure_status,
                failure_error,
                parent_only=state["kill_failed"],
            )
            state["summaries"].append(summary_item)
        state["summaries"].sort(key=lambda value: task_keys.index(task_directory(Path("."), value).name))
        worker_results.append(
            {
                "worker_id": worker_id,
                "summaries": state["summaries"],
                "model_usage": _aggregate_usage(state["summaries"]),
                "error": state["error"],
                "watchdog_triggered": state["watchdog"],
                "cleanup_forced": state["cleanup_forced"],
                "kill_failed": state["kill_failed"],
            }
        )
    summaries = [entry for worker in worker_results for entry in worker["summaries"]]
    summary = {
        "total": len(items),
        "pending": len(pending),
        "completed": len(summaries),
        "reused": reused,
        "run_fingerprint": fingerprint,
        "run_id": run_id,
        "success": sum(item["status"] == "SUCCESS" for item in summaries),
        "failed": sum(item["status"] != "SUCCESS" for item in summaries),
        "tpm_safety_budget": tpm_budget,
        "model_usage": _aggregate_usage(summaries),
        "task_duration_seconds": _distribution(
            [float(item.get("duration_seconds", 0.0)) for item in summaries]
        ),
        "workers": worker_results,
    }
    atomic_write_json(output_dir / "logs" / "summary.json", summary)
    return summary
