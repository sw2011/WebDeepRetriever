from __future__ import annotations

import asyncio
import json
import logging
import math
import time
import uuid
import hashlib
from collections import Counter
from collections.abc import AsyncIterator, Callable
from dataclasses import dataclass, field
from typing import Any

from agents import Model, ModelResponse


logger = logging.getLogger(__name__)
MAX_SERIALIZED_CONTEXT_BYTES = 88_000
_COLD_TOKENS_PER_BYTE = {"agent": 0.30, "vision": 1.0}
_FIXED_TOKEN_MARGIN = {"agent": 1_024, "vision": 2_048}
# The 2026-07-31 Kimi K2.6 baseline had 454 agent requests with
# actual_input_tokens / serialized_bytes p95=0.2675 and max=0.2779.
# Cold start rounds that envelope to 0.30; shared samples then use p95 plus margin.


def model_input_metrics(
    system_instructions: str | None,
    input_items: Any,
    tools: list[Any],
) -> dict[str, Any]:
    tool_schemas = [
        {
            "name": getattr(tool, "name", type(tool).__name__),
            "description": getattr(tool, "description", ""),
            "parameters": getattr(tool, "params_json_schema", None),
        }
        for tool in tools
    ]
    payload = {
        "instructions": system_instructions or "",
        "input": input_items,
        "tools": tool_schemas,
    }
    serialized = json.dumps(
        payload,
        ensure_ascii=False,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")
    category_bytes: Counter[str] = Counter()
    category_bytes["instructions"] = len((system_instructions or "").encode("utf-8"))
    category_bytes["tool_schemas"] = len(
        json.dumps(tool_schemas, ensure_ascii=False, separators=(",", ":"), default=str).encode("utf-8")
    )
    items = input_items if isinstance(input_items, list) else [input_items]
    for item in items:
        encoded_size = len(
            json.dumps(item, ensure_ascii=False, separators=(",", ":"), default=str).encode("utf-8")
        )
        if isinstance(item, dict) and item.get("type") == "function_call":
            category = "function_calls"
        elif isinstance(item, dict) and item.get("type") == "function_call_output":
            category = "function_outputs"
        elif isinstance(item, dict) and item.get("role") == "user":
            category = "user_items"
        else:
            category = "other_items"
        category_bytes[category] += encoded_size
    return {
        "serialized_context_bytes": len(serialized),
        "serialized_context_items": len(items),
        "category_bytes": dict(sorted(category_bytes.items())),
    }


def estimate_input_tokens(
    system_instructions: str | None,
    input_items: Any,
    tools: list[Any],
    *,
    channel: str = "agent",
    tokens_per_byte: float | None = None,
) -> int:
    """Estimate tokens from serialized bytes using measured, channel-specific ratios."""

    serialized_bytes = int(model_input_metrics(system_instructions, input_items, tools)["serialized_context_bytes"])
    normalized_channel = "vision" if channel == "vision" else "agent"
    ratio = tokens_per_byte if tokens_per_byte is not None else _COLD_TOKENS_PER_BYTE[normalized_channel]
    return max(1, math.ceil(serialized_bytes * ratio) + _FIXED_TOKEN_MARGIN[normalized_channel])


@dataclass
class TaskUsageStats:
    worker_id: int | None
    task_idx: int | str
    task_id: str
    requests: list[dict[str, Any]] = field(default_factory=list)
    runtime_snapshot: dict[str, Any] = field(default_factory=dict)

    def set_runtime_snapshot(self, value: dict[str, Any]) -> None:
        self.runtime_snapshot = dict(value)

    def record(
        self,
        *,
        estimated_input_tokens: int,
        wait_seconds: float,
        throttle_reason: str | None,
        usage: Any | None,
        error_type: str | None = None,
        channel: str = "agent",
        input_metrics: dict[str, Any] | None = None,
        model_latency_ms: float = 0.0,
        reservation: dict[str, Any] | None = None,
    ) -> None:
        usage_available = usage is not None and any(
            int(getattr(usage, key, 0) or 0)
            for key in (
                "requests",
                "input_tokens",
                "prompt_tokens",
                "output_tokens",
                "completion_tokens",
                "total_tokens",
            )
        )
        input_tokens = (
            int(getattr(usage, "input_tokens", getattr(usage, "prompt_tokens", 0)))
            if usage_available
            else None
        )
        output_tokens = (
            int(getattr(usage, "output_tokens", getattr(usage, "completion_tokens", 0)))
            if usage_available
            else None
        )
        total_tokens = (
            int(getattr(usage, "total_tokens", (input_tokens or 0) + (output_tokens or 0)))
            if usage_available
            else None
        )
        entry = {
            "request": len(self.requests) + 1,
            "worker_id": self.worker_id,
            "task_idx": self.task_idx,
            "task_id_hash": hashlib.sha256(self.task_id.encode("utf-8")).hexdigest()[:12],
            "channel": channel,
            "estimated_input_tokens": estimated_input_tokens,
            "wait_seconds": round(wait_seconds, 3),
            "throttle_reason": throttle_reason,
            "usage_available": usage_available,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": total_tokens,
            "error_type": error_type,
            "serialized_context_bytes": int((input_metrics or {}).get("serialized_context_bytes", 0)),
            "serialized_context_items": int((input_metrics or {}).get("serialized_context_items", 0)),
            "category_bytes": dict((input_metrics or {}).get("category_bytes", {})),
            "model_latency_ms": round(model_latency_ms, 3),
            "reserved_tokens": int((reservation or {}).get("reserved_tokens", 0)),
            **self.runtime_snapshot,
        }
        self.requests.append(entry)
        logger.info(
            "model_request worker=%s task=%s request=%s estimated_input=%s input=%s output=%s "
            "wait_seconds=%s throttle_reason=%s usage_available=%s error_type=%s",
            self.worker_id,
            self.task_idx,
            entry["request"],
            estimated_input_tokens,
            entry["input_tokens"],
            entry["output_tokens"],
            entry["wait_seconds"],
            throttle_reason or "none",
            entry["usage_available"],
            error_type or "none",
        )

    def to_dict(self) -> dict[str, Any]:
        available = [item for item in self.requests if item["usage_available"]]
        reasons = Counter(
            str(item["throttle_reason"])
            for item in self.requests
            if item.get("throttle_reason")
        )
        return {
            "worker_id": self.worker_id,
            "task_idx": self.task_idx,
            "task_id": self.task_id,
            "request_count": len(self.requests),
            "usage_available_count": len(available),
            "usage_unavailable_count": len(self.requests) - len(available),
            "input_tokens": sum(int(item["input_tokens"]) for item in available),
            "output_tokens": sum(int(item["output_tokens"]) for item in available),
            "total_tokens": sum(int(item["total_tokens"]) for item in available),
            "estimated_input_tokens": sum(int(item["estimated_input_tokens"]) for item in self.requests),
            "throttle_wait_seconds": round(sum(float(item["wait_seconds"]) for item in self.requests), 3),
            "throttle_reasons": dict(sorted(reasons.items())),
            "requests": list(self.requests),
        }


class SharedTPMLimiter:
    """A process-shared conservative sliding-window input-token limiter."""

    def __init__(
        self,
        events: Any,
        lock: Any,
        *,
        token_budget: int,
        window_seconds: float = 60.0,
        clock: Callable[[], float] = time.monotonic,
        sleeper: Callable[[float], Any] = asyncio.sleep,
        calibration_samples: Any | None = None,
    ) -> None:
        if token_budget < 1:
            raise ValueError("TPM 安全线必须大于 0")
        self.events = events
        self.lock = lock
        self.token_budget = token_budget
        self.window_seconds = window_seconds
        self.clock = clock
        self.sleeper = sleeper
        self.calibration_samples = calibration_samples if calibration_samples is not None else []

    @staticmethod
    def _event_parts(event: Any) -> tuple[float, int, str | None]:
        if len(event) >= 3:
            return float(event[0]), int(event[1]), str(event[2])
        return float(event[0]), int(event[1]), None

    def _acquire_lock(self) -> None:
        if not self.lock.acquire(timeout=5.0):
            raise RuntimeError("TPM_LIMITER_LOCK_TIMEOUT: 共享限流锁不可用，已阻止发送")

    def estimate_tokens(self, serialized_bytes: int, channel: str = "agent") -> int:
        normalized_channel = "vision" if channel == "vision" else "agent"
        self._acquire_lock()
        try:
            ratios = sorted(
                float(item[2])
                for item in list(self.calibration_samples)[-512:]
                if len(item) >= 3 and str(item[1]) == normalized_channel and float(item[2]) > 0
            )
        finally:
            self.lock.release()
        if len(ratios) < 20:
            ratio = _COLD_TOKENS_PER_BYTE[normalized_channel]
        else:
            p95 = ratios[max(0, math.ceil(len(ratios) * 0.95) - 1)]
            floor = 0.28 if normalized_channel == "agent" else 0.75
            margin = 1.12 if normalized_channel == "agent" else 1.20
            ratio = max(floor, p95 * margin)
        return max(1, math.ceil(serialized_bytes * ratio) + _FIXED_TOKEN_MARGIN[normalized_channel])

    def _try_acquire(self, estimated_tokens: int) -> tuple[bool, float, int, str | None, float]:
        if estimated_tokens > self.token_budget:
            raise ValueError(
                f"单请求保守输入估算 {estimated_tokens} 超过 TPM 安全线 {self.token_budget}，已阻止发送"
            )
        self._acquire_lock()
        try:
            now = self.clock()
            active = [event for event in list(self.events) if now - self._event_parts(event)[0] < self.window_seconds]
            self.events[:] = active
            used = sum(self._event_parts(event)[1] for event in active)
            if used + estimated_tokens <= self.token_budget:
                reservation_id = uuid.uuid4().hex
                self.events.append((now, estimated_tokens, reservation_id))
                return True, 0.0, used, reservation_id, now
            oldest = min(self._event_parts(event)[0] for event in active)
            delay = max(0.01, self.window_seconds - (now - oldest) + 0.01)
            return False, delay, used, None, now
        finally:
            self.lock.release()

    def try_acquire(self, estimated_tokens: int) -> tuple[bool, float, int]:
        granted, delay, used, _, _ = self._try_acquire(estimated_tokens)
        return granted, delay, used

    async def acquire(self, estimated_tokens: int) -> dict[str, Any]:
        started = self.clock()
        reason: str | None = None
        while True:
            granted, delay, used, reservation_id, reserved_at = self._try_acquire(estimated_tokens)
            if granted:
                return {
                    "wait_seconds": max(0.0, self.clock() - started),
                    "reason": reason,
                    "reserved_tokens": estimated_tokens,
                    "window_tokens_before": used,
                    "reservation_id": reservation_id,
                    "reserved_at": reserved_at,
                }
            reason = "pre_send_tpm_capacity"
            await self.sleeper(min(delay, 1.0))

    def reconcile(
        self,
        reservation: dict[str, Any],
        actual_input_tokens: int | None,
        serialized_bytes: int,
        channel: str = "agent",
    ) -> None:
        if actual_input_tokens is None or actual_input_tokens <= 0:
            return
        reservation_id = reservation.get("reservation_id")
        if not reservation_id:
            return
        normalized_channel = "vision" if channel == "vision" else "agent"
        self._acquire_lock()
        try:
            updated: list[Any] = []
            found = False
            for event in list(self.events):
                timestamp, tokens, event_id = self._event_parts(event)
                if event_id == reservation_id:
                    updated.append((timestamp, int(actual_input_tokens), event_id))
                    found = True
                else:
                    updated.append(event)
            expired = self.clock() - float(reservation.get("reserved_at", self.clock())) >= self.window_seconds
            if not found and not expired:
                raise RuntimeError("TPM_RECONCILE_MISSING_RESERVATION: 预约已丢失，拒绝静默校准")
            if found:
                self.events[:] = updated
            if serialized_bytes > 0:
                self.calibration_samples.append(
                    (self.clock(), normalized_channel, float(actual_input_tokens) / serialized_bytes)
                )
                if len(self.calibration_samples) > 2_048:
                    self.calibration_samples[:] = list(self.calibration_samples)[-1_024:]
        finally:
            self.lock.release()


class ThrottledModel(Model):
    def __init__(self, wrapped: Model, limiter: SharedTPMLimiter | None) -> None:
        self.wrapped = wrapped
        self.limiter = limiter
        self.usage_stats: TaskUsageStats | None = None

    async def get_response(
        self,
        system_instructions: str | None,
        input: Any,
        model_settings: Any,
        tools: list[Any],
        output_schema: Any,
        handoffs: list[Any],
        tracing: Any,
        *,
        previous_response_id: str | None,
        conversation_id: str | None,
        prompt: Any,
    ) -> ModelResponse:
        metrics = model_input_metrics(system_instructions, input, tools)
        serialized_bytes = int(metrics["serialized_context_bytes"])
        cold_estimate = estimate_input_tokens(system_instructions, input, tools)
        if serialized_bytes > MAX_SERIALIZED_CONTEXT_BYTES:
            exc = ValueError(
                f"MODEL_INPUT_LIMIT: 序列化模型输入 {serialized_bytes} 字节超过硬上限 {MAX_SERIALIZED_CONTEXT_BYTES}"
            )
            if self.usage_stats is not None:
                self.usage_stats.record(
                    estimated_input_tokens=cold_estimate,
                    wait_seconds=0.0,
                    throttle_reason="pre_send_context_limit",
                    usage=None,
                    error_type=type(exc).__name__,
                    input_metrics=metrics,
                )
            raise exc
        estimate = cold_estimate
        try:
            if self.limiter is not None and hasattr(self.limiter, "estimate_tokens"):
                estimate = self.limiter.estimate_tokens(serialized_bytes, "agent")
            reservation = (
                await self.limiter.acquire(estimate)
                if self.limiter is not None
                else {"wait_seconds": 0.0, "reason": None, "reserved_tokens": estimate}
            )
        except (ValueError, RuntimeError) as exc:
            if self.usage_stats is not None:
                self.usage_stats.record(
                    estimated_input_tokens=estimate,
                    wait_seconds=0.0,
                    throttle_reason=(
                        "pre_send_tpm_lock_timeout"
                        if isinstance(exc, RuntimeError)
                        else "pre_send_request_exceeds_tpm_budget"
                    ),
                    usage=None,
                    error_type=type(exc).__name__,
                    input_metrics=metrics,
                )
            raise
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            if self.usage_stats is not None:
                self.usage_stats.record(
                    estimated_input_tokens=estimate,
                    wait_seconds=0.0,
                    throttle_reason="pre_send_tpm_limiter_error",
                    usage=None,
                    error_type=type(exc).__name__,
                    input_metrics=metrics,
                )
            raise
        model_started = time.monotonic()
        try:
            response = await self.wrapped.get_response(
                system_instructions,
                input,
                model_settings,
                tools,
                output_schema,
                handoffs,
                tracing,
                previous_response_id=previous_response_id,
                conversation_id=conversation_id,
                prompt=prompt,
            )
        except asyncio.CancelledError as exc:
            latency = (time.monotonic() - model_started) * 1_000
            if self.usage_stats is not None:
                self.usage_stats.record(
                    estimated_input_tokens=estimate,
                    wait_seconds=reservation["wait_seconds"],
                    throttle_reason=reservation["reason"],
                    usage=None,
                    error_type=type(exc).__name__,
                    input_metrics=metrics,
                    model_latency_ms=latency,
                    reservation=reservation,
                )
            raise
        except Exception as exc:
            latency = (time.monotonic() - model_started) * 1_000
            if self.usage_stats is not None:
                self.usage_stats.record(
                    estimated_input_tokens=estimate,
                    wait_seconds=reservation["wait_seconds"],
                    throttle_reason=reservation["reason"],
                    usage=None,
                    error_type=type(exc).__name__,
                    input_metrics=metrics,
                    model_latency_ms=latency,
                    reservation=reservation,
                )
            raise
        latency = (time.monotonic() - model_started) * 1_000
        actual_input_tokens = int(
            getattr(response.usage, "input_tokens", getattr(response.usage, "prompt_tokens", 0)) or 0
        )
        if self.limiter is not None and actual_input_tokens:
            try:
                self.limiter.reconcile(reservation, actual_input_tokens, serialized_bytes, "agent")
            except Exception as exc:
                if self.usage_stats is not None:
                    self.usage_stats.record(
                        estimated_input_tokens=estimate,
                        wait_seconds=reservation["wait_seconds"],
                        throttle_reason="post_send_tpm_reconcile_error",
                        usage=response.usage,
                        error_type=type(exc).__name__,
                        input_metrics=metrics,
                        model_latency_ms=latency,
                        reservation=reservation,
                    )
                raise
        if self.usage_stats is not None:
            self.usage_stats.record(
                estimated_input_tokens=estimate,
                wait_seconds=reservation["wait_seconds"],
                throttle_reason=reservation["reason"],
                usage=response.usage,
                input_metrics=metrics,
                model_latency_ms=latency,
                reservation=reservation,
            )
        return response

    async def stream_response(
        self,
        system_instructions: str | None,
        input: Any,
        model_settings: Any,
        tools: list[Any],
        output_schema: Any,
        handoffs: list[Any],
        tracing: Any,
        *,
        previous_response_id: str | None,
        conversation_id: str | None,
        prompt: Any,
    ) -> AsyncIterator[Any]:
        metrics = model_input_metrics(system_instructions, input, tools)
        serialized_bytes = int(metrics["serialized_context_bytes"])
        cold_estimate = estimate_input_tokens(system_instructions, input, tools)
        if serialized_bytes > MAX_SERIALIZED_CONTEXT_BYTES:
            exc = ValueError(
                f"MODEL_INPUT_LIMIT: 序列化模型输入 {serialized_bytes} 字节超过硬上限 {MAX_SERIALIZED_CONTEXT_BYTES}"
            )
            if self.usage_stats is not None:
                self.usage_stats.record(
                    estimated_input_tokens=cold_estimate,
                    wait_seconds=0.0,
                    throttle_reason="pre_send_context_limit",
                    usage=None,
                    error_type=type(exc).__name__,
                    channel="agent_stream",
                    input_metrics=metrics,
                )
            raise exc
        estimate = cold_estimate
        try:
            if self.limiter is not None and hasattr(self.limiter, "estimate_tokens"):
                estimate = self.limiter.estimate_tokens(serialized_bytes, "agent")
            reservation = (
                await self.limiter.acquire(estimate)
                if self.limiter is not None
                else {"wait_seconds": 0.0, "reason": None, "reserved_tokens": estimate}
            )
        except (ValueError, RuntimeError) as exc:
            if self.usage_stats is not None:
                self.usage_stats.record(
                    estimated_input_tokens=estimate,
                    wait_seconds=0.0,
                    throttle_reason=(
                        "pre_send_tpm_lock_timeout"
                        if isinstance(exc, RuntimeError)
                        else "pre_send_request_exceeds_tpm_budget"
                    ),
                    usage=None,
                    error_type=type(exc).__name__,
                    channel="agent_stream",
                    input_metrics=metrics,
                )
            raise
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            if self.usage_stats is not None:
                self.usage_stats.record(
                    estimated_input_tokens=estimate,
                    wait_seconds=0.0,
                    throttle_reason="pre_send_tpm_limiter_error",
                    usage=None,
                    error_type=type(exc).__name__,
                    channel="agent_stream",
                    input_metrics=metrics,
                )
            raise
        model_started = time.monotonic()
        recorded = False
        terminal_error_type: str | None = None

        def finalize(usage: Any | None, error_type: str | None = None) -> None:
            nonlocal recorded
            if recorded:
                return
            latency = (time.monotonic() - model_started) * 1_000
            actual_input_tokens = int(
                getattr(usage, "input_tokens", getattr(usage, "prompt_tokens", 0)) or 0
            )
            if self.limiter is not None and actual_input_tokens:
                try:
                    self.limiter.reconcile(reservation, actual_input_tokens, serialized_bytes, "agent")
                except Exception as exc:
                    recorded = True
                    if self.usage_stats is not None:
                        self.usage_stats.record(
                            estimated_input_tokens=estimate,
                            wait_seconds=reservation["wait_seconds"],
                            throttle_reason="post_send_tpm_reconcile_error",
                            usage=usage,
                            error_type=type(exc).__name__,
                            channel="agent_stream",
                            input_metrics=metrics,
                            model_latency_ms=latency,
                            reservation=reservation,
                        )
                    raise
            recorded = True
            if self.usage_stats is not None:
                self.usage_stats.record(
                    estimated_input_tokens=estimate,
                    wait_seconds=reservation["wait_seconds"],
                    throttle_reason=reservation["reason"],
                    usage=usage,
                    error_type=error_type,
                    channel="agent_stream",
                    input_metrics=metrics,
                    model_latency_ms=latency,
                    reservation=reservation,
                )

        try:
            async for event in self.wrapped.stream_response(
                system_instructions,
                input,
                model_settings,
                tools,
                output_schema,
                handoffs,
                tracing,
                previous_response_id=previous_response_id,
                conversation_id=conversation_id,
                prompt=prompt,
            ):
                if getattr(event, "type", None) == "response.completed":
                    response = getattr(event, "response", None)
                    finalize(getattr(response, "usage", None))
                yield event
        except asyncio.CancelledError as exc:
            terminal_error_type = type(exc).__name__
            raise
        except Exception as exc:
            terminal_error_type = type(exc).__name__
            raise
        finally:
            if not recorded:
                finalize(None, terminal_error_type)

    def get_retry_advice(self, request: Any) -> Any:
        return self.wrapped.get_retry_advice(request)

    async def _cleanup_on_run_end(self, owner: object) -> None:
        await self.wrapped._cleanup_on_run_end(owner)

    async def close(self) -> None:
        await self.wrapped.close()
