from __future__ import annotations

import asyncio
import json
import logging
import time
from collections import Counter
from collections.abc import AsyncIterator, Callable
from dataclasses import dataclass, field
from typing import Any

from agents import Model, ModelResponse


logger = logging.getLogger(__name__)


def estimate_input_tokens(
    system_instructions: str | None,
    input_items: Any,
    tools: list[Any],
) -> int:
    """Return a deliberately conservative byte-based upper bound for input tokens."""

    tool_schemas = [
        {
            "name": getattr(tool, "name", type(tool).__name__),
            "description": getattr(tool, "description", ""),
            "parameters": getattr(tool, "params_json_schema", None),
        }
        for tool in tools
    ]
    serialized = json.dumps(
        {
            "instructions": system_instructions or "",
            "input": input_items,
            "tools": tool_schemas,
        },
        ensure_ascii=False,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")
    # Byte length is above the token count for the byte-level tokenizers used by
    # OpenAI-compatible providers. The fixed margin covers chat framing overhead.
    return max(1, len(serialized) + 8_192)


@dataclass
class TaskUsageStats:
    worker_id: int | None
    task_idx: int | str
    task_id: str
    requests: list[dict[str, Any]] = field(default_factory=list)

    def record(
        self,
        *,
        estimated_input_tokens: int,
        wait_seconds: float,
        throttle_reason: str | None,
        usage: Any | None,
        error_type: str | None = None,
        channel: str = "agent",
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
            "channel": channel,
            "estimated_input_tokens": estimated_input_tokens,
            "wait_seconds": round(wait_seconds, 3),
            "throttle_reason": throttle_reason,
            "usage_available": usage_available,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": total_tokens,
            "error_type": error_type,
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
    ) -> None:
        if token_budget < 1:
            raise ValueError("TPM 安全线必须大于 0")
        self.events = events
        self.lock = lock
        self.token_budget = token_budget
        self.window_seconds = window_seconds
        self.clock = clock
        self.sleeper = sleeper

    def try_acquire(self, estimated_tokens: int) -> tuple[bool, float, int]:
        if estimated_tokens > self.token_budget:
            raise ValueError(
                f"单请求保守输入估算 {estimated_tokens} 超过 TPM 安全线 {self.token_budget}，已阻止发送"
            )
        acquired = self.lock.acquire(timeout=5.0)
        if not acquired:
            raise RuntimeError("TPM_LIMITER_LOCK_TIMEOUT: 共享限流锁不可用，已阻止发送")
        try:
            now = self.clock()
            active = [
                (float(timestamp), int(tokens))
                for timestamp, tokens in list(self.events)
                if now - float(timestamp) < self.window_seconds
            ]
            self.events[:] = active
            used = sum(tokens for _, tokens in active)
            if used + estimated_tokens <= self.token_budget:
                self.events.append((now, estimated_tokens))
                return True, 0.0, used
            oldest = min(timestamp for timestamp, _ in active)
            delay = max(0.01, self.window_seconds - (now - oldest) + 0.01)
            return False, delay, used
        finally:
            self.lock.release()

    async def acquire(self, estimated_tokens: int) -> dict[str, Any]:
        started = self.clock()
        reason: str | None = None
        while True:
            granted, delay, used = self.try_acquire(estimated_tokens)
            if granted:
                return {
                    "wait_seconds": max(0.0, self.clock() - started),
                    "reason": reason,
                    "reserved_tokens": estimated_tokens,
                    "window_tokens_before": used,
                }
            reason = "pre_send_tpm_capacity"
            await self.sleeper(min(delay, 1.0))


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
        estimate = estimate_input_tokens(system_instructions, input, tools)
        try:
            reservation = (
                await self.limiter.acquire(estimate)
                if self.limiter is not None
                else {"wait_seconds": 0.0, "reason": None}
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
                )
            raise
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
            if self.usage_stats is not None:
                self.usage_stats.record(
                    estimated_input_tokens=estimate,
                    wait_seconds=reservation["wait_seconds"],
                    throttle_reason=reservation["reason"],
                    usage=None,
                    error_type=type(exc).__name__,
                )
            raise
        except Exception as exc:
            if self.usage_stats is not None:
                self.usage_stats.record(
                    estimated_input_tokens=estimate,
                    wait_seconds=reservation["wait_seconds"],
                    throttle_reason=reservation["reason"],
                    usage=None,
                    error_type=type(exc).__name__,
                )
            raise
        if self.usage_stats is not None:
            self.usage_stats.record(
                estimated_input_tokens=estimate,
                wait_seconds=reservation["wait_seconds"],
                throttle_reason=reservation["reason"],
                usage=response.usage,
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
        estimate = estimate_input_tokens(system_instructions, input, tools)
        try:
            reservation = (
                await self.limiter.acquire(estimate)
                if self.limiter is not None
                else {"wait_seconds": 0.0, "reason": None}
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
                )
            raise
        if self.usage_stats is not None:
            self.usage_stats.record(
                estimated_input_tokens=estimate,
                wait_seconds=reservation["wait_seconds"],
                throttle_reason=reservation["reason"],
                usage=None,
                channel="agent_stream",
            )
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
            yield event

    def get_retry_advice(self, request: Any) -> Any:
        return self.wrapped.get_retry_advice(request)

    async def _cleanup_on_run_end(self, owner: object) -> None:
        await self.wrapped._cleanup_on_run_end(owner)

    async def close(self) -> None:
        await self.wrapped.close()
