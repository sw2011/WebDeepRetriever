from __future__ import annotations

import asyncio
import json
import multiprocessing as mp
import threading
from types import SimpleNamespace

import pytest
from agents import FunctionToolResult, MaxTurnsExceeded, RunContextWrapper, Usage
from agents.tool_context import ToolContext
from agents.run_config import CallModelData, ModelInputData
from openai import AsyncOpenAI

from web_agent.contracts import TaskContract
from web_agent.evidence import EvidenceStore
from web_agent.runtime import (
    TOOLS,
    BoundedToolOutputFilter,
    NoProgressLoopError,
    ProtocolRunHooks,
    ProtocolIIIAgent,
    TaskRuntimeContext,
    _coverage_items,
    _error,
    _json,
    _kimi_request_options,
    _model_settings,
    _project_observation,
    _verified_finish_behavior,
    _vision_completion,
)
from web_agent.token_control import SharedTPMLimiter, TaskUsageStats, ThrottledModel, estimate_input_tokens
from web_agent.verifier import CompletionVerifier


class DummyActor:
    async def audit_step(self, label: str) -> str:
        return label


def _reserve_from_process(events: object, lock: object, gate: object, ready: object, results: object) -> None:
    ready.put(True)  # type: ignore[attr-defined]
    gate.wait()  # type: ignore[attr-defined]
    limiter = SharedTPMLimiter(events, lock, token_budget=100, window_seconds=60)
    results.put(limiter.try_acquire(60))  # type: ignore[attr-defined]


def make_context(max_steps: int = 100) -> TaskRuntimeContext:
    return TaskRuntimeContext(
        actor=DummyActor(),  # type: ignore[arg-type]
        contract=TaskContract.from_item(
            {"task_idx": 1, "task_id": "x", "website": "https://example.test", "task": "query"},
            max_steps=max_steps,
        ),
        evidence_store=EvidenceStore(),
        verifier=CompletionVerifier(),
        vision_client=AsyncOpenAI(api_key="test", base_url="http://127.0.0.1:9/v1", max_retries=0),
        vision_model="test-model",
    )


def test_all_function_tools_have_strict_closed_schemas() -> None:
    assert {tool.name for tool in TOOLS} >= {
        "observe",
        "click",
        "fill",
        "select",
        "set_checked",
        "press",
        "scroll",
        "tabs",
        "dialog",
        "upload",
        "download",
        "extract",
        "network",
        "document",
        "recall_evidence",
        "record_coverage",
        "visual_inspect",
        "visual_document",
        "finish",
    }
    for tool in TOOLS:
        assert tool.strict_json_schema is True
        assert tool.params_json_schema.get("additionalProperties") is False


def test_kimi_k26_uses_tool_compatible_settings() -> None:
    settings = _model_settings("kimi-k2.6")
    assert settings.temperature == 0.6
    assert settings.tool_choice == "required"
    assert settings.parallel_tool_calls is False
    assert settings.extra_body == {"thinking": {"type": "disabled"}}

    default_settings = _model_settings("gpt-4.1-mini")
    assert default_settings.temperature == 0
    assert default_settings.extra_body is None
    assert _kimi_request_options("kimi-k2.6") == {
        "temperature": 0.6,
        "extra_body": {"thinking": {"type": "disabled"}},
    }
    assert _kimi_request_options("gpt-4.1-mini") == {}
    agent = ProtocolIIIAgent("kimi-k2.6", "http://127.0.0.1:9/v1", "test")
    assert agent.client.max_retries == 0


def test_hard_step_limit_rejects_call_101() -> None:
    context = make_context(100)
    for index in range(100):
        context.record_call("observe", {"index": index})
    with pytest.raises(RuntimeError, match="STEP_LIMIT"):
        context.record_call("observe", {})


def test_tool_errors_are_sanitized_before_model_output() -> None:
    output = _error(
        "observe",
        RuntimeError("connect https://user:password@example.test?access_token=private Bearer private"),
    )
    assert "private" not in output
    assert "password" not in output
    context = make_context()
    context.record_call("tabs", {"url": "https://example.test?access_token=private"})
    assert "private" not in str(context.actions)


def test_coverage_item_extraction_handles_dom_and_network_shapes() -> None:
    assert _coverage_items({"data": [["A", 1], ["B", 2]]}) == [["A", 1], ["B", 2]]
    assert _coverage_items({"records": [{"body": {"results": [{"id": 1}, {"id": 2}]}}]}) == [
        {"id": 1},
        {"id": 2},
    ]
    assert _coverage_items({"data": "not countable"}) == []


def test_single_user_run_keeps_only_latest_observation_and_summarizes_evidence() -> None:
    items: list[dict[str, object]] = [{"role": "user", "content": "task"}]
    for index in range(10):
        payload = json.dumps(
            {
                "ok": True,
                "url": "https://example.test",
                "dom_hash": str(index),
                "evidence_id": f"ev-{index:05d}",
                "elements": [{"bid": str(index), "text": "x" * 10_000}],
            }
        )
        items.extend(
            [
                {"type": "function_call", "call_id": str(index), "name": "observe", "arguments": "{}"},
                {"type": "function_call_output", "call_id": str(index), "output": payload},
            ]
        )
    model_data = ModelInputData(input=items, instructions="instructions")  # type: ignore[arg-type]
    filtered = BoundedToolOutputFilter(keep_recent_outputs=2, old_output_chars=100)(
        CallModelData(model_data=model_data, agent=SimpleNamespace(), context=None)  # type: ignore[arg-type]
    )
    outputs = [item["output"] for item in filtered.input if isinstance(item, dict) and item.get("type") == "function_call_output"]
    assert sum("compacted_tool_output" in str(output) for output in outputs) == 9
    assert "compacted_tool_output" not in str(outputs[-1])
    assert "ev-00000" in str(outputs[0])
    assert sum(len(str(output)) for output in outputs) <= 72_000


def test_history_compaction_keeps_latest_content_and_useful_old_preview() -> None:
    items: list[dict[str, object]] = [{"role": "user", "content": "task"}]
    for index in range(3):
        items.extend(
            [
                {
                    "type": "function_call",
                    "call_id": str(index),
                    "name": "extract",
                    "arguments": json.dumps({"kind": "list"}),
                },
                {
                    "type": "function_call_output",
                    "call_id": str(index),
                    "output": (
                        _json(
                            {
                                "ok": True,
                                "evidence_id": f"ev-{index:05d}",
                                "data": [f"row-{index}", "detail" * 10_000],
                            },
                            limit=1_000,
                        )
                        if index == 0
                        else json.dumps(
                            {
                                "ok": True,
                                "evidence_id": f"ev-{index:05d}",
                                "data": [f"row-{index}", "detail" * 500],
                            }
                        )
                    ),
                },
            ]
        )
    model_data = ModelInputData(input=items, instructions="instructions")  # type: ignore[arg-type]
    filtered = BoundedToolOutputFilter(old_output_chars=120)(
        CallModelData(model_data=model_data, agent=SimpleNamespace(), context=None)  # type: ignore[arg-type]
    )
    outputs = [
        str(item["output"])
        for item in filtered.input
        if isinstance(item, dict) and item.get("type") == "function_call_output"
    ]
    assert "compacted_tool_output" in outputs[0]
    assert "row-0" in outputs[0]
    assert "ev-00000" in outputs[0]
    assert "compacted_tool_output" not in outputs[-1]


def test_large_dom_projection_is_bounded_relevant_and_bid_preserving() -> None:
    elements = [
        {
            "bid": f"p{index}",
            "tag": "p",
            "text": "generic body " + ("x" * 300),
            "visible": True,
            "rect": [0, index * 20, 500, 20],
        }
        for index in range(1_000)
    ]
    elements.append(
        {
            "bid": "target42",
            "tag": "button",
            "text": "下载关键指标",
            "visible": True,
            "rect": [0, 0, 120, 30],
            "disabled": False,
            "shadow": False,
        }
    )
    result = {
        "url": "https://example.test",
        "title": "large",
        "dom_hash": "abc",
        "evidence_id": "ev-00001",
        "elements": elements,
        "truncated": False,
    }
    projected = _project_observation(result, "下载关键指标", unchanged=False)
    encoded = json.dumps(projected, ensure_ascii=False)
    assert len(encoded) <= 20_500
    assert projected["total_element_count"] == 1_001
    assert projected["elements_truncated_for_model"] is True
    assert any(item["bid"] == "target42" for item in projected["elements"])
    assert all("shadow" not in item for item in projected["elements"] if item["bid"] != "target42")

    repeated = _project_observation(result, "下载关键指标", unchanged=True)
    assert len(repeated["elements"]) <= 24
    assert len(json.dumps(repeated, ensure_ascii=False)) <= 6_500
    assert repeated["bids_remain_valid"] is True
    assert any(item["bid"] == "target42" for item in repeated["elements"])


def test_relevant_body_flood_cannot_hide_visible_navigation_control() -> None:
    elements = [
        {"bid": f"p{index}", "tag": "p", "text": "annual report", "visible": True}
        for index in range(200)
    ]
    elements.append(
        {"bid": "next", "tag": "button", "text": "Next", "visible": True, "rect": [0, 0, 80, 30]}
    )
    projected = _project_observation(
        {
            "url": "https://example.test",
            "title": "reports",
            "dom_hash": "a",
            "evidence_id": "ev-1",
            "elements": elements,
            "truncated": False,
        },
        "annual report",
        unchanged=False,
    )
    assert any(item["bid"] == "next" for item in projected["elements"])


def test_large_select_does_not_empty_repeated_observation() -> None:
    options = [
        {"value": "v" * 300, "label": "label" * 100, "selected": False}
        for _ in range(40)
    ]
    result = {
        "url": "https://example.test",
        "title": "select",
        "dom_hash": "a",
        "evidence_id": "ev-1",
        "elements": [
            {
                "bid": "select1",
                "tag": "select",
                "label": "critical choice",
                "visible": True,
                "options": options,
            },
            {"bid": "button1", "tag": "button", "text": "Continue", "visible": True},
        ],
        "truncated": False,
    }
    projected = _project_observation(result, "critical choice", unchanged=True)
    assert {item["bid"] for item in projected["elements"]} == {"select1", "button1"}
    assert len(json.dumps(projected, ensure_ascii=False)) <= 6_500


def test_repeated_observation_filter_keeps_last_changed_bid_catalog() -> None:
    elements = [{"bid": f"b{index}", "tag": "button", "text": f"Action {index}"} for index in range(120)]
    first = json.dumps({"ok": True, "unchanged": False, "elements": elements})
    repeated = json.dumps({"ok": True, "unchanged": True, "elements": elements[:24]})
    items: list[dict[str, object]] = [{"role": "user", "content": "task"}]
    for call_id, output in (("1", first), ("2", repeated)):
        items.extend(
            [
                {"type": "function_call", "call_id": call_id, "name": "observe", "arguments": "{}"},
                {"type": "function_call_output", "call_id": call_id, "output": output},
            ]
        )
    filtered = BoundedToolOutputFilter()(
        CallModelData(
            model_data=ModelInputData(input=items, instructions="instructions"),  # type: ignore[arg-type]
            agent=SimpleNamespace(),
            context=None,
        )
    )
    outputs = "\n".join(
        str(item["output"])
        for item in filtered.input
        if isinstance(item, dict) and item.get("type") == "function_call_output"
    )
    assert "b119" in outputs


@pytest.mark.asyncio
async def test_no_progress_loop_stops_before_another_model_request() -> None:
    context = make_context()
    assert context.note_page_state("https://example.test", "same") is False
    for _ in range(context.no_progress_limit):
        assert context.note_page_state("https://example.test", "same") is True
    assert context.loop_detected is True
    with pytest.raises(NoProgressLoopError, match="NO_PROGRESS_LOOP"):
        await ProtocolRunHooks().on_llm_start(
            RunContextWrapper(context),
            SimpleNamespace(),  # type: ignore[arg-type]
            "system",
            [],
        )


def test_new_page_states_and_scroll_progress_reset_loop_counter() -> None:
    context = make_context()
    for index in range(5):
        context.note_page_state("https://example.test", f"page-{index}")
        assert context.no_progress_streak == 0
    context.note_page_state("https://example.test", "page-4")
    assert context.no_progress_streak == 1
    context.record_receipt(
        {
            "action_id": "act-1",
            "action": "scroll",
            "success": True,
            "before_url": "https://example.test",
            "after_url": "https://example.test",
            "before_dom_hash": "page-4",
            "after_dom_hash": "page-4",
            "postconditions": {"before": 0, "after": 200},
            "evidence_ids": [],
        }
    )
    assert context.no_progress_streak == 0
    assert context.loop_detected is False


def test_repeated_same_value_actions_do_not_bypass_loop_guard() -> None:
    context = make_context()
    context.note_page_state("https://example.test", "same")
    for index in range(context.no_progress_limit):
        context.record_receipt(
            {
                "action_id": f"act-{index}",
                "action": "fill",
                "success": True,
                "before_url": "https://example.test",
                "after_url": "https://example.test",
                "before_dom_hash": "same",
                "after_dom_hash": "same",
                "postconditions": {"value_changed": False},
                "evidence_ids": [],
            }
        )
    assert context.loop_detected is True


def test_shared_tpm_limiter_reservation_is_atomic_across_instances() -> None:
    events: list[tuple[float, int]] = []
    lock = threading.RLock()
    first = SharedTPMLimiter(events, lock, token_budget=100, clock=lambda: 10.0)
    second = SharedTPMLimiter(events, lock, token_budget=100, clock=lambda: 10.0)
    barrier = threading.Barrier(2)
    results: list[tuple[bool, float, int]] = []

    def reserve(limiter: SharedTPMLimiter) -> None:
        barrier.wait()
        results.append(limiter.try_acquire(60))

    threads = [threading.Thread(target=reserve, args=(limiter,)) for limiter in (first, second)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert sum(granted for granted, _, _ in results) == 1
    assert len(events) == 1
    assert next(delay for granted, delay, _ in results if not granted) > 0
    with pytest.raises(ValueError, match="阻止发送"):
        first.try_acquire(101)


def test_shared_tpm_limiter_is_atomic_across_spawned_processes() -> None:
    context = mp.get_context("spawn")
    manager = context.Manager()
    processes: list[mp.Process] = []
    try:
        events = manager.list()
        lock = manager.RLock()
        gate = context.Event()
        ready = context.Queue()
        results = context.Queue()
        processes = [
            context.Process(target=_reserve_from_process, args=(events, lock, gate, ready, results))
            for _ in range(2)
        ]
        for process in processes:
            process.start()
        assert ready.get(timeout=5) is True
        assert ready.get(timeout=5) is True
        gate.set()
        reservations = [results.get(timeout=5), results.get(timeout=5)]
        for process in processes:
            process.join(timeout=5)
        assert all(process.exitcode == 0 for process in processes)
        assert sum(granted for granted, _, _ in reservations) == 1
        assert len(events) == 1
    finally:
        for process in processes:
            if process.is_alive():
                process.terminate()
                process.join(timeout=5)
        manager.shutdown()


def test_usage_stats_handles_available_and_missing_usage_without_sensitive_data() -> None:
    stats = TaskUsageStats(2, 7, "task-private")
    stats.record(
        estimated_input_tokens=100,
        wait_seconds=1.25,
        throttle_reason="pre_send_tpm_capacity",
        usage=Usage(requests=1, input_tokens=40, output_tokens=5, total_tokens=45),
    )
    stats.record(
        estimated_input_tokens=120,
        wait_seconds=0,
        throttle_reason=None,
        usage=None,
        error_type="RateLimitError",
    )
    value = stats.to_dict()
    assert value["request_count"] == 2
    assert value["input_tokens"] == 40
    assert value["usage_unavailable_count"] == 1
    assert value["throttle_reasons"] == {"pre_send_tpm_capacity": 1}
    assert "private" not in json.dumps(value["requests"])


def test_token_estimate_is_conservative_for_unicode_and_tool_schema() -> None:
    estimate = estimate_input_tokens("中文系统", [{"role": "user", "content": "数据" * 100}], TOOLS)
    raw_bytes = len(("中文系统" + "数据" * 100).encode("utf-8"))
    assert estimate > raw_bytes


def test_json_limit_is_hard_even_when_preview_needs_escaping() -> None:
    output = _json(
        {"ok": True, "evidence_id": "ev-00001", "data": ('"\\中文' * 20_000)},
        limit=1_000,
    )
    assert len(output) <= 1_000
    assert json.loads(output)["evidence_id"] == "ev-00001"


@pytest.mark.asyncio
async def test_tpm_wait_expires_reservation_without_api_retry() -> None:
    now = [0.0]
    events: list[tuple[float, int]] = [(0.0, 80)]

    async def advance(delay: float) -> None:
        now[0] += delay

    limiter = SharedTPMLimiter(
        events,
        threading.RLock(),
        token_budget=100,
        window_seconds=2.0,
        clock=lambda: now[0],
        sleeper=advance,
    )
    reservation = await limiter.acquire(30)
    assert reservation["reason"] == "pre_send_tpm_capacity"
    assert reservation["wait_seconds"] >= 2.0
    assert events == [(now[0], 30)]


@pytest.mark.asyncio
async def test_oversized_request_is_observed_and_blocked_before_model_call() -> None:
    limiter = SharedTPMLimiter([], threading.RLock(), token_budget=100)
    model = ThrottledModel(SimpleNamespace(), limiter)  # type: ignore[arg-type]
    stats = TaskUsageStats(0, 1, "x")
    model.usage_stats = stats
    with pytest.raises(ValueError, match="阻止发送"):
        await model.get_response(
            "system",
            [],
            None,
            [],
            None,
            [],
            None,
            previous_response_id=None,
            conversation_id=None,
            prompt=None,
        )
    summary = stats.to_dict()
    assert summary["usage_unavailable_count"] == 1
    assert summary["throttle_reasons"] == {"pre_send_request_exceeds_tpm_budget": 1}


@pytest.mark.asyncio
async def test_limiter_proxy_failure_is_observed_before_model_call() -> None:
    class BrokenLimiter:
        async def acquire(self, estimated_tokens: int) -> dict[str, object]:
            assert estimated_tokens > 0
            raise BrokenPipeError("manager unavailable")

    model = ThrottledModel(SimpleNamespace(), BrokenLimiter())  # type: ignore[arg-type]
    stats = TaskUsageStats(0, 1, "x")
    model.usage_stats = stats
    with pytest.raises(BrokenPipeError):
        await model.get_response(
            "system",
            [],
            None,
            [],
            None,
            [],
            None,
            previous_response_id=None,
            conversation_id=None,
            prompt=None,
        )
    assert stats.to_dict()["throttle_reasons"] == {"pre_send_tpm_limiter_error": 1}


def test_unavailable_shared_lock_fails_closed_with_bounded_error() -> None:
    class UnavailableLock:
        def acquire(self, timeout: float) -> bool:
            assert timeout == 5.0
            return False

        def release(self) -> None:
            raise AssertionError("unacquired lock must not be released")

    limiter = SharedTPMLimiter([], UnavailableLock(), token_budget=100)  # type: ignore[arg-type]
    with pytest.raises(RuntimeError, match="TPM_LIMITER_LOCK_TIMEOUT"):
        limiter.try_acquire(10)


def test_tpm_reservation_timestamp_is_taken_after_lock_wait() -> None:
    now = [0.0]

    class DelayedLock:
        calls = 0

        def acquire(self, timeout: float) -> bool:
            assert timeout == 5.0
            if self.calls == 0:
                now[0] += 5.0
            self.calls += 1
            return True

        def release(self) -> None:
            return None

    events: list[tuple[float, int]] = []
    limiter = SharedTPMLimiter(
        events,
        DelayedLock(),  # type: ignore[arg-type]
        token_budget=100,
        window_seconds=60,
        clock=lambda: now[0],
    )
    assert limiter.try_acquire(60)[0] is True
    assert events == [(5.0, 60)]
    now[0] = 60.0
    assert limiter.try_acquire(60)[0] is False
    now[0] = 65.01
    assert limiter.try_acquire(60)[0] is True


@pytest.mark.asyncio
async def test_cancelled_model_request_records_unavailable_usage() -> None:
    class CancelledModel:
        async def get_response(self, *args, **kwargs):  # noqa: ANN002, ANN003
            raise asyncio.CancelledError

    model = ThrottledModel(CancelledModel(), None)  # type: ignore[arg-type]
    stats = TaskUsageStats(0, 1, "x")
    model.usage_stats = stats
    with pytest.raises(asyncio.CancelledError):
        await model.get_response(
            "system",
            [],
            None,
            [],
            None,
            [],
            None,
            previous_response_id=None,
            conversation_id=None,
            prompt=None,
        )
    assert stats.to_dict()["usage_unavailable_count"] == 1
    assert stats.requests[0]["error_type"] == "CancelledError"


@pytest.mark.asyncio
async def test_cancelled_vision_request_records_unavailable_usage() -> None:
    async def cancel(**kwargs):  # noqa: ANN003
        raise asyncio.CancelledError

    context = make_context()
    context.usage_stats = TaskUsageStats(0, 1, "x")
    context.vision_client = SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=cancel))
    )  # type: ignore[assignment]
    with pytest.raises(asyncio.CancelledError):
        await _vision_completion(context, [{"role": "user", "content": "local"}])
    assert context.usage_stats.requests[0]["channel"] == "vision"
    assert context.usage_stats.requests[0]["error_type"] == "CancelledError"


@pytest.mark.asyncio
async def test_recall_evidence_reads_tail_without_refetching_page() -> None:
    context = make_context()
    evidence = context.evidence_store.add(
        "dom",
        "https://example.test",
        "rows",
        {"kind": "list", "data": [f"row-{index:04d}" for index in range(100)]},
    )
    tool = next(item for item in TOOLS if item.name == "recall_evidence")
    arguments = json.dumps({"evidence_id": evidence.evidence_id, "offset": 90, "limit": 10})
    output = await tool.on_invoke_tool(
        ToolContext(
            context,
            tool_name="recall_evidence",
            tool_call_id="recall",
            tool_arguments=arguments,
        ),
        arguments,
    )
    payload = json.loads(output)
    assert payload["content"][-1] == "row-0099"
    assert payload["next_offset"] is None
    assert payload["evidence_id"] == evidence.evidence_id


@pytest.mark.asyncio
async def test_recall_evidence_pages_through_single_oversized_record() -> None:
    context = make_context()
    evidence = context.evidence_store.add(
        "network",
        "https://example.test",
        "large record",
        {"records": [{"body": "x" * 40_000 + "TAIL_SENTINEL"}]},
    )
    tool = next(item for item in TOOLS if item.name == "recall_evidence")
    offset = 0
    chunks: list[str] = []
    while True:
        arguments = json.dumps(
            {"evidence_id": evidence.evidence_id, "offset": offset, "limit": 12_000}
        )
        output = await tool.on_invoke_tool(
            ToolContext(
                context,
                tool_name="recall_evidence",
                tool_call_id=f"recall-{offset}",
                tool_arguments=arguments,
            ),
            arguments,
        )
        payload = json.loads(output)
        assert payload["unit"] == "char"
        assert payload["encoding"] == "json"
        assert payload.get("truncated_for_model") is not True
        chunks.append(payload["content"])
        if payload["next_offset"] is None:
            break
        offset = payload["next_offset"]

    recalled = "".join(chunks)
    assert json.loads(recalled)[0]["body"].endswith("TAIL_SENTINEL")


@pytest.mark.asyncio
async def test_only_accepted_finish_stops_tool_loop() -> None:
    context = make_context()
    wrapper = RunContextWrapper(context)
    finish_tool = next(tool for tool in TOOLS if tool.name == "finish")
    rejected = FunctionToolResult(
        tool=finish_tool,
        output=json.dumps({"accepted": False}),
        run_item=None,
    )
    decision = await _verified_finish_behavior(wrapper, [rejected])
    assert decision.is_final_output is False
    context.finish_accepted = True
    accepted = FunctionToolResult(
        tool=finish_tool,
        output=json.dumps({"accepted": True}),
        run_item=None,
    )
    decision = await _verified_finish_behavior(wrapper, [accepted])
    assert decision.is_final_output is True


@pytest.mark.asyncio
async def test_coverage_certificate_must_be_runtime_signed_and_untampered() -> None:
    context = make_context()
    context.contract = TaskContract.from_item(
        {"task_idx": 1, "task_id": "x", "website": "https://example.test", "task": "列出所有记录"}
    )
    rows = context.evidence_store.add("dom", "https://example.test", "rows", {"data": ["A", "B", "B"]})
    terminal = context.evidence_store.add("dom", "https://example.test", "next disabled", {"disabled": True})
    context.visited_urls.append("https://example.test")
    coverage_tool = next(tool for tool in TOOLS if tool.name == "record_coverage")
    coverage_args = {
        "strategy": "pagination",
        "item_evidence_ids": [rows.evidence_id],
        "pages_visited": 2,
        "expected_total": 2,
        "terminal_reason": "next_disabled",
        "terminal_evidence_id": terminal.evidence_id,
    }
    coverage_output = await coverage_tool.on_invoke_tool(
        ToolContext(context, tool_name="record_coverage", tool_call_id="1", tool_arguments=json.dumps(coverage_args)),
        json.dumps(coverage_args),
    )
    certificate = json.loads(coverage_output)["certificate"]
    assert certificate["unique_item_count"] == 2
    assert certificate["duplicate_item_count"] == 1

    finish_tool = next(tool for tool in TOOLS if tool.name == "finish")
    finish_args = {
        "answer": "A, B",
        "evidence_ids": [rows.evidence_id],
        "evidence_bindings": [{"path": "$", "evidence_ids": [rows.evidence_id]}],
        "coverage": certificate,
    }
    tampered = {**finish_args, "coverage": {**certificate, "unique_item_count": 999}}
    tampered_output = await finish_tool.on_invoke_tool(
        ToolContext(context, tool_name="finish", tool_call_id="2", tool_arguments=json.dumps(tampered)),
        json.dumps(tampered),
    )
    assert json.loads(tampered_output)["accepted"] is False
    accepted_output = await finish_tool.on_invoke_tool(
        ToolContext(context, tool_name="finish", tool_call_id="3", tool_arguments=json.dumps(finish_args)),
        json.dumps(finish_args),
    )
    assert json.loads(accepted_output)["accepted"] is True
    assert context.finish_accepted is True
    assert len(context.final_evidence_ids) == 2
    assert context.coverage_evidence_ids[certificate["item_fingerprint"]] in context.final_evidence_ids


@pytest.mark.asyncio
async def test_illegal_finish_tool_output_is_returned_as_error_not_success() -> None:
    context = make_context()
    finish_tool = next(tool for tool in TOOLS if tool.name == "finish")
    illegal = json.dumps({"answer": "unsupported", "unexpected": "field"})
    output = await finish_tool.on_invoke_tool(
        ToolContext(context, tool_name="finish", tool_call_id="bad", tool_arguments=illegal),
        illegal,
    )
    assert "error" in str(output).lower()
    assert context.finish_accepted is False


@pytest.mark.asyncio
async def test_max_turns_never_becomes_success(monkeypatch) -> None:
    async def exceed(*args, **kwargs):  # noqa: ANN002, ANN003
        raise MaxTurnsExceeded("limit")

    monkeypatch.setattr("web_agent.runtime.Runner.run", exceed)
    agent = ProtocolIIIAgent("test", "http://127.0.0.1:9/v1", "key")
    result = await agent.run(DummyActor(), make_context().contract, EvidenceStore())  # type: ignore[arg-type]
    assert result["status"] == "FAIL_MAX_STEPS"
    assert result["agent_answer"] is None


def test_default_main_does_not_import_uitars() -> None:
    source = open("src/agent/main.py", encoding="utf-8").read()
    assert "UITARSAgent" not in source
    assert "from agent import" not in source
