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

from web_agent.browser_actor import ActorCallDeadlineExceeded
from web_agent.contracts import CoverageCertificate, TaskContract
from web_agent.evidence import EvidenceStore
from web_agent.runtime import (
    TOOLS,
    BoundedToolOutputFilter,
    NoProgressLoopError,
    ProtocolRunHooks,
    ProtocolIIIAgent,
    TaskRuntimeContext,
    TerminalBrowserError,
    _coverage_items,
    _answer_shape_reasons,
    _error,
    _json,
    _model_settings,
    _project_observation,
    _tool_failure,
    _verified_finish_behavior,
    _vision_completion,
)
from web_agent.token_control import (
    MAX_SERIALIZED_CONTEXT_BYTES,
    SharedTPMLimiter,
    TaskUsageStats,
    ThrottledModel,
    estimate_input_tokens,
    model_input_metrics,
)
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
        "extract_many",
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
    agent = ProtocolIIIAgent("kimi-k2.6", "http://127.0.0.1:9/v1", "test")
    assert agent.client.max_retries == 0


def test_hard_step_limit_rejects_call_61() -> None:
    context = make_context(100)
    for index in range(60):
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
    assert len(filtered.input) == 2
    assert not any(isinstance(item, dict) and item.get("type") == "function_call" for item in filtered.input)
    checkpoint = str(filtered.input[-1])
    assert "WORKING_MEMORY_CHECKPOINT" in checkpoint
    assert "ev-00009" in checkpoint
    assert len(json.dumps(filtered.input, ensure_ascii=False).encode("utf-8")) <= 72_000


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
    checkpoint = str(filtered.input[-1])
    assert "compacted_tool_output" in checkpoint
    assert "row-0" in checkpoint
    assert "ev-00000" in checkpoint
    assert not any(isinstance(item, dict) and item.get("type") == "function_call" for item in filtered.input)


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
    encoded = json.dumps(projected, ensure_ascii=False, separators=(",", ":"))
    assert len(encoded.encode("utf-8")) <= 20_000
    assert projected["total_element_count"] == 1_001
    assert projected["elements_truncated_for_model"] is True
    assert any(item["bid"] == "target42" for item in projected["elements"])
    assert all("shadow" not in item for item in projected["elements"] if item["bid"] != "target42")

    repeated = _project_observation(result, "下载关键指标", unchanged=True)
    assert len(repeated["elements"]) <= 24
    repeated_bytes = json.dumps(repeated, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    assert len(repeated_bytes) <= 6_000
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


def test_projection_selects_by_priority_then_restores_dom_order() -> None:
    elements = [
        {"bid": "first", "tag": "p", "text": "intro", "visible": True},
        {"bid": "target", "tag": "button", "text": "Download annual report", "visible": True},
        {"bid": "last", "tag": "button", "text": "Continue", "visible": True},
    ]
    projected = _project_observation(
        {
            "url": "https://example.test",
            "title": "reports",
            "dom_hash": "a",
            "evidence_id": "ev-1",
            "elements": elements,
            "truncated": False,
        },
        "download annual report",
        unchanged=False,
    )
    assert [item["bid"] for item in projected["elements"]] == ["first", "target", "last"]


def test_new_and_changed_elements_are_prioritized_with_bounded_context_and_scroll() -> None:
    elements = [
        {"bid": f"row{index}", "tag": "p", "text": "generic", "visible": True}
        for index in range(200)
    ]
    elements.extend(
        [
            {
                "bid": "new-option",
                "tag": "button",
                "role": "option",
                "text": "Late choice",
                "visible": True,
                "new": True,
                "context": ["listbox:Result choices", "section:Filters", "ignored"],
            },
            {
                "bid": "spa-result",
                "tag": "p",
                "text": "Loaded later",
                "visible": True,
                "changed": True,
            },
        ]
    )
    result = {
        "url": "https://example.test",
        "title": "dynamic",
        "dom_hash": "a",
        "evidence_id": "ev-1",
        "elements": elements,
        "scroll": [
            {
                "kind": "page",
                "frame": 0,
                "frame_url": "https://example.test",
                "top": True,
                "bottom": False,
                "position": 0,
                "remaining": 900,
                "viewport": 700,
                "extent": 1600,
            },
            {
                "kind": "container",
                "frame": 0,
                "bid": "list",
                "role": "listbox",
                "label": "Result choices",
                "top": False,
                "bottom": True,
                "position": 400,
                "remaining": 0,
                "viewport": 100,
                "extent": 500,
            },
        ],
        "truncated": False,
    }
    projected = _project_observation(result, "unrelated task", unchanged=False, max_elements=12)
    by_bid = {item["bid"]: item for item in projected["elements"]}
    assert {"new-option", "spa-result"} <= set(by_bid)
    assert by_bid["new-option"]["context"] == ["listbox:Result choices", "section:Filters"]
    assert projected["scroll"][0]["top"] is True
    assert projected["scroll"][1]["bottom"] is True
    assert len(json.dumps(projected, ensure_ascii=False, separators=(",", ":")).encode("utf-8")) <= 20_000


def test_scroll_metadata_cannot_break_repeated_projection_budget_or_hide_button() -> None:
    scroll = [
        {
            "kind": "page",
            "frame": 0,
            "frame_url": "https://example.test/" + ("路径" * 200),
            "top": True,
            "bottom": False,
            "position": 0,
            "remaining": 5_000,
            "viewport": 700,
            "extent": 5_700,
        }
    ]
    scroll.extend(
        {
            "kind": "container",
            "frame": 0,
            "frame_url": "https://example.test/" + ("路径" * 200),
            "bid": f"scroll{index}",
            "role": "listbox",
            "label": "选项" * 120,
            "visible": True,
            "top": True,
            "bottom": False,
            "position": 0,
            "remaining": 900,
            "viewport": 100,
            "extent": 1_000,
        }
        for index in range(20)
    )
    projected = _project_observation(
        {
            "url": "https://example.test/" + ("查询" * 20_000),
            "title": "标题" * 20_000,
            "dom_hash": "a",
            "evidence_id": "ev-1",
            "elements": [
                {"bid": "go", "tag": "button", "text": "Continue", "visible": True},
                *[
                    {
                        "frame": index + 1,
                        "frame_error": "错误" * 200,
                        "url": "https://example.test/" + ("路径" * 2_000),
                    }
                    for index in range(5)
                ],
            ],
            "scroll": scroll,
            "truncated": False,
        },
        "continue",
        unchanged=True,
    )
    encoded = json.dumps(projected, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    assert len(encoded) <= 6_000
    assert projected["elements"][0]["bid"] == "go"
    assert projected["scroll"][0]["kind"] == "page"
    assert projected["scroll_truncated_for_model"] is True

    small = _project_observation(
        {
            "url": "https://example.test/" + ("查询" * 20_000),
            "title": "标题" * 20_000,
            "dom_hash": "a",
            "evidence_id": "ev-1",
            "elements": [
                {"bid": "go", "tag": "button", "text": "Continue", "visible": True},
                *[
                    {
                        "frame": index + 1,
                        "frame_error": "错误" * 200,
                        "url": "https://example.test/" + ("路径" * 2_000),
                    }
                    for index in range(5)
                ],
            ],
            "scroll": scroll,
            "truncated": False,
        },
        "continue",
        unchanged=False,
        max_chars=1_200,
    )
    assert len(json.dumps(small, ensure_ascii=False, separators=(",", ":")).encode("utf-8")) <= 1_200
    assert small["elements"][0]["bid"] == "go"
    assert small["frame_errors_truncated_for_model"] is True


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


def test_select_without_option_budget_keeps_truncation_marker() -> None:
    options = [{"value": "v" * 300, "label": "选项" * 100, "selected": False} for _ in range(40)]
    result = {
        "url": "https://example.test",
        "title": "selects",
        "dom_hash": "a",
        "evidence_id": "ev-1",
        "elements": [
            {"bid": "select1", "tag": "select", "visible": True, "options": options},
            {"bid": "select2", "tag": "select", "visible": True, "options": options},
            *[
                {"bid": f"button{index}", "tag": "button", "text": f"Action {index}", "visible": True}
                for index in range(5)
            ],
        ],
        "truncated": False,
    }
    projected = _project_observation(result, "action", unchanged=False, max_chars=1_200)
    selects = [item for item in projected["elements"] if item["tag"] == "select"]
    assert len(selects) == 2
    assert all(item.get("options") or item.get("options_truncated") is True for item in selects)
    assert len(json.dumps(projected, ensure_ascii=False, separators=(",", ":")).encode("utf-8")) <= 1_200


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
    assert "b119" in str(filtered.input[-1])


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


@pytest.mark.asyncio
async def test_poisoned_actor_stops_before_another_model_request() -> None:
    context = make_context()
    context.actor = SimpleNamespace(  # type: ignore[assignment]
        poisoned=True,
        poisoned_reason="ACTOR_POISONED: click 终态不确定",
    )
    with pytest.raises(TerminalBrowserError, match="ACTOR_POISONED"):
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
        assert ready.get(timeout=15) is True
        assert ready.get(timeout=15) is True
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
    assert [(event[0], event[1]) for event in events] == [(now[0], 30)]


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
    assert [(event[0], event[1]) for event in events] == [(5.0, 60)]
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
    coverage_result = json.loads(coverage_output)
    coverage_id = coverage_result["coverage_id"]
    assert coverage_result["unique_item_count"] == 2
    assert coverage_result["duplicate_item_count"] == 1

    finish_tool = next(tool for tool in TOOLS if tool.name == "finish")
    finish_args = {
        "answer": "A, B",
        "evidence_ids": [rows.evidence_id],
        "evidence_bindings": [{"path": "$", "evidence_ids": [rows.evidence_id]}],
        "coverage_id": coverage_id,
    }
    unknown = {**finish_args, "coverage_id": "cov-forged"}
    tampered_output = await finish_tool.on_invoke_tool(
        ToolContext(context, tool_name="finish", tool_call_id="2", tool_arguments=json.dumps(unknown)),
        json.dumps(unknown),
    )
    assert json.loads(tampered_output)["accepted"] is False
    accepted_output = await finish_tool.on_invoke_tool(
        ToolContext(context, tool_name="finish", tool_call_id="3", tool_arguments=json.dumps(finish_args)),
        json.dumps(finish_args),
    )
    assert json.loads(accepted_output)["accepted"] is True
    assert context.finish_accepted is True
    assert len(context.final_evidence_ids) == 2
    assert context.coverage_evidence_ids[coverage_id] in context.final_evidence_ids


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
async def test_finish_rejects_when_audit_timeout_poisons_browser() -> None:
    class PoisonOnAudit(DummyActor):
        poisoned = False
        poisoned_reason = None

        async def audit_step(self, label: str) -> str:
            self.poisoned = True
            self.poisoned_reason = "ACTOR_POISONED: finish 终态不确定"
            raise ActorCallDeadlineExceeded(
                label,
                dispatched=True,
                task_generation=3,
                attempt=7,
            )

    context = make_context()
    context.actor = PoisonOnAudit()  # type: ignore[assignment]
    evidence = context.evidence_store.add("dom", "https://example.test", "answer", {"data": "42"})
    context.visited_urls.append("https://example.test")
    finish_tool = next(tool for tool in TOOLS if tool.name == "finish")
    arguments = json.dumps({"answer": "42", "evidence_ids": [evidence.evidence_id]})

    output = await finish_tool.on_invoke_tool(
        ToolContext(context, tool_name="finish", tool_call_id="poisoned", tool_arguments=arguments),
        arguments,
    )
    payload = json.loads(output)

    assert payload["accepted"] is False
    assert payload["terminal_uncertain"] is True
    assert context.finish_accepted is False
    assert context.terminal_browser_error == "ACTOR_POISONED: finish 终态不确定"


@pytest.mark.asyncio
async def test_coverage_is_not_signed_after_audit_poisons_browser() -> None:
    class PoisonOnAudit(DummyActor):
        poisoned = False
        poisoned_reason = None

        async def audit_step(self, label: str) -> str:
            self.poisoned = True
            self.poisoned_reason = "ACTOR_POISONED: coverage 终态不确定"
            raise ActorCallDeadlineExceeded(label, dispatched=True)

    context = make_context()
    context.actor = PoisonOnAudit()  # type: ignore[assignment]
    context.contract = TaskContract.from_item(
        {"task_idx": 1, "task_id": "x", "website": "https://example.test", "task": "列出所有记录"}
    )
    rows = context.evidence_store.add("dom", "https://example.test", "rows", {"data": ["A"]})
    terminal = context.evidence_store.add("dom", "https://example.test", "terminal", {"disabled": True})
    arguments = json.dumps(
        {
            "strategy": "pagination",
            "item_evidence_ids": [rows.evidence_id],
            "pages_visited": 1,
            "expected_total": 1,
            "terminal_reason": "next_disabled",
            "terminal_evidence_id": terminal.evidence_id,
        }
    )
    coverage_tool = next(tool for tool in TOOLS if tool.name == "record_coverage")

    output = await coverage_tool.on_invoke_tool(
        ToolContext(context, tool_name="record_coverage", tool_call_id="poisoned", tool_arguments=arguments),
        arguments,
    )
    payload = json.loads(output)

    assert payload["ok"] is False
    assert payload["terminal_uncertain"] is True
    assert context.coverage_records == {}
    assert context.terminal_browser_error == "ACTOR_POISONED: coverage 终态不确定"


@pytest.mark.asyncio
async def test_poisoned_actor_cannot_reuse_cached_coverage_certificate() -> None:
    context = make_context()
    context.contract = TaskContract.from_item(
        {"task_idx": 1, "task_id": "x", "website": "https://example.test", "task": "列出所有记录"}
    )
    rows = context.evidence_store.add("dom", "https://example.test", "rows", {"data": ["A"]})
    terminal = context.evidence_store.add("dom", "https://example.test", "terminal", {"disabled": True})
    arguments = json.dumps(
        {
            "strategy": "pagination",
            "item_evidence_ids": [rows.evidence_id],
            "pages_visited": 1,
            "expected_total": 1,
            "terminal_reason": "next_disabled",
            "terminal_evidence_id": terminal.evidence_id,
        }
    )
    coverage_tool = next(tool for tool in TOOLS if tool.name == "record_coverage")
    first = await coverage_tool.on_invoke_tool(
        ToolContext(context, tool_name="record_coverage", tool_call_id="first", tool_arguments=arguments),
        arguments,
    )
    assert json.loads(first)["ok"] is True
    context.actor.poisoned = True
    context.actor.poisoned_reason = "ACTOR_POISONED: late failure"

    second = await coverage_tool.on_invoke_tool(
        ToolContext(context, tool_name="record_coverage", tool_call_id="second", tool_arguments=arguments),
        arguments,
    )
    payload = json.loads(second)

    assert payload["ok"] is False
    assert payload["cache_hit"] is False
    assert payload["terminal_uncertain"] is True
    assert payload["safe_to_retry"] is False


def test_browser_result_rebinds_runtime_action_to_actor_attempt() -> None:
    context = make_context()
    context.record_call("observe", {})
    context.record_tool_outcome(
        "observe",
        {},
        {"ok": True, "task_generation": 4, "attempt": 9},
    )
    assert context.actions[-1]["task_generation"] == 4
    assert context.actions[-1]["attempt"] == 9
    assert context.tool_outcomes[-1]["task_generation"] == 4
    assert context.tool_outcomes[-1]["attempt"] == 9


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


def test_state_action_cycles_period_one_to_four_are_stopped() -> None:
    for period in range(1, 5):
        context = make_context()
        context.no_progress_limit = 100
        for index in [*range(period), *range(period)]:
            arguments = {"slot": index}
            context.record_call("dialog", arguments)
            context.record_tool_outcome("dialog", arguments, {"ok": False, "error": "offline"})
        assert context.loop_detected is True
        assert f"周期 {period}" in str(context.loop_reason)


@pytest.mark.asyncio
async def test_identical_rejected_finish_stops_before_third_request() -> None:
    context = make_context()
    finish_tool = next(tool for tool in TOOLS if tool.name == "finish")
    arguments = json.dumps({"answer": "7", "evidence_ids": []})
    for index in range(2):
        output = await finish_tool.on_invoke_tool(
            ToolContext(context, tool_name="finish", tool_call_id=str(index), tool_arguments=arguments),
            arguments,
        )
        assert json.loads(output)["accepted"] is False
    assert context.tool_steps == 2
    assert context.loop_detected is True
    with pytest.raises(NoProgressLoopError, match="REPEATED_FINISH"):
        await ProtocolRunHooks().on_llm_start(
            RunContextWrapper(context), SimpleNamespace(), "system", []  # type: ignore[arg-type]
        )


@pytest.mark.asyncio
async def test_extract_cache_avoids_duplicate_collection_and_cache_exceptions() -> None:
    class CountingActor(DummyActor):
        def __init__(self) -> None:
            self.calls = 0
            self.fail_once = True

        async def extract(self, kind: str, bid: str | None, limit: int) -> dict[str, object]:
            self.calls += 1
            if self.fail_once:
                self.fail_once = False
                raise RuntimeError("offline capture failure")
            return {
                "data": ["A", "B"],
                "evidence_id": "ev-cached",
                "url": "https://example.test",
                "content_hash": "content-1",
                "semantic_page_fingerprint": "page-1",
            }

    context = make_context()
    actor = CountingActor()
    context.actor = actor  # type: ignore[assignment]
    context.current_semantic_fingerprint = "page-1"
    tool = next(item for item in TOOLS if item.name == "extract")
    arguments = json.dumps({"kind": "list", "bid": None, "limit": 100})
    failed = await tool.on_invoke_tool(
        ToolContext(context, tool_name="extract", tool_call_id="1", tool_arguments=arguments), arguments
    )
    assert json.loads(failed)["ok"] is False
    first = await tool.on_invoke_tool(
        ToolContext(context, tool_name="extract", tool_call_id="2", tool_arguments=arguments), arguments
    )
    second = await tool.on_invoke_tool(
        ToolContext(context, tool_name="extract", tool_call_id="3", tool_arguments=arguments), arguments
    )
    assert json.loads(first)["cache_hit"] is False
    assert json.loads(second)["cache_hit"] is True
    assert actor.calls == 2


@pytest.mark.asyncio
async def test_extract_cache_does_not_store_new_state_under_old_state_key() -> None:
    class DriftingActor(DummyActor):
        def __init__(self) -> None:
            self.calls = 0

        async def extract(self, kind: str, bid: str | None, limit: int) -> dict[str, object]:
            self.calls += 1
            return {
                "data": [f"state-b-call-{self.calls}"],
                "evidence_id": f"ev-{self.calls}",
                "content_hash": f"hash-{self.calls}",
                "semantic_page_fingerprint": "state-b",
            }

    context = make_context()
    actor = DriftingActor()
    context.actor = actor  # type: ignore[assignment]
    context.current_semantic_fingerprint = "state-a"
    tool = next(item for item in TOOLS if item.name == "extract")
    arguments = json.dumps({"kind": "list", "bid": None, "limit": 100})
    await tool.on_invoke_tool(
        ToolContext(context, tool_name="extract", tool_call_id="1", tool_arguments=arguments), arguments
    )
    context.current_semantic_fingerprint = "state-a"
    await tool.on_invoke_tool(
        ToolContext(context, tool_name="extract", tool_call_id="2", tool_arguments=arguments), arguments
    )
    assert actor.calls == 2


@pytest.mark.asyncio
async def test_extract_many_is_sequential_bounded_and_deduplicates_requests() -> None:
    class ManyActor(DummyActor):
        def __init__(self) -> None:
            self.batches: list[list[dict[str, object]]] = []

        async def extract_many(self, requests: list[dict[str, object]]) -> list[dict[str, object]]:
            self.batches.append(requests)
            return [
                {
                    "data": [request["kind"]],
                    "evidence_id": f"ev-{index}",
                    "content_hash": f"hash-{index}",
                    "semantic_page_fingerprint": "page-1",
                }
                for index, request in enumerate(requests)
            ]

    context = make_context()
    actor = ManyActor()
    context.actor = actor  # type: ignore[assignment]
    context.current_semantic_fingerprint = "page-1"
    tool = next(item for item in TOOLS if item.name == "extract_many")
    arguments = json.dumps(
        {
            "requests": [
                {"kind": "text", "bid": None, "limit": 100},
                {"kind": "text", "bid": None, "limit": 100},
                {"kind": "list", "bid": None, "limit": 100},
            ]
        }
    )
    output = await tool.on_invoke_tool(
        ToolContext(context, tool_name="extract_many", tool_call_id="many", tool_arguments=arguments), arguments
    )
    payload = json.loads(output)
    assert len(payload["results"]) == 3
    assert payload["cache_hit_count"] == 1
    assert len(actor.batches) == 1 and len(actor.batches[0]) == 2


@pytest.mark.asyncio
async def test_extract_many_rejects_mixed_semantic_states_without_publishing_cache() -> None:
    class DriftingManyActor(DummyActor):
        async def extract_many(self, requests: list[dict[str, object]]) -> list[dict[str, object]]:
            assert [request["kind"] for request in requests] == ["list"]
            return [
                {
                    "data": ["B"],
                    "evidence_id": "ev-b",
                    "content_hash": "hash-b",
                    "semantic_page_fingerprint": "state-b",
                }
            ]

    context = make_context()
    context.actor = DriftingManyActor()  # type: ignore[assignment]
    context.current_semantic_fingerprint = "state-a"
    text_request = {"kind": "text", "bid": None, "limit": 100}
    cached_key = TaskRuntimeContext._signature("extract", {"state": "state-a", **text_request})
    context.extract_cache[cached_key] = {
        "ok": True,
        **text_request,
        "data": ["A"],
        "evidence_id": "ev-a",
        "content_hash": "hash-a",
        "semantic_page_fingerprint": "state-a",
    }
    tool = next(item for item in TOOLS if item.name == "extract_many")
    arguments = json.dumps(
        {"requests": [text_request, {"kind": "list", "bid": None, "limit": 100}]}
    )
    output = await tool.on_invoke_tool(
        ToolContext(context, tool_name="extract_many", tool_call_id="mixed", tool_arguments=arguments),
        arguments,
    )
    payload = json.loads(output)
    assert payload["ok"] is False
    assert payload["semantic_page_fingerprints"] == ["state-a", "state-b"]
    assert list(context.extract_cache) == [cached_key]


@pytest.mark.asyncio
async def test_repeated_recall_chunk_is_no_progress_and_stops() -> None:
    context = make_context()
    evidence = context.evidence_store.add("dom", "https://example.test", "rows", {"data": ["A"]})
    tool = next(item for item in TOOLS if item.name == "recall_evidence")
    arguments = json.dumps({"evidence_id": evidence.evidence_id, "offset": 0, "limit": 100})
    outputs = []
    for index in range(3):
        outputs.append(
            json.loads(
                await tool.on_invoke_tool(
                    ToolContext(context, tool_name="recall_evidence", tool_call_id=str(index), tool_arguments=arguments),
                    arguments,
                )
            )
        )
    assert outputs[0]["progressed"] is True
    assert outputs[-1]["progressed"] is False
    assert context.loop_detected is True


@pytest.mark.asyncio
async def test_sec_style_same_tabs_new_url_loop_stops_under_twenty_steps() -> None:
    class ReusingActor(DummyActor):
        async def tabs(self, action: str, index: int | None, url: str | None) -> dict[str, object]:
            return {
                "tabs": [{"index": 0, "url": "https://example.test/report", "active": True}],
                "reused": True,
                "semantic_page_fingerprint": "blocked",
            }

    context = make_context()
    context.actor = ReusingActor()  # type: ignore[assignment]
    context.current_url = "https://example.test/report"
    context.current_semantic_fingerprint = "blocked"
    context.seen_urls.add("https://example.test/report")
    context.seen_page_states.add(("https://example.test/report", "blocked"))
    tool = next(item for item in TOOLS if item.name == "tabs")
    arguments = json.dumps(
        {"action": "new", "index": None, "url": "https://example.test/report"}
    )
    while not context.loop_detected:
        await tool.on_invoke_tool(
            ToolContext(context, tool_name="tabs", tool_call_id=str(context.tool_steps), tool_arguments=arguments),
            arguments,
        )
    assert context.tool_steps < 20


def test_real_spa_pagination_scroll_and_virtual_list_progress_are_not_killed() -> None:
    context = make_context()
    for index in range(12):
        arguments = {"bid": "next" if index < 4 else "virtual", "index": index}
        context.record_call("click" if index < 4 else "scroll", arguments)
        result = {
            "ok": True,
            "url": "https://example.test/collection",
            "semantic_page_fingerprint": f"semantic-{index}",
            "postconditions": {"before": index * 100, "after": (index + 1) * 100},
        }
        context.record_tool_outcome("click" if index < 4 else "scroll", arguments, result)
    assert context.loop_detected is False
    assert context.no_progress_streak == 0
    assert context.progress_credit == 12


def test_adaptive_budget_is_base_thirty_plus_progress_with_absolute_sixty() -> None:
    stalled = make_context()
    stalled.no_progress_limit = 100
    for index in range(30):
        arguments = {"index": index}
        stalled.record_call("dialog", arguments)
        stalled.record_tool_outcome("dialog", arguments, {"ok": False})
    stalled.assert_model_budget()
    assert stalled.loop_detected is True
    assert stalled.adaptive_step_budget == 30

    progressing = make_context()
    progressing.no_progress_limit = 100
    for index in range(30):
        arguments = {"index": index}
        progressing.record_call("observe", arguments)
        progressing.record_tool_outcome(
            "observe",
            arguments,
            {
                "url": "https://example.test",
                "semantic_page_fingerprint": f"state-{index}",
            },
        )
    assert progressing.adaptive_step_budget == 60
    for index in range(30, 60):
        progressing.record_call("dialog", {"index": index})
    with pytest.raises(RuntimeError, match="STEP_LIMIT"):
        progressing.record_call("dialog", {"index": 60})


def test_context_checkpoint_removes_call_arguments_and_has_hard_bound() -> None:
    context = make_context()
    context.current_url = "https://example.test/current"
    context.current_semantic_fingerprint = "semantic-current"
    context.latest_elements = {
        f"b{index}": {"bid": f"b{index}", "tag": "button", "text": "x" * 500}
        for index in range(200)
    }
    for index in range(100):
        context.evidence_store.add(
            "dom", "https://example.test", f"evidence {index}", {"data": "x" * 2_000}
        )
    items: list[dict[str, object]] = [{"role": "user", "content": "original task"}]
    for index in range(20):
        items.extend(
            [
                {
                    "type": "function_call",
                    "call_id": str(index),
                    "name": "finish",
                    "arguments": json.dumps({"answer": "x" * 20_000}),
                },
                {
                    "type": "function_call_output",
                    "call_id": str(index),
                    "output": json.dumps({"accepted": False, "reasons": ["invalid"]}),
                },
            ]
        )
    filtered = BoundedToolOutputFilter()(
        CallModelData(
            model_data=ModelInputData(input=items, instructions="instructions"),  # type: ignore[arg-type]
            agent=SimpleNamespace(),
            context=context,
        )
    )
    serialized = json.dumps(filtered.input, ensure_ascii=False).encode("utf-8")
    assert len(serialized) <= 60_000
    assert b"function_call" not in serialized
    assert b"semantic-current" in serialized


def test_checkpoint_keeps_projected_bid_catalog_and_bounds_rich_selects() -> None:
    context = make_context()
    context.current_semantic_fingerprint = "semantic-current"
    projected = _project_observation(
        {
            "url": "https://example.test",
            "title": "report",
            "dom_hash": "raw",
            "elements": [
                *[
                    {"bid": f"noise-{index}", "tag": "div", "text": "noise", "visible": True}
                    for index in range(130)
                ],
                {"bid": "next-page", "tag": "button", "text": "Next report page", "visible": True},
            ],
        },
        "Next report page",
        unchanged=False,
    )
    context.latest_model_elements = list(projected["elements"])
    filtered = BoundedToolOutputFilter()(
        CallModelData(
            model_data=ModelInputData(input=[{"role": "user", "content": "task"}], instructions="i"),  # type: ignore[arg-type]
            agent=SimpleNamespace(),
            context=context,
        )
    )
    assert "next-page" in json.dumps(filtered.input, ensure_ascii=False)

    context.latest_model_elements = [
        {
            "bid": f"select-{index}",
            "tag": "select",
            "label": "x" * 300,
            "options": [
                {"value": "v" * 160, "label": "l" * 160, "selected": False}
                for _ in range(40)
            ],
        }
        for index in range(20)
    ]
    bounded = BoundedToolOutputFilter()(
        CallModelData(
            model_data=ModelInputData(input=[{"role": "user", "content": "task"}], instructions="i"),  # type: ignore[arg-type]
            agent=SimpleNamespace(),
            context=context,
        )
    )
    assert len(json.dumps(bounded.input, ensure_ascii=False).encode("utf-8")) <= 60_000


@pytest.mark.asyncio
async def test_schema_failures_are_counted_and_repeated_finish_stops() -> None:
    context = make_context()
    finish_tool = next(tool for tool in TOOLS if tool.name == "finish")
    invalid = json.dumps({"answer": "7"})
    for index in range(2):
        output = await finish_tool.on_invoke_tool(
            ToolContext(context, tool_name="finish", tool_call_id=str(index), tool_arguments=invalid),
            invalid,
        )
        assert json.loads(output)["ok"] is False
    assert context.tool_steps == 2
    assert len(context.tool_outcomes) == 2
    assert "REPEATED_FINISH" in str(context.loop_reason)


def test_timeout_failure_completes_pending_outcome_without_double_counting() -> None:
    context = make_context()
    arguments = {"answer": "7", "evidence_ids": []}
    context.record_call("finish", arguments)
    tool_context = ToolContext(
        context,
        tool_name="finish",
        tool_call_id="timeout",
        tool_arguments=json.dumps(arguments),
    )
    payload = json.loads(_tool_failure(tool_context, TimeoutError("offline timeout")))
    assert payload["ok"] is False
    assert context.tool_steps == 1
    assert len(context.tool_outcomes) == 1


def test_password_fill_timeout_reuses_redacted_pending_arguments() -> None:
    context = make_context()
    context.record_call("fill", {"bid": "password", "value": "[REDACTED]"})
    tool_context = ToolContext(
        context,
        tool_name="fill",
        tool_call_id="timeout",
        tool_arguments=json.dumps({"bid": "password", "value": "plain-text-secret"}),
    )
    payload = json.loads(_tool_failure(tool_context, TimeoutError("offline timeout")))
    serialized = json.dumps(
        {"payload": payload, "actions": context.actions, "outcomes": context.tool_outcomes},
        ensure_ascii=False,
    )
    assert context.tool_steps == 1
    assert len(context.tool_outcomes) == 1
    assert "plain-text-secret" not in serialized


def test_changed_action_updates_checkpoint_and_clears_stale_bids() -> None:
    context = make_context()
    context.current_url = "https://example.test/old"
    context.current_semantic_fingerprint = "old-state"
    context.latest_model_elements = [{"bid": "stale", "tag": "button", "text": "Old"}]
    arguments = {"bid": "next"}
    context.record_call("click", arguments)
    context.record_tool_outcome(
        "click",
        arguments,
        {
            "success": True,
            "before_url": "https://example.test/old",
            "after_url": "https://example.test/new",
            "postconditions": {
                "after_semantic_page_fingerprint": "new-state",
                "post_observation": {
                    "url": "https://example.test/new",
                    "title": "New",
                    "semantic_page_fingerprint": "new-state",
                    "semantic_element_count": 2,
                },
            },
        },
    )
    filtered = BoundedToolOutputFilter()(
        CallModelData(
            model_data=ModelInputData(input=[{"role": "user", "content": "task"}], instructions="i"),  # type: ignore[arg-type]
            agent=SimpleNamespace(),
            context=context,
        )
    )
    checkpoint = json.dumps(filtered.input, ensure_ascii=False)
    assert context.latest_model_elements == []
    assert context.current_url == "https://example.test/new"
    assert "new-state" in checkpoint and "post_observation" in checkpoint
    assert '"stale"' not in checkpoint


@pytest.mark.asyncio
async def test_scalar_finish_safely_defaults_root_evidence_binding() -> None:
    context = make_context()
    evidence = context.evidence_store.add("dom", "https://example.test", "answer", {"data": "7"})
    context.visited_urls.append("https://example.test")
    tool = next(item for item in TOOLS if item.name == "finish")
    arguments = json.dumps({"answer": "7", "evidence_ids": [evidence.evidence_id]})
    output = await tool.on_invoke_tool(
        ToolContext(context, tool_name="finish", tool_call_id="finish", tool_arguments=arguments), arguments
    )
    assert json.loads(output)["accepted"] is True
    assert context.final_bindings == {"$": [evidence.evidence_id]}


@pytest.mark.asyncio
async def test_structured_answer_requires_leaf_bindings_and_checks_numeric_units() -> None:
    context = make_context()
    evidence = context.evidence_store.add(
        "dom", "https://example.test", "answers", {"data": ["A", "B", "7 ft"]}
    )
    context.visited_urls.append("https://example.test")
    tool = next(item for item in TOOLS if item.name == "finish")
    structured = json.dumps({"items": ["A", "B"]})
    arguments = json.dumps({"answer": structured, "evidence_ids": [evidence.evidence_id]})
    output = await tool.on_invoke_tool(
        ToolContext(context, tool_name="finish", tool_call_id="structured", tool_arguments=arguments), arguments
    )
    payload = json.loads(output)
    assert payload["accepted"] is False
    assert any("$.items[0]" in reason for reason in payload["reasons"])
    assert "$" not in context.final_bindings

    mismatch = json.dumps({"answer": "999 miles", "evidence_ids": [evidence.evidence_id]})
    mismatch_output = await tool.on_invoke_tool(
        ToolContext(context, tool_name="finish", tool_call_id="mismatch", tool_arguments=mismatch), mismatch
    )
    mismatch_payload = json.loads(mismatch_output)
    assert mismatch_payload["accepted"] is False
    assert any("数值" in reason or "单位" in reason for reason in mismatch_payload["reasons"])


@pytest.mark.asyncio
async def test_latest_coverage_handle_is_used_and_coverage_publish_is_atomic(monkeypatch) -> None:
    context = make_context()
    context.contract = TaskContract.from_item(
        {"task_idx": 1, "task_id": "x", "website": "https://example.test", "task": "列出所有记录"}
    )
    first_rows = context.evidence_store.add("dom", "https://example.test", "first", {"data": ["A", "B"]})
    latest_rows = context.evidence_store.add("dom", "https://example.test", "latest", {"data": ["A", "B", "C"]})
    terminal = context.evidence_store.add("dom", "https://example.test", "terminal", {"disabled": True})
    context.visited_urls.append("https://example.test")
    coverage_tool = next(tool for tool in TOOLS if tool.name == "record_coverage")

    async def issue(evidence_id: str, total: int, call_id: str) -> str:
        arguments = json.dumps(
            {
                "strategy": "pagination",
                "item_evidence_ids": [evidence_id],
                "pages_visited": total,
                "expected_total": total,
                "terminal_reason": "next_disabled",
                "terminal_evidence_id": terminal.evidence_id,
            }
        )
        return await coverage_tool.on_invoke_tool(
            ToolContext(context, tool_name="record_coverage", tool_call_id=call_id, tool_arguments=arguments),
            arguments,
        )

    await issue(first_rows.evidence_id, 2, "cov-1")
    latest = json.loads(await issue(latest_rows.evidence_id, 3, "cov-2"))["coverage_id"]
    assert context.latest_coverage_id == latest

    finish_tool = next(tool for tool in TOOLS if tool.name == "finish")
    bindings = [
        {"path": f"$[{index}]", "evidence_ids": [latest_rows.evidence_id]}
        for index in range(3)
    ]
    finish_args = json.dumps(
        {"answer": json.dumps(["A", "B", "C"]), "evidence_ids": [latest_rows.evidence_id], "evidence_bindings": bindings}
    )
    output = await finish_tool.on_invoke_tool(
        ToolContext(context, tool_name="finish", tool_call_id="finish-latest", tool_arguments=finish_args),
        finish_args,
    )
    assert json.loads(output)["accepted"] is True
    assert context.final_coverage and context.final_coverage.unique_item_count == 3

    failed_context = make_context()
    rows = failed_context.evidence_store.add("dom", "https://example.test", "rows", {"data": ["A"]})
    end = failed_context.evidence_store.add("dom", "https://example.test", "end", {"disabled": True})
    original_add = failed_context.evidence_store.add

    def fail_receipt(source, url, summary, payload):  # noqa: ANN001, ANN202
        if source == "receipt":
            raise OSError("offline disk failure")
        return original_add(source, url, summary, payload)

    monkeypatch.setattr(failed_context.evidence_store, "add", fail_receipt)
    failed_arguments = json.dumps(
        {
            "strategy": "pagination",
            "item_evidence_ids": [rows.evidence_id],
            "pages_visited": 1,
            "expected_total": 1,
            "terminal_reason": "next_disabled",
            "terminal_evidence_id": end.evidence_id,
        }
    )
    failed = await coverage_tool.on_invoke_tool(
        ToolContext(failed_context, tool_name="record_coverage", tool_call_id="failed", tool_arguments=failed_arguments),
        failed_arguments,
    )
    assert json.loads(failed)["ok"] is False
    assert failed_context.coverage_records == {}
    assert len(failed_context.tool_outcomes) == 1


def test_tabs_outcome_uses_active_url_for_semantic_state() -> None:
    context = make_context()
    arguments = {"action": "list", "index": None, "url": None}
    context.record_call("tabs", arguments)
    context.record_tool_outcome(
        "tabs",
        arguments,
        {
            "ok": True,
            "tabs": [
                {"url": "https://example.test/active", "active": True},
                {"url": "https://example.test/background", "active": False},
            ],
            "semantic_page_fingerprint": "active-state",
        },
    )
    assert context.current_url == "https://example.test/active"
    assert context.last_observation_state == ("https://example.test/active", "active-state")


def test_plain_english_and_chinese_list_shapes_enforce_complete_counts() -> None:
    certificate = CoverageCertificate(
        strategy="pagination",
        unique_item_count=3,
        pages_visited=2,
        expected_total=3,
        terminal_reason="next_disabled",
        terminal_evidence_id="ev-terminal",
        item_fingerprint="a" * 64,
    )
    top = TaskContract.from_item(
        {"task_idx": 1, "task_id": "top", "website": "https://example.test", "task": "Top 3 companies"}
    )
    english_all = TaskContract.from_item(
        {"task_idx": 2, "task_id": "all", "website": "https://example.test", "task": "List all companies"}
    )
    assert _answer_shape_reasons(top, "A, B", certificate)
    assert _answer_shape_reasons(english_all, "A\nB", certificate)
    assert _answer_shape_reasons(top, "A、B、C", certificate) == []

    chinese_top = TaskContract.from_item(
        {
            "task_idx": 3,
            "task_id": "chinese-top",
            "website": "https://example.test",
            "task": "最高的前三个季度是哪三个",
        }
    )
    assert _answer_shape_reasons(chinese_top, "第一季度", certificate)
    assert _answer_shape_reasons(chinese_top, "第一季度、第二季度、第三季度", certificate) == []

    wid = TaskContract.from_item(
        {
            "task_idx": 4,
            "task_id": "wid",
            "website": "https://example.test",
            "task": "在WID中选择Country=Mongolia、Indicator=Top 10% income share、Year=2010,图表上hover显示的值是多少(%)?",
        }
    )
    assert wid.requires_coverage is False
    assert _answer_shape_reasons(wid, "32%", None) == []


@pytest.mark.asyncio
async def test_finish_rejects_unit_mismatch_and_requires_explicit_task_unit() -> None:
    finish_tool = next(item for item in TOOLS if item.name == "finish")

    mismatch_context = make_context()
    mismatch_evidence = mismatch_context.evidence_store.add(
        "dom", "https://example.test", "height", {"data": "100 ft"}
    )
    mismatch_context.visited_urls.append("https://example.test")
    mismatch_arguments = json.dumps(
        {"answer": "100 m", "evidence_ids": [mismatch_evidence.evidence_id]}
    )
    mismatch_output = await finish_tool.on_invoke_tool(
        ToolContext(
            mismatch_context,
            tool_name="finish",
            tool_call_id="unit-mismatch",
            tool_arguments=mismatch_arguments,
        ),
        mismatch_arguments,
    )
    assert any("单位" in reason for reason in json.loads(mismatch_output)["reasons"])

    percent_context = make_context()
    percent_context.contract = TaskContract.from_item(
        {
            "task_idx": 2,
            "task_id": "percent",
            "website": "https://example.test",
            "task": "What is the reported value (%)?",
        }
    )
    percent_evidence = percent_context.evidence_store.add(
        "dom", "https://example.test", "percent", {"data": "52%"}
    )
    percent_context.visited_urls.append("https://example.test")
    missing_unit_arguments = json.dumps(
        {"answer": "52", "evidence_ids": [percent_evidence.evidence_id]}
    )
    missing_unit_output = await finish_tool.on_invoke_tool(
        ToolContext(
            percent_context,
            tool_name="finish",
            tool_call_id="percent-missing",
            tool_arguments=missing_unit_arguments,
        ),
        missing_unit_arguments,
    )
    assert any("缺少题目明确要求的单位" in reason for reason in json.loads(missing_unit_output)["reasons"])

    accepted_context = make_context()
    accepted_context.contract = percent_context.contract
    accepted_evidence = accepted_context.evidence_store.add(
        "dom", "https://example.test", "percent", {"data": "52%"}
    )
    accepted_context.visited_urls.append("https://example.test")
    accepted_arguments = json.dumps(
        {"answer": "52%", "evidence_ids": [accepted_evidence.evidence_id]}
    )
    accepted_output = await finish_tool.on_invoke_tool(
        ToolContext(
            accepted_context,
            tool_name="finish",
            tool_call_id="percent-accepted",
            tool_arguments=accepted_arguments,
        ),
        accepted_arguments,
    )
    assert json.loads(accepted_output)["accepted"] is True


def test_tpm_cold_start_calibration_and_reconcile_low_estimate() -> None:
    metrics = model_input_metrics("system", [{"role": "user", "content": "x" * 10_000}], [])
    serialized = metrics["serialized_context_bytes"]
    cold = estimate_input_tokens("system", [{"role": "user", "content": "x" * 10_000}], [])
    samples = [(float(index), "agent", 0.25) for index in range(20)]
    events: list[tuple[float, int, str]] = []
    limiter = SharedTPMLimiter(
        events,
        threading.RLock(),
        token_budget=10_000,
        clock=lambda: 30.0,
        calibration_samples=samples,
    )
    calibrated = limiter.estimate_tokens(serialized, "agent")
    assert calibrated < cold
    assert calibrated >= int(serialized * 0.28)

    async def reserve_and_reconcile() -> None:
        reservation = await limiter.acquire(calibrated)
        limiter.reconcile(reservation, calibrated + 500, serialized, "agent")

    asyncio.run(reserve_and_reconcile())
    assert events[-1][1] == calibrated + 500
    assert len(samples) == 21


@pytest.mark.asyncio
async def test_missing_usage_keeps_reservation_and_context_limit_blocks_send() -> None:
    events: list[tuple[float, int, str]] = []
    limiter = SharedTPMLimiter(events, threading.RLock(), token_budget=100_000)
    reservation = await limiter.acquire(1_000)
    limiter.reconcile(reservation, None, 10_000, "agent")
    assert events[0][1] == 1_000

    class MustNotRun:
        async def get_response(self, *args, **kwargs):  # noqa: ANN002, ANN003
            raise AssertionError("model must not run")

    model = ThrottledModel(MustNotRun(), None)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="MODEL_INPUT_LIMIT"):
        await model.get_response(
            "system",
            [{"role": "user", "content": "x" * (MAX_SERIALIZED_CONTEXT_BYTES + 1)}],
            None,
            [],
            None,
            [],
            None,
            previous_response_id=None,
            conversation_id=None,
            prompt=None,
        )


@pytest.mark.asyncio
async def test_cancel_and_reconcile_lock_failure_keep_fail_closed_reservation() -> None:
    events: list[tuple[float, int, str]] = []
    limiter = SharedTPMLimiter(events, threading.RLock(), token_budget=100_000)

    class CancelledModel:
        async def get_response(self, *args, **kwargs):  # noqa: ANN002, ANN003
            raise asyncio.CancelledError

    model = ThrottledModel(CancelledModel(), limiter)  # type: ignore[arg-type]
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
    assert len(events) == 1 and events[0][1] > 0

    class UnavailableLock:
        def acquire(self, timeout: float) -> bool:
            return False

        def release(self) -> None:
            raise AssertionError("must not release")

    reservation = {"reservation_id": events[0][2]}
    limiter.lock = UnavailableLock()
    with pytest.raises(RuntimeError, match="TPM_LIMITER_LOCK_TIMEOUT"):
        limiter.reconcile(reservation, 500, 2_000, "agent")
    assert events[0][1] > 0


@pytest.mark.asyncio
async def test_request_observability_records_bounded_categories_and_latencies() -> None:
    class SuccessfulModel:
        async def get_response(self, *args, **kwargs):  # noqa: ANN002, ANN003
            return SimpleNamespace(
                usage=Usage(requests=1, input_tokens=123, output_tokens=4, total_tokens=127)
            )

    stats = TaskUsageStats(3, 8, "sensitive-task-id")
    stats.set_runtime_snapshot(
        {
            "last_tool": "extract",
            "semantic_state": "semantic-1",
            "progress_reason": ["new_content_hash"],
            "repeat_count": 0,
            "cycle_count": 0,
            "cache_hit": False,
            "tool_latency_ms": 2.5,
            "browser_latency_ms": 2.0,
        }
    )
    model = ThrottledModel(SuccessfulModel(), None)  # type: ignore[arg-type]
    model.usage_stats = stats
    await model.get_response(
        "system",
        [{"role": "user", "content": "offline"}],
        None,
        [],
        None,
        [],
        None,
        previous_response_id=None,
        conversation_id=None,
        prompt=None,
    )
    request = stats.requests[0]
    assert request["worker_id"] == 3 and request["task_idx"] == 8
    assert request["serialized_context_bytes"] > 0
    assert request["serialized_context_items"] == 1
    assert request["category_bytes"]["user_items"] > 0
    assert request["last_tool"] == "extract"
    assert request["model_latency_ms"] >= 0
    assert "sensitive-task-id" not in json.dumps(request)


@pytest.mark.asyncio
async def test_stream_usage_reconciles_before_terminal_event_and_missing_usage_stays_reserved() -> None:
    usage = Usage(requests=1, input_tokens=5_000, output_tokens=10, total_tokens=5_010)

    class StreamModel:
        def __init__(self, terminal_usage):  # noqa: ANN001
            self.terminal_usage = terminal_usage
            self.calls = 0

        async def stream_response(self, *args, **kwargs):  # noqa: ANN002, ANN003
            self.calls += 1
            yield SimpleNamespace(
                type="response.completed",
                response=SimpleNamespace(usage=self.terminal_usage),
            )

    events: list[tuple[float, int, str]] = []
    samples: list[tuple[float, str, float]] = []
    limiter = SharedTPMLimiter(
        events,
        threading.RLock(),
        token_budget=100_000,
        calibration_samples=samples,
    )
    wrapped = StreamModel(usage)
    model = ThrottledModel(wrapped, limiter)  # type: ignore[arg-type]
    stats = TaskUsageStats(0, 1, "stream-task")
    model.usage_stats = stats
    stream = model.stream_response(
        "system",
        [{"role": "user", "content": "offline"}],
        None,
        [],
        None,
        [],
        None,
        previous_response_id=None,
        conversation_id=None,
        prompt=None,
    )
    terminal = await anext(stream)
    assert terminal.type == "response.completed"
    await stream.aclose()
    assert events[0][1] == 5_000
    assert len(samples) == 1
    assert len(stats.requests) == 1 and stats.requests[0]["input_tokens"] == 5_000

    missing_events: list[tuple[float, int, str]] = []
    missing_wrapped = StreamModel(None)
    missing_model = ThrottledModel(
        missing_wrapped,
        SharedTPMLimiter(missing_events, threading.RLock(), token_budget=100_000),
    )  # type: ignore[arg-type]
    missing_stats = TaskUsageStats(0, 2, "missing-stream-usage")
    missing_model.usage_stats = missing_stats
    missing_stream = missing_model.stream_response(
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
    await anext(missing_stream)
    await missing_stream.aclose()
    assert missing_events[0][1] == missing_stats.requests[0]["estimated_input_tokens"]
    assert missing_stats.requests[0]["usage_available"] is False


@pytest.mark.asyncio
async def test_stream_context_limit_is_observed_without_calling_wrapped_model() -> None:
    class MustNotStream:
        calls = 0

        async def stream_response(self, *args, **kwargs):  # noqa: ANN002, ANN003
            self.calls += 1
            yield SimpleNamespace(type="unexpected")

    wrapped = MustNotStream()
    model = ThrottledModel(wrapped, None)  # type: ignore[arg-type]
    stats = TaskUsageStats(0, 1, "oversized-stream")
    model.usage_stats = stats
    with pytest.raises(ValueError, match="MODEL_INPUT_LIMIT"):
        async for _ in model.stream_response(
            "system",
            [{"role": "user", "content": "x" * (MAX_SERIALIZED_CONTEXT_BYTES + 1)}],
            None,
            [],
            None,
            [],
            None,
            previous_response_id=None,
            conversation_id=None,
            prompt=None,
        ):
            pass
    assert wrapped.calls == 0
    assert stats.requests[0]["throttle_reason"] == "pre_send_context_limit"


def test_tpm_reservations_are_unique_across_threads_and_expired_reconcile_is_safe() -> None:
    events: list[tuple[float, int, str]] = []
    limiter = SharedTPMLimiter(events, threading.RLock(), token_budget=100)
    barrier = threading.Barrier(2)
    reservations: list[dict[str, object]] = []

    def reserve() -> None:
        barrier.wait()
        reservations.append(asyncio.run(limiter.acquire(10)))

    threads = [threading.Thread(target=reserve) for _ in range(2)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    reservation_ids = {str(item["reservation_id"]) for item in reservations}
    assert len(reservation_ids) == 2
    assert reservation_ids == {str(event[2]) for event in events}

    now = [0.0]
    expiring_events: list[tuple[float, int, str]] = []
    samples: list[tuple[float, str, float]] = []
    expiring = SharedTPMLimiter(
        expiring_events,
        threading.RLock(),
        token_budget=100,
        clock=lambda: now[0],
        calibration_samples=samples,
    )

    async def expire_and_reconcile() -> tuple[dict[str, object], dict[str, object]]:
        old = await expiring.acquire(10)
        now[0] = 61.0
        new = await expiring.acquire(20)
        expiring.reconcile(old, 30, 100, "agent")
        return old, new

    old, new = asyncio.run(expire_and_reconcile())
    assert [event[2] for event in expiring_events] == [new["reservation_id"]]
    assert samples[-1][2] == 0.3
    with pytest.raises(RuntimeError, match="MISSING_RESERVATION"):
        expiring.reconcile({"reservation_id": "unknown", "reserved_at": now[0]}, 10, 100, "agent")
