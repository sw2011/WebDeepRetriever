from __future__ import annotations

import json
from types import SimpleNamespace

import pytest
from agents import FunctionToolResult, MaxTurnsExceeded, RunContextWrapper
from agents.tool_context import ToolContext
from agents.run_config import CallModelData, ModelInputData
from openai import AsyncOpenAI

from web_agent.contracts import TaskContract
from web_agent.evidence import EvidenceStore
from web_agent.runtime import (
    TOOLS,
    BoundedToolOutputFilter,
    ProtocolIIIAgent,
    TaskRuntimeContext,
    _coverage_items,
    _error,
    _kimi_request_options,
    _model_settings,
    _verified_finish_behavior,
)
from web_agent.verifier import CompletionVerifier


class DummyActor:
    async def audit_step(self, label: str) -> str:
        return label


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


def test_single_user_run_trims_old_tool_outputs_by_count() -> None:
    items: list[dict[str, object]] = [{"role": "user", "content": "task"}]
    for index in range(10):
        items.extend(
            [
                {"type": "function_call", "call_id": str(index), "name": "observe", "arguments": "{}"},
                {"type": "function_call_output", "call_id": str(index), "output": "x" * 10_000},
            ]
        )
    model_data = ModelInputData(input=items, instructions="instructions")  # type: ignore[arg-type]
    filtered = BoundedToolOutputFilter(keep_recent_outputs=2, old_output_chars=100)(
        CallModelData(model_data=model_data, agent=SimpleNamespace(), context=None)  # type: ignore[arg-type]
    )
    outputs = [item["output"] for item in filtered.input if isinstance(item, dict) and item.get("type") == "function_call_output"]
    assert sum("trimmed_old_tool_output" in str(output) for output in outputs) == 8
    assert outputs[-1] == "x" * 10_000


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
