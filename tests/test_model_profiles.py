from __future__ import annotations

import os

import pytest
from openai import AsyncOpenAI

from web_agent.model_profiles import (
    GENERIC_OPENAI_COMPATIBLE,
    MOONSHOT_KIMI_K26,
    OPENAI_GPT41,
    build_model_settings,
    capability_profile,
    normalized_model_name,
)


def test_known_and_unknown_model_profiles_are_explicit() -> None:
    assert capability_profile("gpt-4.1-mini") is OPENAI_GPT41
    assert capability_profile("kimi-k2.6") is MOONSHOT_KIMI_K26
    assert capability_profile("future-model") is GENERIC_OPENAI_COMPATIBLE
    assert GENERIC_OPENAI_COMPATIBLE.thinking == "unknown"
    assert GENERIC_OPENAI_COMPATIBLE.reasoning == "unknown"


def test_provider_prefix_does_not_hide_known_underlying_model() -> None:
    assert normalized_model_name("openai/gpt-4.1-mini") == "gpt-4.1-mini"
    assert capability_profile("openai/gpt-4.1-mini") is OPENAI_GPT41
    assert capability_profile("moonshot/kimi-k2.6") is MOONSHOT_KIMI_K26
    assert capability_profile("gateway/moonshot/kimi-k2.6") is MOONSHOT_KIMI_K26
    assert capability_profile("vendor/unknown-model") is GENERIC_OPENAI_COMPATIBLE


def test_profile_settings_cover_required_tool_compatibility() -> None:
    kimi = build_model_settings("moonshot/kimi-k2.6")
    assert kimi.temperature == 0.6
    assert kimi.tool_choice == "required"
    assert kimi.parallel_tool_calls is False
    assert kimi.reasoning is None
    assert kimi.extra_body == {"thinking": {"type": "disabled"}}

    openai = build_model_settings("openai/gpt-4.1-mini")
    assert openai.temperature == 0
    assert openai.tool_choice == "required"
    assert openai.parallel_tool_calls is False
    assert openai.reasoning is None
    assert openai.extra_body is None


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"temperature": 0}, "temperature"),
        ({"tool_choice": "auto"}, "tool_choice"),
        ({"parallel_tool_calls": True}, "parallel_tool_calls"),
        ({"reasoning": {"effort": "high"}}, "reasoning"),
        ({"extra_body": {"thinking": {"type": "enabled"}}}, "extra_body.thinking.type"),
    ],
)
def test_kimi_profile_rejects_parameter_conflicts(
    overrides: dict[str, object],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        build_model_settings("kimi-k2.6", overrides)


def test_unknown_profile_keeps_conservative_existing_defaults() -> None:
    settings = build_model_settings("provider/future-model")
    assert settings.temperature == 0
    assert settings.tool_choice == "required"
    assert settings.parallel_tool_calls is False
    assert settings.extra_body is None


def test_compatible_extra_body_extension_is_merged_without_mutating_profile() -> None:
    settings = build_model_settings("kimi-k2.6", {"extra_body": {"trace_id": "contract-test"}})
    assert settings.extra_body == {
        "thinking": {"type": "disabled"},
        "trace_id": "contract-test",
    }
    assert MOONSHOT_KIMI_K26.extra_body == {"thinking": {"type": "disabled"}}


@pytest.mark.parametrize("field", ["temperature", "tool_choice", "parallel_tool_calls", "reasoning_effort"])
def test_extra_body_cannot_bypass_top_level_profile_conflicts(field: str) -> None:
    with pytest.raises(ValueError, match=field):
        build_model_settings("kimi-k2.6", {"extra_body": {field: "conflict"}})


@pytest.mark.live
@pytest.mark.asyncio
async def test_live_kimi_tool_call_contract() -> None:
    if os.getenv("WEBRETRIEVER_RUN_LIVE_MODEL_TESTS") != "1":
        pytest.skip("live 模型契约测试需要显式设置 WEBRETRIEVER_RUN_LIVE_MODEL_TESTS=1")
    api_key = os.getenv("MOONSHOT_API_KEY")
    if not api_key:
        pytest.skip("未提供 MOONSHOT_API_KEY")
    model = os.getenv("WEBRETRIEVER_LIVE_MODEL", "kimi-k2.6")
    profile = capability_profile(model)
    if profile is not MOONSHOT_KIMI_K26:
        pytest.skip("live 契约当前只覆盖已有真实依据的 kimi-k2.6 画像")

    client = AsyncOpenAI(
        api_key=api_key,
        base_url=os.getenv("WEBRETRIEVER_LIVE_API_BASE", "https://api.moonshot.cn/v1"),
        max_retries=0,
    )
    response = await client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": "调用 echo 工具并传入 value=ok"}],
        tools=[
            {
                "type": "function",
                "function": {
                    "name": "echo",
                    "description": "返回输入值",
                    "parameters": {
                        "type": "object",
                        "properties": {"value": {"type": "string"}},
                        "required": ["value"],
                        "additionalProperties": False,
                    },
                },
            }
        ],
        max_tokens=80,
        **profile.agent_request_options(),
    )
    calls = response.choices[0].message.tool_calls or []
    assert len(calls) == 1
    assert calls[0].function.name == "echo"
