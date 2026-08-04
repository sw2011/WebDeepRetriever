from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Any, Literal, Mapping

from agents import ModelSettings


Capability = Literal["supported", "unsupported", "unknown", "must_disable"]


@dataclass(frozen=True)
class ModelCapabilityProfile:
    name: str
    provider: str
    models: tuple[str, ...]
    thinking: Capability
    reasoning: Capability
    temperature: float
    tool_choice: Literal["required"] = "required"
    parallel_tool_calls: Literal[False] = False
    extra_body: Mapping[str, Any] | None = None

    def agent_request_options(self) -> dict[str, Any]:
        options: dict[str, Any] = {
            "tool_choice": self.tool_choice,
            "parallel_tool_calls": self.parallel_tool_calls,
            "temperature": self.temperature,
        }
        if self.extra_body is not None:
            options["extra_body"] = _copy_mapping(self.extra_body)
        return options

    def vision_request_options(self) -> dict[str, Any]:
        if self.thinking != "must_disable":
            return {}
        return {
            "temperature": self.temperature,
            "extra_body": _copy_mapping(self.extra_body or {}),
        }


GENERIC_OPENAI_COMPATIBLE = ModelCapabilityProfile(
    name="generic-openai-compatible",
    provider="unknown",
    models=(),
    thinking="unknown",
    reasoning="unknown",
    temperature=0,
)

OPENAI_GPT41 = ModelCapabilityProfile(
    name="openai-gpt-4.1",
    provider="openai",
    models=("gpt-4.1-mini",),
    thinking="unsupported",
    reasoning="unsupported",
    temperature=0,
)

MOONSHOT_KIMI_K26 = ModelCapabilityProfile(
    name="moonshot-kimi-k2.6",
    provider="moonshot",
    models=("kimi-k2.6",),
    thinking="must_disable",
    reasoning="unsupported",
    temperature=0.6,
    extra_body={"thinking": {"type": "disabled"}},
)

MODEL_CAPABILITY_PROFILES = (OPENAI_GPT41, MOONSHOT_KIMI_K26)
_MODEL_SETTINGS_FIELDS = {field.name for field in fields(ModelSettings)}
_PROTECTED_EXTRA_BODY_FIELDS = (
    _MODEL_SETTINGS_FIELDS
    - {"extra_body", "extra_query", "extra_headers", "extra_args", "retry"}
) | {
    "model",
    "messages",
    "tools",
    "stream",
    "reasoning_effort",
    "max_completion_tokens",
}


def normalized_model_name(model: str) -> str:
    normalized = model.strip().casefold().rstrip("/")
    if not normalized:
        raise ValueError("模型名称不能为空")
    return normalized.rsplit("/", 1)[-1]


def capability_profile(model: str) -> ModelCapabilityProfile:
    model_name = normalized_model_name(model)
    for profile in MODEL_CAPABILITY_PROFILES:
        if model_name in profile.models:
            return profile
    return GENERIC_OPENAI_COMPATIBLE


def build_model_settings(
    model: str,
    overrides: Mapping[str, Any] | None = None,
) -> ModelSettings:
    profile = capability_profile(model)
    options = profile.agent_request_options()
    if overrides:
        unknown = set(overrides) - _MODEL_SETTINGS_FIELDS
        if unknown:
            raise ValueError(f"未知模型参数: {', '.join(sorted(unknown))}")
        for key, value in overrides.items():
            if value is None:
                continue
            if key == "extra_body":
                if not isinstance(value, Mapping):
                    raise ValueError("extra_body 必须是映射")
                protected = sorted(
                    str(item) for item in value if str(item).casefold() in _PROTECTED_EXTRA_BODY_FIELDS
                )
                if protected:
                    raise ValueError(
                        "extra_body 不能覆盖标准请求字段: " + ", ".join(protected)
                    )
                options[key] = _merge_without_conflicts(options.get(key, {}), value, "extra_body")
                continue
            if key in options and options[key] != value:
                raise ValueError(
                    f"模型画像 {profile.name} 要求 {key}={options[key]!r}，不能覆盖为 {value!r}"
                )
            if key == "reasoning" and profile.reasoning != "supported":
                raise ValueError(f"模型画像 {profile.name} 未确认支持 reasoning 参数")
            options[key] = value
    return ModelSettings(**options)


def vision_request_options(model: str) -> dict[str, Any]:
    return capability_profile(model).vision_request_options()


def _copy_mapping(value: Mapping[str, Any]) -> dict[str, Any]:
    return {
        str(key): _copy_mapping(item) if isinstance(item, Mapping) else item
        for key, item in value.items()
    }


def _merge_without_conflicts(
    required: Mapping[str, Any],
    override: Mapping[str, Any],
    path: str,
) -> dict[str, Any]:
    merged = _copy_mapping(required)
    for raw_key, value in override.items():
        key = str(raw_key)
        current_path = f"{path}.{key}"
        if key not in merged:
            merged[key] = _copy_mapping(value) if isinstance(value, Mapping) else value
            continue
        existing = merged[key]
        if isinstance(existing, Mapping) and isinstance(value, Mapping):
            merged[key] = _merge_without_conflicts(existing, value, current_path)
        elif existing != value:
            raise ValueError(
                f"模型画像要求 {current_path}={existing!r}，不能覆盖为 {value!r}"
            )
    return merged
