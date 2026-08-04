from __future__ import annotations

import re
from typing import Any
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit


SENSITIVE_FIELD = re.compile(
    r"(?:password|passwd|token|secret|api[-_]?key|credential|authorization|cookie|session|sess|security|signature|(?:^|[-_])(?:auth|code|sig|key)(?:$|[-_]))",
    re.I,
)
_SENSITIVE_ASSIGNMENT = re.compile(
    r"(?i)\b(password|passwd|token|access[-_]?token|secret|api[-_]?key|credential|authorization|cookie|session|sess|security|signature)"
    r"(\s*[=:]\s*)([^\s,;&]+)"
)
_BEARER_VALUE = re.compile(r"(?i)\bbearer\s+[a-z0-9._~+/=-]+")
_JWT_VALUE = re.compile(r"\beyJ[a-zA-Z0-9_-]{8,}\.[a-zA-Z0-9_-]{8,}\.[a-zA-Z0-9_-]{8,}\b")
_PREFIXED_KEY = re.compile(r"\b(?:sk|ak|org|proj)-[a-zA-Z0-9_-]{8,}\b", re.I)
_URL = re.compile(r"(?:https?|wss?)://[^\s<>'\"]+")


def redact_text(value: str) -> str:
    value = _SENSITIVE_ASSIGNMENT.sub(lambda match: f"{match.group(1)}{match.group(2)}[REDACTED]", value)
    value = _BEARER_VALUE.sub("[REDACTED]", value)
    value = _JWT_VALUE.sub("[REDACTED]", value)
    return _PREFIXED_KEY.sub("[REDACTED]", value)


def sanitize_url(value: str) -> str:
    try:
        parsed = urlsplit(value)
        hostname = parsed.hostname or ""
        if ":" in hostname:
            hostname = f"[{hostname}]"
        try:
            port = f":{parsed.port}" if parsed.port is not None else ""
        except ValueError:
            port = ""
        netloc = f"redacted@{hostname}{port}" if parsed.username or parsed.password else parsed.netloc
        query = urlencode(
            [
                (key, "[REDACTED]" if SENSITIVE_FIELD.search(key) else redact_text(item))
                for key, item in parse_qsl(parsed.query, keep_blank_values=True)
            ],
            doseq=True,
        )
        return urlunsplit((parsed.scheme, netloc, redact_text(parsed.path), query, redact_text(parsed.fragment)))
    except (TypeError, ValueError):
        return redact_text(str(value))


def sanitize_error_text(value: str) -> str:
    redacted = redact_text(value)
    return _URL.sub(lambda match: sanitize_url(match.group(0)), redacted)


def redact_value(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            key: "[REDACTED]" if SENSITIVE_FIELD.search(str(key)) else redact_value(item)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [redact_value(item) for item in value]
    if isinstance(value, tuple):
        return tuple(redact_value(item) for item in value)
    if isinstance(value, str):
        return sanitize_error_text(value)
    return value


def sanitize_exception(exc: BaseException) -> str:
    detail = sanitize_error_text(str(exc)).strip()
    return f"{type(exc).__name__}: {detail}" if detail else type(exc).__name__


def public_error_summary(value: str) -> str:
    sanitized = sanitize_error_text(value)
    lowered = sanitized.casefold()
    if "429" in lowered and ("tpm" in lowered or "rate_limit" in lowered or "rate limit" in lowered):
        return "模型 API 429：组织级 TPM 限额"
    if "econnrefused" in lowered and "connect_over_cdp" in lowered:
        return "浏览器 CDP 连接失败"
    return sanitized
