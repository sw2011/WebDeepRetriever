from __future__ import annotations

import hashlib
import importlib.metadata
import importlib.util
import json
import math
import platform
import subprocess
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

from .runtime import SYSTEM_INSTRUCTIONS, TOOLS
from .sanitization import sanitize_url


MANIFEST_SCHEMA_VERSION = 1
_PACKAGE_ROOT = Path(__file__).resolve().parent


def _find_git_root(start: Path) -> Path | None:
    for candidate in (start, *start.parents):
        if (candidate / ".git").exists():
            return candidate
    return None


_PROJECT_ROOT = _find_git_root(_PACKAGE_ROOT)
_IDENTITY_KEYS = (
    "schema_version",
    "dataset",
    "git",
    "model",
    "prompt",
    "tools",
    "configuration",
    "runtime",
)


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _canonical_hash(value: Any) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
        allow_nan=False,
    ).encode("utf-8")
    return _sha256_bytes(encoded)


def _git_output(*args: str) -> bytes | None:
    if _PROJECT_ROOT is None:
        return None
    try:
        result = subprocess.run(
            ["git", *args],
            cwd=_PROJECT_ROOT,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return result.stdout


def _source_hash() -> str:
    digest = hashlib.sha256()
    roots: list[tuple[str, Path]] = [("web_agent", _PACKAGE_ROOT)]
    try:
        browsergym = importlib.util.find_spec("browsergym")
    except (ImportError, AttributeError, ValueError):
        browsergym = None
    if browsergym is not None:
        roots.extend(
            ("browsergym", Path(location).resolve())
            for location in (browsergym.submodule_search_locations or [])
        )
    file_count = 0
    seen: set[tuple[str, str]] = set()
    for package_name, root in roots:
        for path in sorted(root.rglob("*")):
            if not path.is_file() or path.suffix not in {".py", ".js"}:
                continue
            identity = (package_name, path.relative_to(root).as_posix())
            if identity in seen:
                continue
            seen.add(identity)
            file_count += 1
            digest.update(f"{package_name}/{identity[1]}".encode("utf-8"))
            digest.update(b"\0")
            digest.update(path.read_bytes())
            digest.update(b"\0")
    if file_count == 0:
        raise RuntimeError("未找到可用于运行指纹的已加载源码")
    return digest.hexdigest()


def _git_state() -> dict[str, Any]:
    sha = _git_output("rev-parse", "HEAD")
    status = _git_output("status", "--porcelain=v1", "--untracked-files=all")
    diff = _git_output(
        "diff",
        "--binary",
        "HEAD",
        "--",
        "src/web_agent",
        "vendor/browsergym/src/browsergym",
        "pyproject.toml",
    )
    dirty = bool(status.strip()) if status is not None else None
    return {
        "sha": sha.decode("ascii", errors="replace").strip() if sha else "unknown",
        "dirty": dirty,
        "working_tree_hash": _sha256_bytes(diff or b"") if dirty else None,
        "source_hash": _source_hash(),
    }


def _package_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "unknown"


def _tool_schema_hash() -> str:
    schemas = [
        {
            "name": tool.name,
            "schema": tool.params_json_schema,
        }
        for tool in TOOLS
    ]
    return _canonical_hash(schemas)


def build_run_manifest(
    input_path: Path,
    cdp_urls: list[str],
    model: str,
    api_base: str,
    max_steps: int,
    worker_count: int,
    worker_watchdog_seconds: float,
) -> dict[str, Any]:
    if not math.isfinite(worker_watchdog_seconds) or worker_watchdog_seconds <= 0:
        raise ValueError("worker watchdog 超时必须是有限正数")
    sanitized_api_base = sanitize_url(api_base)
    provider = (urlsplit(sanitized_api_base).hostname or "unknown").casefold()
    identity = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "dataset": {"sha256": _sha256_bytes(Path(input_path).read_bytes())},
        "git": _git_state(),
        "model": {
            "name": model,
            "provider": provider,
            "api_base": sanitized_api_base,
        },
        "prompt": {"sha256": _sha256_bytes(SYSTEM_INSTRUCTIONS.encode("utf-8"))},
        "tools": {"schema_sha256": _tool_schema_hash()},
        "configuration": {
            "worker_count": worker_count,
            "max_steps": min(max(int(max_steps), 1), 100),
            "worker_watchdog_seconds": float(worker_watchdog_seconds),
            "transport": "playwright_cdp",
            "context_mode": "shared_existing",
            "cdp_endpoints": [sanitize_url(value) for value in cdp_urls[:worker_count]],
        },
        "runtime": {
            "python": platform.python_version(),
            "playwright": _package_version("playwright"),
            "openai": _package_version("openai"),
            "openai_agents": _package_version("openai-agents"),
        },
    }
    return {
        **identity,
        "fingerprint": _canonical_hash(identity),
        "run_id": uuid.uuid4().hex,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "dataset_path": str(Path(input_path).resolve()),
    }


def load_manifest(path: Path) -> dict[str, Any] | None:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(value, dict) or not isinstance(value.get("fingerprint"), str):
        return None
    if any(key not in value for key in _IDENTITY_KEYS):
        return None
    identity = {key: value[key] for key in _IDENTITY_KEYS}
    if value["fingerprint"] != _canonical_hash(identity):
        return None
    return value


def manifest_matches(existing: dict[str, Any] | None, current: dict[str, Any]) -> bool:
    return bool(
        existing
        and existing.get("fingerprint") == current.get("fingerprint")
        and existing.get("schema_version") == MANIFEST_SCHEMA_VERSION
        and isinstance(existing.get("run_id"), str)
        and bool(existing["run_id"].strip())
    )
