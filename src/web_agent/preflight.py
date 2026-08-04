from __future__ import annotations

import importlib.metadata
import json
import math
import multiprocessing as mp
import os
import platform
import stat
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable
from urllib.parse import parse_qs, parse_qsl, urlencode, urlsplit, urlunsplit

from .sanitization import sanitize_error_text, sanitize_exception, sanitize_url


PREFLIGHT_SCHEMA_VERSION = 1


class PreflightProbeError(RuntimeError):
    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


def _package_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "missing"


def _atomic_write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)
    with tempfile.NamedTemporaryFile(
        "w",
        encoding="utf-8",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    ) as output:
        output.write(encoded)
        output.flush()
        os.fsync(output.fileno())
        temporary = Path(output.name)
    temporary.replace(path)


def _endpoint_identity(value: str) -> str:
    parsed = urlsplit(value)
    host = (parsed.hostname or "").casefold()
    port = parsed.port
    if port is not None and not (
        (parsed.scheme.casefold() in {"http", "ws"} and port == 80)
        or (parsed.scheme.casefold() in {"https", "wss"} and port == 443)
    ):
        host = f"{host}:{port}"
    public = urlsplit(sanitize_url(value))
    query = urlencode(sorted(parse_qsl(public.query, keep_blank_values=True)))
    return urlunsplit((parsed.scheme.casefold(), host, parsed.path or "/", query, ""))


def validate_cdp_urls(cdp_urls: list[str]) -> tuple[list[str], dict[str, Any] | None]:
    if not 1 <= len(cdp_urls) <= 8:
        return [], {
            "code": "PREFLIGHT_CDP_WORKER_COUNT",
            "message": "CDP worker 数必须为 1 到 8",
        }
    identities: list[str] = []
    sanitized: list[str] = []
    for index, value in enumerate(cdp_urls):
        public_url = sanitize_url(value)
        sanitized.append(public_url)
        try:
            parsed = urlsplit(value)
            valid = parsed.scheme.casefold() in {"http", "https", "ws", "wss"} and bool(parsed.hostname)
            identity = _endpoint_identity(value) if valid else ""
        except ValueError:
            valid = False
            identity = ""
        if not valid:
            return sanitized, {
                "code": "PREFLIGHT_INVALID_CDP_URL",
                "message": f"worker {index} 的 CDP URL 非法",
                "worker_id": index,
                "endpoint": public_url,
            }
        identities.append(identity)
    if len(set(identities)) != len(identities):
        return sanitized, {
            "code": "PREFLIGHT_DUPLICATE_CDP_URL",
            "message": "CDP URL 必须指向互不重复的浏览器端点",
        }
    return sanitized, None


def _check_output_writable(output_dir: Path) -> dict[str, Any]:
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
        if not output_dir.is_dir():
            raise NotADirectoryError(str(output_dir))
        logs_dir = output_dir / "logs"
        logs_dir.mkdir(parents=True, exist_ok=True)
        if not logs_dir.is_dir():
            raise NotADirectoryError(str(logs_dir))
        for directory in (output_dir, logs_dir):
            mode = directory.stat().st_mode
            if mode & (stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH) == 0:
                raise PermissionError(f"目录权限位不允许写入: {directory}")
            with tempfile.NamedTemporaryFile(
                "w",
                encoding="utf-8",
                dir=directory,
                prefix=".preflight-write-",
                suffix=".tmp",
                delete=False,
            ) as handle:
                handle.write("ok")
                handle.flush()
                os.fsync(handle.fileno())
                temporary = Path(handle.name)
            destination = temporary.with_suffix(".verified")
            temporary.replace(destination)
            destination.unlink()
    except Exception as exc:
        return {
            "ok": False,
            "code": "PREFLIGHT_OUTPUT_NOT_WRITABLE",
            "message": sanitize_exception(exc),
            "path": str(output_dir),
        }
    return {"ok": True, "path": str(output_dir.resolve()), "atomic_replace": True}


def _probe_existing_pages(browser: Any) -> dict[str, Any]:
    contexts = list(browser.contexts)
    if not contexts:
        raise PreflightProbeError("PREFLIGHT_NO_CONTEXT", "浏览器不存在可用 BrowserContext")
    existing_page_count = sum(
        1
        for context in contexts
        for page in context.pages
        if not page.is_closed()
    )
    context = contexts[0]
    page: Any = None
    cdp: Any = None
    missing: list[str] = []
    version: dict[str, Any] = {}
    probe_error: BaseException | None = None
    try:
        try:
            page = context.new_page()
        except Exception as exc:
            raise PreflightProbeError(
                "PREFLIGHT_NO_PAGE",
                f"首个 BrowserContext 无法创建探测 Page: {sanitize_exception(exc)}",
            ) from exc
        try:
            dom_value = page.evaluate(
                "() => ({readyState: document.readyState, hasDocument: Boolean(document.documentElement)})"
            )
            if not dom_value.get("hasDocument"):
                missing.append("page_dom")
        except Exception:
            missing.append("page_dom")
        try:
            cdp = context.new_cdp_session(page)
        except Exception:
            missing.extend(("cdp_dom", "cdp_accessibility", "cdp_browser_version"))
        if cdp is not None:
            try:
                document = cdp.send("DOM.getDocument", {"depth": 0})
                if not document.get("root"):
                    missing.append("cdp_dom")
            except Exception:
                missing.append("cdp_dom")
            try:
                ax_tree = cdp.send("Accessibility.getFullAXTree", {"depth": 1})
                if not isinstance(ax_tree.get("nodes"), list):
                    missing.append("cdp_accessibility")
            except Exception:
                missing.append("cdp_accessibility")
            try:
                version = cdp.send("Browser.getVersion")
            except Exception:
                missing.append("cdp_browser_version")
        if not browser.is_connected():
            raise PreflightProbeError("PREFLIGHT_CDP_DISCONNECTED", "能力探测期间 CDP 断开")
        if missing:
            raise PreflightProbeError(
                "PREFLIGHT_MISSING_CAPABILITY",
                f"首个 BrowserContext 的临时 Page 缺少必要能力: {','.join(sorted(set(missing)))}",
            )
        product = str(version.get("product") or browser.version or "unknown")
        return {
            "chrome": product,
            "chrome_major": product.split("/", 1)[-1].split(".", 1)[0],
            "context_count": len(contexts),
            "page_count": existing_page_count,
            "selected_context_index": 0,
            "selected_page": "temporary_about_blank",
        }
    except BaseException as exc:
        probe_error = exc
        raise
    finally:
        if cdp is not None:
            try:
                cdp.detach()
            except Exception:
                pass
        if page is not None:
            try:
                page.close(run_before_unload=False)
            except Exception as exc:
                if probe_error is None:
                    raise PreflightProbeError(
                        "PREFLIGHT_TEMP_PAGE_CLEANUP_FAILED",
                        f"探测 Page 无法关闭: {sanitize_exception(exc)}",
                    ) from exc


def _probe_cdp(cdp_url: str, worker_id: int) -> dict[str, Any]:
    from playwright.sync_api import sync_playwright

    playwright = sync_playwright().start()
    browser: Any = None
    try:
        headers: dict[str, str] = {}
        token = parse_qs(urlsplit(cdp_url).query).get("access_token", [None])[0]
        if token:
            headers["X-Access-Token"] = token
        kwargs: dict[str, Any] = {"timeout": 8_000}
        if headers:
            kwargs["headers"] = headers
        try:
            browser = playwright.chromium.connect_over_cdp(cdp_url, **kwargs)
        except Exception as exc:
            raise PreflightProbeError("PREFLIGHT_CDP_CONNECT_FAILED", sanitize_exception(exc)) from exc
        if not browser.is_connected():
            raise PreflightProbeError("PREFLIGHT_CDP_DISCONNECTED", "CDP 连接建立后立即断开")
        page_probe = _probe_existing_pages(browser)
        return {
            "worker_id": worker_id,
            "status": "ok",
            "code": "PREFLIGHT_OK",
            "endpoint": sanitize_url(cdp_url),
            "browser": "chromium",
            **page_probe,
            "transport": "playwright_cdp",
            "context_mode": "shared_existing",
            "capabilities": {
                "browser": True,
                "context": True,
                "page": True,
                "dom": True,
                "accessibility": True,
                "cdp": True,
            },
        }
    finally:
        try:
            playwright.stop()
        except Exception:
            pass


def _probe_process_entry(
    probe: Callable[[str, int], dict[str, Any]],
    cdp_url: str,
    worker_id: int,
    connection: Any,
) -> None:
    try:
        try:
            worker = dict(probe(cdp_url, worker_id))
            worker["worker_id"] = worker_id
            worker["endpoint"] = sanitize_url(cdp_url)
            if worker.get("message"):
                worker["message"] = sanitize_error_text(str(worker["message"]))
        except PreflightProbeError as exc:
            worker = {
                "worker_id": worker_id,
                "status": "error",
                "code": exc.code,
                "endpoint": sanitize_url(cdp_url),
                "message": sanitize_exception(exc),
            }
        except BaseException as exc:
            worker = {
                "worker_id": worker_id,
                "status": "error",
                "code": "PREFLIGHT_PROBE_FAILED",
                "endpoint": sanitize_url(cdp_url),
                "message": sanitize_exception(exc),
            }
        connection.send(worker)
    finally:
        connection.close()


def _stop_probe_process(process: Any) -> None:
    try:
        if process.is_alive():
            process.terminate()
    except Exception:
        pass
    try:
        process.join(timeout=1.0)
    except Exception:
        pass
    try:
        if process.is_alive() and hasattr(process, "kill"):
            process.kill()
            process.join(timeout=1.0)
    except Exception:
        pass


def _run_probes(
    cdp_urls: list[str],
    probe: Callable[[str, int], dict[str, Any]],
    timeout_seconds: float,
) -> list[dict[str, Any]]:
    context = mp.get_context("spawn")
    states: dict[int, dict[str, Any]] = {}
    workers: list[dict[str, Any]] = []
    for worker_id, cdp_url in enumerate(cdp_urls):
        receive, send = context.Pipe(duplex=False)
        process = context.Process(
            target=_probe_process_entry,
            args=(probe, cdp_url, worker_id, send),
        )
        try:
            process.start()
        except Exception as exc:
            receive.close()
            send.close()
            workers.append(
                {
                    "worker_id": worker_id,
                    "status": "error",
                    "code": "PREFLIGHT_PROBE_FAILED",
                    "endpoint": sanitize_url(cdp_url),
                    "message": sanitize_exception(exc),
                }
            )
            continue
        send.close()
        states[worker_id] = {
            "process": process,
            "connection": receive,
            "started": time.monotonic(),
            "endpoint": sanitize_url(cdp_url),
        }

    try:
        while states:
            now = time.monotonic()
            for worker_id, state in list(states.items()):
                process = state["process"]
                connection = state["connection"]
                if connection.poll():
                    try:
                        worker = connection.recv()
                    except (EOFError, OSError) as exc:
                        worker = {
                            "worker_id": worker_id,
                            "status": "error",
                            "code": "PREFLIGHT_PROBE_FAILED",
                            "endpoint": state["endpoint"],
                            "message": sanitize_exception(exc),
                        }
                    workers.append(worker)
                    connection.close()
                    process.join(timeout=1.0)
                    if process.is_alive():
                        _stop_probe_process(process)
                    del states[worker_id]
                    continue
                if now - float(state["started"]) >= timeout_seconds:
                    _stop_probe_process(process)
                    connection.close()
                    workers.append(
                        {
                            "worker_id": worker_id,
                            "status": "error",
                            "code": "PREFLIGHT_PROBE_TIMEOUT",
                            "endpoint": state["endpoint"],
                            "message": f"CDP 能力探测超过 {timeout_seconds:g} 秒硬截止",
                        }
                    )
                    del states[worker_id]
                    continue
                if not process.is_alive():
                    process.join(timeout=0.1)
                    if connection.poll(0.05):
                        continue
                    connection.close()
                    workers.append(
                        {
                            "worker_id": worker_id,
                            "status": "error",
                            "code": "PREFLIGHT_PROBE_FAILED",
                            "endpoint": state["endpoint"],
                            "message": f"探测进程退出且未返回结果，退出码 {process.exitcode}",
                        }
                    )
                    del states[worker_id]
            if states:
                time.sleep(0.01)
    finally:
        for state in states.values():
            _stop_probe_process(state["process"])
            state["connection"].close()
    workers.sort(key=lambda value: int(value["worker_id"]))
    return workers


def _capability_signature(worker: dict[str, Any]) -> dict[str, Any]:
    return {
        "browser": worker.get("browser"),
        "chrome_major": worker.get("chrome_major"),
        "transport": worker.get("transport"),
        "context_mode": worker.get("context_mode"),
        "capabilities": worker.get("capabilities"),
    }


def run_preflight(
    cdp_urls: list[str],
    output_dir: Path,
    *,
    probe: Callable[[str, int], dict[str, Any]] = _probe_cdp,
    probe_timeout_seconds: float = 20.0,
) -> dict[str, Any]:
    report: dict[str, Any] = {
        "schema_version": PREFLIGHT_SCHEMA_VERSION,
        "status": "error",
        "code": "PREFLIGHT_NOT_RUN",
        "audited_at": datetime.now(timezone.utc).isoformat(),
        "model_requests": 0,
        "target_navigation": False,
        "runtime": {
            "python": platform.python_version(),
            "playwright": _package_version("playwright"),
        },
        "workers": [],
    }
    output = _check_output_writable(Path(output_dir))
    report["output"] = output
    if not output["ok"]:
        report.update({"code": output["code"], "message": output["message"]})
        return report
    sanitized_urls, url_error = validate_cdp_urls(cdp_urls)
    report["cdp_endpoints"] = sanitized_urls
    if url_error:
        report.update(url_error)
        _atomic_write_json(Path(output_dir) / "logs" / "preflight.json", report)
        return report
    if not math.isfinite(probe_timeout_seconds) or probe_timeout_seconds <= 0:
        report.update(
            {
                "code": "PREFLIGHT_INVALID_PROBE_TIMEOUT",
                "message": "CDP 探测超时必须是有限正数",
            }
        )
        _atomic_write_json(Path(output_dir) / "logs" / "preflight.json", report)
        return report

    workers = _run_probes(cdp_urls, probe, probe_timeout_seconds)
    report["workers"] = workers
    failed = [worker for worker in workers if worker.get("status") != "ok"]
    if failed:
        report.update(
            {
                "code": "PREFLIGHT_WORKER_FAILED",
                "message": f"{len(failed)}/{len(workers)} 个 CDP worker 未通过能力探测",
            }
        )
    else:
        signatures = [_capability_signature(worker) for worker in workers]
        if any(signature != signatures[0] for signature in signatures[1:]):
            report.update(
                {
                    "code": "PREFLIGHT_CAPABILITY_MISMATCH",
                    "message": "各 worker 的浏览器版本或能力不一致",
                }
            )
        else:
            report.update(
                {
                    "status": "ok",
                    "code": "PREFLIGHT_OK",
                    "message": "所有 worker 运行能力一致且可用",
                    "capability_signature": signatures[0],
                }
            )
    try:
        _atomic_write_json(Path(output_dir) / "logs" / "preflight.json", report)
    except Exception as exc:
        report.update(
            {
                "status": "error",
                "code": "PREFLIGHT_OUTPUT_NOT_WRITABLE",
                "message": sanitize_exception(exc),
            }
        )
    return report
