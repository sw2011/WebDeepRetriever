from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from .preflight import run_preflight
from .runner import run_tasks


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="WebRetriever Protocol III verified web agent")
    parser.add_argument("--input", default=os.getenv("WEBRETRIEVER_INPUT"))
    parser.add_argument("--output", default=os.getenv("WEBRETRIEVER_OUTPUT"))
    parser.add_argument(
        "--cdp_url",
        nargs="+",
        default=[value for value in os.getenv("WEBRETRIEVER_CDP_URLS", "").split(",") if value],
    )
    parser.add_argument("--model", default=os.getenv("WEBRETRIEVER_MODEL", "gpt-4.1-mini"))
    parser.add_argument("--api_base", default=os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1"))
    parser.add_argument("--api_key", default=os.getenv("OPENAI_API_KEY"))
    parser.add_argument("--max_steps", type=int, default=int(os.getenv("WEBRETRIEVER_MAX_STEPS", "100")))
    parser.add_argument(
        "--worker_watchdog_seconds",
        type=float,
        default=float(os.getenv("WEBRETRIEVER_WORKER_WATCHDOG_SECONDS", "900")),
    )
    parser.add_argument("--force_rerun", action="store_true")
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--healthcheck", action="store_true", help="--preflight 的兼容别名")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if args.healthcheck or args.preflight:
        output_dir = Path(args.output or os.getenv("WEBRETRIEVER_OUTPUT", "output"))
        report = run_preflight(args.cdp_url, output_dir)
        print(json.dumps(report, ensure_ascii=False, indent=2))
        if report["status"] != "ok":
            raise SystemExit(2)
        return
    if not args.input or not args.output or not args.cdp_url or not args.api_key:
        parser.error("必须提供 --input、--output、--cdp_url 和 --api_key（或对应环境变量）")
    report = run_preflight(args.cdp_url, Path(args.output))
    if report["status"] != "ok":
        print(json.dumps(report, ensure_ascii=False, indent=2))
        raise SystemExit(2)
    summary = run_tasks(
        Path(args.input),
        Path(args.output),
        args.cdp_url,
        args.model,
        args.api_base,
        args.api_key,
        args.max_steps,
        force_rerun=args.force_rerun,
        worker_watchdog_seconds=args.worker_watchdog_seconds,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
