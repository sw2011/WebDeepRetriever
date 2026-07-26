from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

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
    parser.add_argument("--healthcheck", action="store_true")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if args.healthcheck:
        print(json.dumps({"status": "ok", "playwright_transport": "cdp", "max_workers": 8}))
        return
    if not args.input or not args.output or not args.cdp_url or not args.api_key:
        parser.error("必须提供 --input、--output、--cdp_url 和 --api_key（或对应环境变量）")
    summary = run_tasks(
        Path(args.input),
        Path(args.output),
        args.cdp_url,
        args.model,
        args.api_base,
        args.api_key,
        args.max_steps,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
