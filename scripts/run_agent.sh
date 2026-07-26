#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

INPUT="${WEBRETRIEVER_INPUT:-$PROJECT_DIR/data/example_tasks.json}"
OUTPUT="${WEBRETRIEVER_OUTPUT:-$PROJECT_DIR/output}"
MODEL="${WEBRETRIEVER_MODEL:-gpt-4.1-mini}"
API_BASE="${OPENAI_BASE_URL:-https://api.openai.com/v1}"
MAX_STEPS="${WEBRETRIEVER_MAX_STEPS:-100}"

if [[ -z "${OPENAI_API_KEY:-}" ]]; then
  echo "缺少 OPENAI_API_KEY" >&2
  exit 2
fi
if [[ -z "${WEBRETRIEVER_CDP_URLS:-}" ]]; then
  echo "缺少 WEBRETRIEVER_CDP_URLS（多个 URL 用逗号分隔）" >&2
  exit 2
fi

IFS=',' read -r -a CDP_URLS <<< "$WEBRETRIEVER_CDP_URLS"

exec "$PROJECT_DIR/.venv/bin/python" -m web_agent.cli \
  --input "$INPUT" \
  --output "$OUTPUT" \
  --cdp_url "${CDP_URLS[@]}" \
  --model "$MODEL" \
  --api_base "$API_BASE" \
  --api_key "$OPENAI_API_KEY" \
  --max_steps "$MAX_STEPS"
