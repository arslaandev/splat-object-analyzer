#!/usr/bin/env bash
set -euo pipefail
ROOT="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"
cd "$ROOT"
export HF_HOME="${HF_HOME:-$ROOT/.cache/huggingface}"
export TORCH_HOME="${TORCH_HOME:-$ROOT/.cache/torch}"
export TORCH_EXTENSIONS_DIR="${TORCH_EXTENSIONS_DIR:-$ROOT/.cache/torch-extensions}"
export MAX_JOBS="${MAX_JOBS:-4}"
export PYTHONUNBUFFERED=1
exec "$ROOT/.venv/bin/python" -m uvicorn api_server:app \
    --host "${SPLAT_API_HOST:-127.0.0.1}" --port "${SPLAT_API_PORT:-8000}" \
    --workers 1 --no-server-header --timeout-graceful-shutdown 30
