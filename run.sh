#!/bin/sh
set -eu
ROOT=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
PY="$ROOT/.venv/bin/python"
if [ ! -x "$PY" ]; then
    echo 'Run ./setup.sh first; see README.md.' >&2
    exit 1
fi
export PATH="$ROOT/.venv/bin:$PATH"
export HF_HOME="${HF_HOME:-$ROOT/.cache/huggingface}"
export TORCH_HOME="${TORCH_HOME:-$ROOT/.cache/torch}"
export TORCH_EXTENSIONS_DIR="${TORCH_EXTENSIONS_DIR:-$ROOT/.cache/torch-extensions}"
export PYTORCH_ENABLE_MPS_FALLBACK=1
export PYTHONUNBUFFERED=1
export MAX_JOBS="${MAX_JOBS:-4}"
exec "$PY" "$ROOT/run_local.py" "$@"

