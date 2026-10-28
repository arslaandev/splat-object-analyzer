#!/usr/bin/env bash
set -euo pipefail
ROOT="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"

usage() {
    cat <<'EOF'
Usage: ./setup.sh [mac|cuda]

Requires Python 3.11 or 3.12, Git, and a GPU compiler toolchain.
Mac: Xcode command-line tools. NVIDIA: CUDA-enabled torch/torchvision in .venv,
plus nvcc matching that PyTorch installation. See README.md.
PYTHON_BIN can select Python; uv is used when available, otherwise pip.
EOF
}
if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then usage; exit 0; fi
if [[ $# -gt 1 ]]; then usage >&2; exit 2; fi
BACKEND="${1:-}"
if [[ -z "$BACKEND" ]]; then
    if [[ "$(uname -s)" == Darwin && "$(uname -m)" == arm64 ]]; then BACKEND=mac; else BACKEND=cuda; fi
fi
if [[ "$BACKEND" != mac && "$BACKEND" != cuda ]]; then usage >&2; exit 2; fi
if [[ "$BACKEND" == mac ]]; then
    [[ "$(uname -s)" == Darwin && "$(uname -m)" == arm64 ]] || { echo 'The mac backend requires Apple Silicon.' >&2; exit 1; }
    xcode-select -p >/dev/null 2>&1 || { echo 'Install Xcode command-line tools: xcode-select --install' >&2; exit 1; }
fi
PYTHON_BIN="${PYTHON_BIN:-python3.12}"
if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then PYTHON_BIN=python3; fi
"$PYTHON_BIN" -c 'import sys; assert (3, 11) <= sys.version_info[:2] <= (3, 12), "Use Python 3.11 or 3.12"'
command -v git >/dev/null 2>&1 || { echo 'Install Git before setup.' >&2; exit 1; }
export UV_CACHE_DIR="${UV_CACHE_DIR:-$ROOT/.cache/uv}"
export PIP_CACHE_DIR="${PIP_CACHE_DIR:-$ROOT/.cache/pip}"
export TORCH_EXTENSIONS_DIR="${TORCH_EXTENSIONS_DIR:-$ROOT/.cache/torch-extensions}"
export MAX_JOBS="${MAX_JOBS:-4}"
if [[ ! -x "$ROOT/.venv/bin/python" ]]; then
    if command -v uv >/dev/null 2>&1; then uv venv "$ROOT/.venv" --python "$PYTHON_BIN";
    else "$PYTHON_BIN" -m venv "$ROOT/.venv"; fi
fi
VENV_PY="$ROOT/.venv/bin/python"
install_packages() {
    if command -v uv >/dev/null 2>&1; then uv pip install --python "$VENV_PY" "$@";
    else "$VENV_PY" -m pip install "$@"; fi
}

if [[ "$BACKEND" == cuda ]]; then
    "$VENV_PY" -c 'import torch, torchvision; assert torch.cuda.is_available(), "Install CUDA-enabled PyTorch into .venv first; see README.md"'
    command -v nvcc >/dev/null 2>&1 || { echo 'nvcc is required to compile gsplat. Install the matching CUDA toolkit.' >&2; exit 1; }
    export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-$("$VENV_PY" -c 'import torch; a,b=torch.cuda.get_device_capability(); print(f"{a}.{b}")')}"
fi
install_packages setuptools==81.0.0 wheel==0.48.0
install_packages -r "$ROOT/requirements-$BACKEND.txt"

checkout_dependency() {
    local url="$1" destination="$2" revision="$3"
    if [[ ! -d "$destination/.git" ]]; then
        if [[ -e "$destination" ]]; then echo "Unmanaged dependency directory: $destination" >&2; exit 1; fi
        git clone --filter=blob:none "$url" "$destination"
    fi
    git -C "$destination" diff --quiet && git -C "$destination" diff --cached --quiet || {
        echo "Dependency has local edits; preserve them before setup: $destination" >&2; exit 1;
    }
    git -C "$destination" checkout --detach "$revision"
}
mkdir -p "$ROOT/third_party"
checkout_dependency https://github.com/xinyu1205/recognize-anything.git \
    "$ROOT/third_party/recognize-anything" 7cb804a8609e9f4b1a50b7f31436d2df40bb9481
install_packages --no-deps -e "$ROOT/third_party/recognize-anything"
if [[ "$BACKEND" == mac ]]; then
    checkout_dependency https://github.com/iffyloop/gsplat-mps.git \
        "$ROOT/third_party/gsplat-mps" 10372c263f8e54427183de8a5855a62f94c2f206
    git -C "$ROOT/third_party/gsplat-mps" submodule update --init --recursive
    # Use the tested JIT path: it resolves the Metal shader via its absolute path
    # and works with command-line tools without a precompiled .metallib.
    BUILD_NO_MPS=1 install_packages --no-build-isolation -e "$ROOT/third_party/gsplat-mps"
else
    install_packages --no-build-isolation gsplat==1.5.3
fi
"$VENV_PY" -c 'import torch, ram; print("Setup complete; torch", torch.__version__)'
echo "Analyze: $ROOT/analyze-splat.sh /path/to/scene.spz"

