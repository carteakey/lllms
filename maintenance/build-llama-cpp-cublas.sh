#!/bin/bash
#
# build-llama-cpp-cublas.sh
# -------------------------
# Builds llama.cpp with GGML_CUDA_FORCE_CUBLAS=ON into a separate build
# directory (build-cublas/) so the default build is untouched.
#
# Use this for models with non-standard quants (e.g. mxfp4) where cuBLAS
# tensor core paths outperform the default GGML CUDA kernels.
#
# Output binaries: vendor/llama.cpp/build-cublas/bin/llama-*
#
# Override binary path in bench scripts via:
#   LLAMA_BENCH=vendor/llama.cpp/build-cublas/bin/llama-bench \
#     ./bench-models/bench-llama-cpp-gpt-oss-120b.sh
#
# Pass CUDA_ARCH=<SM> to target a different GPU architecture.
# Defaults to auto-detect, fallback to 89 (Ada / RTX 40-series).

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CUDA_ARCH="${CUDA_ARCH:-}"

log()    { echo "-> $*"; }
log_ok() { echo "[OK] $*"; }

# ---------------------------------------------------------------------------
# CUDA arch detection
# ---------------------------------------------------------------------------

detect_cuda_arch() {
    if ! command -v nvidia-smi >/dev/null 2>&1; then
        echo "89"
        return
    fi
    local gpu_name
    gpu_name=$(nvidia-smi --query-gpu=name --format=csv,noheader,nounits 2>/dev/null | head -1 || echo "")
    case "$gpu_name" in
        *"RTX 40"*|*"RTX 4090"*|*"RTX 4080"*|*"RTX 4070"*|*"RTX 4060"*) echo "89" ;;
        *"RTX 30"*|*"RTX 3090"*|*"RTX 3080"*|*"RTX 3070"*|*"RTX 3060"*) echo "86" ;;
        *"RTX 20"*|*"RTX 2080"*|*"RTX 2070"*|*"RTX 2060"*)               echo "75" ;;
        *"GTX 1080"*|*"GTX 1070"*|*"GTX 1060"*)                           echo "61" ;;
        *"Tesla V100"*)                                                     echo "70" ;;
        *"Tesla T4"*)                                                       echo "75" ;;
        *"A100"*)                                                           echo "80" ;;
        *)                                                                  echo "89" ;;
    esac
}

if [ -z "$CUDA_ARCH" ]; then
    CUDA_ARCH=$(detect_cuda_arch)
    log "Auto-detected CUDA architecture: sm${CUDA_ARCH}"
fi

# ---------------------------------------------------------------------------
# Repo check
# ---------------------------------------------------------------------------

LLAMA_REPO="$SCRIPT_DIR/vendor/llama.cpp"
LLAMA_BUILD="$LLAMA_REPO/build-cublas"

if [ ! -d "$LLAMA_REPO" ]; then
    log "ERROR: llama.cpp repo not found at $LLAMA_REPO"
    log "Run maintenance/build-llama-cpp.sh first to clone it."
    exit 1
fi

log "llama.cpp repo: $LLAMA_REPO"
log "Build dir:      $LLAMA_BUILD"
log "CUDA arch:      sm${CUDA_ARCH}"
log "Extra flags:    GGML_CUDA_FORCE_CUBLAS=ON, GGML_CUDA_FORCE_DMMV=OFF"
echo

# ---------------------------------------------------------------------------
# Configure & build
# ---------------------------------------------------------------------------

mkdir -p "$LLAMA_BUILD"
cd "$LLAMA_BUILD"

cmake .. \
    -DCMAKE_BUILD_TYPE=Release \
    -DLLAMA_CURL=ON \
    -DBUILD_SHARED_LIBS=OFF \
    -DGGML_CUDA=ON \
    -DGGML_NATIVE=ON \
    -DGGML_LTO=ON \
    -DGGML_OPENMP=ON \
    -DGGML_CUDA_GRAPHS=ON \
    -DGGML_VULKAN=OFF \
    -DGGML_RPC=OFF \
    -DGGML_BLAS=OFF \
    -DGGML_CUDA_F16=ON \
    -DGGML_CCACHE=OFF \
    -DGGML_CUDA_FA_ALL_QUANTS=ON \
    -DGGML_CUDA_FORCE_CUBLAS=ON \
    -DGGML_CUDA_FORCE_DMMV=OFF \
    -DCMAKE_CUDA_ARCHITECTURES="$CUDA_ARCH"

cmake --build . --config Release \
    --target llama-server llama-bench llama-cli llama-fit-params \
    --parallel

# ---------------------------------------------------------------------------
# Done
# ---------------------------------------------------------------------------

log ""
log_ok "Build complete: $LLAMA_BUILD/bin/"
log "To bench gpt-oss-120b with cuBLAS:"
log "  LLAMA_BENCH=${LLAMA_BUILD}/bin/llama-bench \\"
log "    bash bench-models/bench-llama-cpp-gpt-oss-120b.sh"
log ""
log "To run server with cuBLAS:"
log "  LLAMA_SERVER=${LLAMA_BUILD}/bin/llama-server \\"
log "    bash run-models/run-llama-cpp-gpt-oss-120b.sh"
