#!/bin/bash
#
# llama-test-pr.sh
# ----------------
# Build a local llama.cpp test branch by merging one or more PR heads.
#
# Usage:
#   ./maintenance/llama-test-pr.sh 20275 [20280 ...]
#
# Env overrides:
#   LLAMA_REPO_URL   (default: https://github.com/ggml-org/llama.cpp)
#   LLAMA_REPO_DIR   (default: <repo>/vendor/llama.cpp-pr-test-<joined-prs>)
#   BASE_REF         (default: master)
#   TEST_BRANCH      (default: pr-test-<joined-prs>)
#   CUDA_ARCH        (default: auto-detect)
#   SKIP_BUILD       (default: false)

set -euo pipefail

if [ "$#" -lt 1 ]; then
    echo "Usage: $0 <pr-number> [<pr-number> ...]" >&2
    exit 1
fi

for _pr in "$@"; do
    if ! [[ "$_pr" =~ ^[0-9]+$ ]]; then
        echo "Invalid PR number: $_pr" >&2
        exit 1
    fi
done

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
PR_SET="$(IFS=-; echo "$*")"
LLAMA_REPO_URL="${LLAMA_REPO_URL:-https://github.com/ggml-org/llama.cpp}"
LLAMA_REPO_DIR="${LLAMA_REPO_DIR:-$REPO_DIR/vendor/llama.cpp-pr-test-${PR_SET}}"
BASE_REF="${BASE_REF:-master}"
TEST_BRANCH="${TEST_BRANCH:-pr-test-${PR_SET}}"
CUDA_ARCH="${CUDA_ARCH:-}"
SKIP_BUILD="${SKIP_BUILD:-false}"

log() { echo "-> $*"; }
log_ok() { echo "[OK] $*"; }
check_command() { command -v "$1" >/dev/null 2>&1; }

detect_cuda_arch() {
    if ! check_command nvidia-smi; then
        echo "89"
        return
    fi
    local gpu_name
    gpu_name=$(nvidia-smi --query-gpu=name --format=csv,noheader,nounits 2>/dev/null | head -1 || echo "")
    case "$gpu_name" in
        *"RTX 50"*|*"RTX 40"*|*"RTX 4090"*|*"RTX 4080"*|*"RTX 4070"*|*"RTX 4060"*) echo "89" ;;
        *"RTX 30"*|*"RTX 3090"*|*"RTX 3080"*|*"RTX 3070"*|*"RTX 3060"*) echo "86" ;;
        *"RTX 20"*|*"RTX 2080"*|*"RTX 2070"*|*"RTX 2060"*) echo "75" ;;
        *"GTX 1080"*|*"GTX 1070"*|*"GTX 1060"*) echo "61" ;;
        *"Tesla V100"*) echo "70" ;;
        *"Tesla T4"*) echo "75" ;;
        *"A100"*) echo "80" ;;
        *) echo "89" ;;
    esac
}

prepare_repo() {
    if [ ! -d "$LLAMA_REPO_DIR/.git" ]; then
        log "Cloning ${LLAMA_REPO_URL} into ${LLAMA_REPO_DIR}"
        git clone "$LLAMA_REPO_URL" "$LLAMA_REPO_DIR"
    else
        local current_origin
        current_origin="$(git -C "$LLAMA_REPO_DIR" remote get-url origin 2>/dev/null || echo "")"
        if [ "$current_origin" != "$LLAMA_REPO_URL" ]; then
            log "Updating origin remote to ${LLAMA_REPO_URL}"
            git -C "$LLAMA_REPO_DIR" remote set-url origin "$LLAMA_REPO_URL"
        fi
    fi

    log "Fetching latest refs"
    git -C "$LLAMA_REPO_DIR" fetch --all --tags

    log "Creating ${TEST_BRANCH} from origin/${BASE_REF}"
    git -C "$LLAMA_REPO_DIR" checkout -B "$TEST_BRANCH" "origin/${BASE_REF}"
}

merge_prs() {
    local pr
    for pr in "$@"; do
        log "Fetching PR #${pr}"
        git -C "$LLAMA_REPO_DIR" fetch origin "pull/${pr}/head:pr-${pr}"
        log "Merging pr-${pr} into ${TEST_BRANCH}"
        git -C "$LLAMA_REPO_DIR" merge --no-ff -m "merge(pr-${pr}): local test integration" "pr-${pr}"
    done
}

build_llama() {
    if [ -z "$CUDA_ARCH" ]; then
        CUDA_ARCH="$(detect_cuda_arch)"
        log "Auto-detected CUDA architecture: $CUDA_ARCH"
    fi

    local build_dir="$LLAMA_REPO_DIR/build"
    mkdir -p "$build_dir"
    cd "$build_dir"

    cmake .. \
      -DCMAKE_BUILD_TYPE=Release \
      -DLLAMA_CURL=ON \
      -DBUILD_SHARED_LIBS=OFF \
      -DGGML_CUDA=ON \
      -DGGML_NATIVE=ON \
      -DGGML_LTO=ON \
      -DGGML_OPENMP=ON \
      -DGGML_CUDA_GRAPHS=ON \
      -DGGML_CUDA_FA_ALL_QUANTS=ON \
      -DCMAKE_CUDA_ARCHITECTURES="$CUDA_ARCH"

    cmake --build . --config Release --target llama-server llama-batched-bench llama-cli llama-bench llama-sweep-bench llama-fit-params --parallel
}

prepare_repo
merge_prs "$@"
log_ok "Merged PRs into ${TEST_BRANCH}"

if [ "$SKIP_BUILD" = "true" ]; then
    log "SKIP_BUILD set - done."
    exit 0
fi

build_llama
log "Done! Test build in: ${LLAMA_REPO_DIR}/build/bin"
