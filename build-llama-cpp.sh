#!/bin/bash
#
# build-llama-cpp.sh
# ------------------
# Builds ggerganov/llama.cpp on Ubuntu Linux with CUDA support.
#
# • Automatically detects and installs missing dependencies
# • Detects CUDA installation and architecture
# • Re-usable: just run the script; it installs only what is missing
# • Pass CUDA_ARCH=<SM> to target a different GPU architecture
#   (defaults to auto-detect, fallback to 89 = Ada; GTX-1070 = 61, RTX-30-series = 86, etc.)

set -e  # Exit on any error

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CUDA_ARCH="${CUDA_ARCH:-}"
SKIP_BUILD="${SKIP_BUILD:-false}"

# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

log() {
    echo "-> $*"
}

log_ok() {
    echo "[OK] $*"
}

check_command() {
    command -v "$1" >/dev/null 2>&1
}

check_cuda() {
    if [ -d "/usr/local/cuda" ] || [ -d "/opt/cuda" ] || check_command nvcc; then
        return 0
    fi
    return 1
}

detect_cuda_arch() {
    if ! check_command nvidia-smi; then
        echo "89"  # Default fallback
        return
    fi

    # Try to detect GPU and map to compute capability
    local gpu_name
    gpu_name=$(nvidia-smi --query-gpu=name --format=csv,noheader,nounits 2>/dev/null | head -1 || echo "")

    case "$gpu_name" in
        *"RTX 40"*|*"RTX 4090"*|*"RTX 4080"*|*"RTX 4070"*|*"RTX 4060"*) echo "89" ;;
        *"RTX 30"*|*"RTX 3090"*|*"RTX 3080"*|*"RTX 3070"*|*"RTX 3060"*) echo "86" ;;
        *"RTX 20"*|*"RTX 2080"*|*"RTX 2070"*|*"RTX 2060"*) echo "75" ;;
        *"GTX 1080"*|*"GTX 1070"*|*"GTX 1060"*) echo "61" ;;
        *"Tesla V100"*) echo "70" ;;
        *"Tesla T4"*) echo "75" ;;
        *"A100"*) echo "80" ;;
        *) echo "89" ;;  # Default to modern architecture
    esac
}

install_dependencies() {
    local missing_deps=()

    # Check for required packages
    local deps=("build-essential" "cmake" "curl" "libcurl4-openssl-dev" "git")

    for dep in "${deps[@]}"; do
        if ! dpkg -l | grep -q "^ii  $dep "; then
            missing_deps+=("$dep")
        fi
    done

    # Check for optional but useful packages
    if ! check_command lspci; then
        missing_deps+=("pciutils")
    fi

    if [ ${#missing_deps[@]} -gt 0 ]; then
        log "Installing missing dependencies: ${missing_deps[*]}"
        sudo apt-get update
        sudo apt-get install -y "${missing_deps[@]}"
    fi
}

install_cuda() {
    if check_cuda; then
        return 0
    fi

    log "CUDA not found. Installing CUDA toolkit..."

    # Add NVIDIA package repository
    wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu$(lsb_release -rs | tr -d .)/x86_64/cuda-keyring_1.0-1_all.deb
    sudo dpkg -i cuda-keyring_1.0-1_all.deb
    sudo apt-get update
    sudo apt-get -y install cuda-toolkit

    # Add to PATH
    echo 'export PATH=/usr/local/cuda/bin:$PATH' >> ~/.bashrc
    echo 'export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
    export PATH=/usr/local/cuda/bin:$PATH
    export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
}

# ---------------------------------------------------------------------------
# Main routine
# ---------------------------------------------------------------------------

log "Checking and installing dependencies..."

# Install system dependencies
install_dependencies
log_ok "System dependencies"

# Check for CUDA
if check_cuda; then
    log_ok "CUDA Toolkit"
    if [ -z "$CUDA_ARCH" ]; then
        CUDA_ARCH=$(detect_cuda_arch)
        log "Auto-detected CUDA architecture: $CUDA_ARCH"
    fi
else
    log "CUDA not found - building CPU-only version"
    CUDA_ARCH=""
fi

if [ "$SKIP_BUILD" = "true" ]; then
    log "SKIP_BUILD set - done."
    exit 0
fi

# ---------------------------------------------------------------------------
# Clone & build ggerganov/llama.cpp
# ---------------------------------------------------------------------------

LLAMA_REPO="$SCRIPT_DIR/vendor/llama.cpp"
LLAMA_BUILD="$LLAMA_REPO/build"

if [ ! -d "$LLAMA_REPO" ]; then
    log "Cloning upstream llama.cpp into $LLAMA_REPO"
    git clone https://github.com/ggerganov/llama.cpp "$LLAMA_REPO"
else
    log "Updating existing llama.cpp in $LLAMA_REPO"
    git -C "$LLAMA_REPO" pull --ff-only
fi

# Configure build
mkdir -p "$LLAMA_BUILD"
cd "$LLAMA_BUILD"

CMAKE_ARGS=(
    "-DCMAKE_BUILD_TYPE=Release"
    "-DLLAMA_CURL=ON"
    "-DBUILD_SHARED_LIBS=OFF"
)

if [ -n "$CUDA_ARCH" ]; then
    log "Configuring llama.cpp with CUDA support (architecture: $CUDA_ARCH)..."
    CMAKE_ARGS+=(
        "-DGGML_CUDA=ON"
        "-DGGML_CUBLAS=ON"
        "-DGGML_CUDA_FA_ALL_QUANTS=ON"
        "-DCMAKE_CUDA_ARCHITECTURES=$CUDA_ARCH"
    )
else
    log "Configuring llama.cpp for CPU-only build..."
fi

log "Generating build configuration..."
cmake .. "${CMAKE_ARGS[@]}"

log "Building llama.cpp tools (Release)..."
cmake --build . --config Release --target llama-server llama-batched-bench llama-cli llama-bench --parallel

# Copy binaries to root for easy access
if [ -d "bin" ]; then
    cp bin/llama-* "$LLAMA_REPO/" 2>/dev/null || true
fi

log ""
log "Done! llama.cpp binaries are in: $LLAMA_BUILD/bin"
if [ -n "$CUDA_ARCH" ]; then
    log "Built with CUDA support for architecture: $CUDA_ARCH"
else
    log "Built for CPU-only (CUDA not available)"
fi
