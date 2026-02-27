#!/bin/bash
#
# build-llama-cpp-vulkan.sh
# -------------------------
# Builds ggerganov/llama.cpp on Linux/macOS with Vulkan support.
#
# • Automatically detects OS and installs missing dependencies
# • Detects Vulkan SDK installation
# • Re-usable: just run the script; it installs only what is missing
# • Builds into vendor/llama.cpp/build-vulkan for easy identification

set -e  # Exit on any error

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
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

detect_os() {
    case "$(uname -s)" in
        Linux*)
            # Check for Arch-based systems first
            if [ -f /etc/arch-release ] || [ -f /etc/artix-release ]; then
                echo "arch"
                return
            fi

            if [ -f /etc/os-release ]; then
                . /etc/os-release
                # Check ID_LIKE for Arch-based distributions
                if echo "$ID_LIKE" | grep -qi "arch"; then
                    echo "arch"
                elif echo "$ID" | grep -qi "arch"; then
                    echo "arch"
                else
                    echo "$ID"
                fi
            elif [ -f /etc/lsb-release ]; then
                . /etc/lsb-release
                echo "$DISTRIB_ID" | tr '[:upper:]' '[:lower:]'
            else
                echo "unknown"
            fi
            ;;
        Darwin*)
            echo "macos"
            ;;
        *)
            echo "unknown"
            ;;
    esac
}

check_vulkan() {
    # Check for Vulkan SDK or system installation
    if [ -n "$VULKAN_SDK" ] && [ -d "$VULKAN_SDK" ]; then
        return 0
    fi

    # Check for system-wide Vulkan headers
    if [ -f "/usr/include/vulkan/vulkan.h" ] || [ -f "/usr/local/include/vulkan/vulkan.h" ]; then
        return 0
    fi

    # Check for vulkaninfo command
    if check_command vulkaninfo; then
        return 0
    fi

    return 1
}

install_dependencies_ubuntu() {
    local missing_deps=()

    # Check for required packages
    local deps=("build-essential" "cmake" "curl" "libcurl4-openssl-dev" "git")

    for dep in "${deps[@]}"; do
        if ! dpkg -l | grep -q "^ii  $dep "; then
            missing_deps+=("$dep")
        fi
    done

    # Vulkan dependencies
    local vulkan_deps=("libvulkan-dev" "vulkan-tools" "libshaderc-dev" "glslang-tools")
    for dep in "${vulkan_deps[@]}"; do
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

install_dependencies_arch() {
    local missing_deps=()

    # Check for required packages on Arch
    local deps=("base-devel" "cmake" "curl" "git" "vulkan-headers" "vulkan-icd-loader" "vulkan-tools" "shaderc" "glslang")

    for dep in "${deps[@]}"; do
        if ! pacman -Q "$dep" >/dev/null 2>&1; then
            missing_deps+=("$dep")
        fi
    done

    # Check for optional but useful packages
    if ! check_command lspci; then
        if ! pacman -Q pciutils >/dev/null 2>&1; then
            missing_deps+=("pciutils")
        fi
    fi

    if [ ${#missing_deps[@]} -gt 0 ]; then
        log "Installing missing dependencies: ${missing_deps[*]}"
        sudo pacman -Sy --noconfirm --needed "${missing_deps[@]}"
    fi
}

install_dependencies_macos() {
    if ! check_command brew; then
        log "ERROR: Homebrew not found. Please install Homebrew first: https://brew.sh"
        exit 1
    fi

    local missing_deps=()
    local deps=("cmake" "curl" "molten-vk" "vulkan-headers" "shaderc" "glslang")

    for dep in "${deps[@]}"; do
        if ! brew list "$dep" >/dev/null 2>&1; then
            missing_deps+=("$dep")
        fi
    done

    if [ ${#missing_deps[@]} -gt 0 ]; then
        log "Installing missing dependencies: ${missing_deps[*]}"
        brew install "${missing_deps[@]}"
    fi
}

install_dependencies() {
    local os_type
    os_type=$(detect_os)

    log "Detected OS: $os_type"

    case "$os_type" in
        ubuntu|debian|linuxmint|pop)
            install_dependencies_ubuntu
            ;;
        arch|manjaro|endeavouros|cachyos|garuda|artix)
            install_dependencies_arch
            ;;
        macos)
            install_dependencies_macos
            ;;
        *)
            log "WARNING: Unsupported OS ($os_type). Attempting Ubuntu-style package installation..."
            install_dependencies_ubuntu
            ;;
    esac
}

setup_vulkan_env() {
    local os_type
    os_type=$(detect_os)

    if [ "$os_type" = "macos" ]; then
        # Set up MoltenVK environment for macOS
        export VULKAN_SDK="$(brew --prefix molten-vk)"
        export VK_ICD_FILENAMES="$VULKAN_SDK/share/vulkan/icd.d/MoltenVK_icd.json"
        export VK_LAYER_PATH="$VULKAN_SDK/share/vulkan/explicit_layer.d"
        log "Set up MoltenVK environment for macOS"
    fi
}

# ---------------------------------------------------------------------------
# Main routine
# ---------------------------------------------------------------------------

log "Checking and installing dependencies..."

# Install system dependencies
install_dependencies
log_ok "System dependencies"

# Set up Vulkan environment
setup_vulkan_env

# Check for Vulkan
if check_vulkan; then
    log_ok "Vulkan SDK/Runtime"
else
    log "WARNING: Vulkan not fully detected. Build may fail."
    log "If build fails, please install Vulkan SDK manually:"
    log "  - Linux: https://vulkan.lunarg.com/sdk/home"
    log "  - macOS: brew install molten-vk"
fi

if [ "$SKIP_BUILD" = "true" ]; then
    log "SKIP_BUILD set - done."
    exit 0
fi

# ---------------------------------------------------------------------------
# Clone & build ggerganov/llama.cpp with Vulkan
# ---------------------------------------------------------------------------

LLAMA_REPO="$SCRIPT_DIR/vendor/llama.cpp"
LLAMA_BUILD="$LLAMA_REPO/build-vulkan"

if [ ! -d "$LLAMA_REPO" ]; then
    log "Cloning upstream llama.cpp into $LLAMA_REPO"
    git clone https://github.com/ggerganov/llama.cpp "$LLAMA_REPO"
else
    log "Updating existing llama.cpp in $LLAMA_REPO"
    git -C "$LLAMA_REPO" pull --ff-only || log "WARNING: Could not pull latest changes (may have local modifications)"
fi

# Configure build
mkdir -p "$LLAMA_BUILD"
cd "$LLAMA_BUILD"

CMAKE_ARGS=(
    "-DCMAKE_BUILD_TYPE=Release"
    "-DLLAMA_CURL=ON"
    "-DBUILD_SHARED_LIBS=OFF"
    "-DGGML_VULKAN=ON"
)

log "Configuring llama.cpp with Vulkan support..."
log "Generating build configuration..."
cmake .. "${CMAKE_ARGS[@]}"

log "Building llama.cpp tools (Release) with Vulkan..."
cmake --build . --config Release --target llama-server llama-batched-bench llama-cli llama-bench --parallel

# Copy binaries to root for easy access
if [ -d "bin" ]; then
    cp bin/llama-* "$LLAMA_REPO/" 2>/dev/null || true
fi

log ""
log "Done! llama.cpp Vulkan binaries are in: $LLAMA_BUILD/bin"
log "Built with Vulkan support"
log ""
log "To use, set LLAMA_CPP_VULKAN_PATH in your run script to:"
log "  $LLAMA_BUILD/bin/llama-server"
