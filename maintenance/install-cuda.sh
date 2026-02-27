#!/bin/bash
#
# install-cuda.sh
# ---------------
# Installs CUDA toolkit on Ubuntu/Arch Linux systems
#
# • Automatically detects OS (Ubuntu/Arch) and uses appropriate installation method
# • For Ubuntu: Downloads and installs CUDA 13.0 from NVIDIA repositories
# • For Arch: Installs CUDA from official Arch repositories
# • Re-usable: checks if CUDA is already installed before proceeding

set -e  # Exit on any error

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
    # Check for Arch-based systems first (many have /etc/os-release)
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
}

check_cuda() {
    if [ -d "/usr/local/cuda" ] || [ -d "/opt/cuda" ] || check_command nvcc; then
        return 0
    fi
    return 1
}

# ---------------------------------------------------------------------------
# Ubuntu CUDA installation
# ---------------------------------------------------------------------------

install_cuda_ubuntu() {
    log "Installing CUDA 13.0 for Ubuntu 24.04..."

    # Download and install pin file
    if [ ! -f cuda-ubuntu2404.pin ]; then
        log "Downloading CUDA repository pin file..."
        wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/cuda-ubuntu2404.pin
    fi
    sudo mv cuda-ubuntu2404.pin /etc/apt/preferences.d/cuda-repository-pin-600

    # Check if local installer exists
    local cuda_deb="cuda-repo-ubuntu2404-13-0-local_13.0.0-580.65.06-1_amd64.deb"
    if [ ! -f "$cuda_deb" ]; then
        log "Local CUDA installer not found. You need to download it from:"
        log "https://developer.download.nvidia.com/compute/cuda/13.0.0/local_installers/cuda-repo-ubuntu2404-13-0-local_13.0.0-580.65.06-1_amd64.deb"
        log ""
        log "Would you like to download it now? (requires ~4GB)"
        read -p "Download CUDA installer? [y/N]: " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            log "Downloading CUDA 13.0 local installer..."
            wget https://developer.download.nvidia.com/compute/cuda/13.0.0/local_installers/cuda-repo-ubuntu2404-13-0-local_13.0.0-580.65.06-1_amd64.deb
        else
            log "Skipping download. Please download manually and run this script again."
            exit 1
        fi
    fi

    log "Installing CUDA repository package..."
    sudo dpkg -i "$cuda_deb"

    log "Copying CUDA keyring..."
    sudo cp /var/cuda-repo-ubuntu2404-13-0-local/cuda-*-keyring.gpg /usr/share/keyrings/

    log "Updating package list..."
    sudo apt-get update

    log "Installing CUDA toolkit 13.0..."
    sudo apt-get -y install cuda-toolkit-13-0

    # Add to PATH
    if ! grep -q "/usr/local/cuda/bin" ~/.bashrc; then
        echo 'export PATH=/usr/local/cuda/bin:$PATH' >> ~/.bashrc
        echo 'export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
    fi
    export PATH=/usr/local/cuda/bin:$PATH
    export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
}

# ---------------------------------------------------------------------------
# Arch CUDA installation
# ---------------------------------------------------------------------------

install_cuda_arch() {
    log "Installing CUDA for Arch Linux..."

    # Update package database
    sudo pacman -Sy

    # Install CUDA
    log "Installing CUDA package..."
    sudo pacman -S --noconfirm --needed cuda

    # Add to PATH
    if ! grep -q "/opt/cuda/bin" ~/.bashrc; then
        echo 'export PATH=/opt/cuda/bin:$PATH' >> ~/.bashrc
        echo 'export LD_LIBRARY_PATH=/opt/cuda/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
    fi
    export PATH=/opt/cuda/bin:$PATH
    export LD_LIBRARY_PATH=/opt/cuda/lib64:$LD_LIBRARY_PATH
}

# ---------------------------------------------------------------------------
# Main routine
# ---------------------------------------------------------------------------

log "CUDA Installation Script"
log "========================"

# Check if CUDA is already installed
if check_cuda; then
    log_ok "CUDA is already installed"
    nvcc --version 2>/dev/null || true
    exit 0
fi

# Detect OS
OS_TYPE=$(detect_os)
log "Detected OS: $OS_TYPE"

case "$OS_TYPE" in
    ubuntu|debian|linuxmint|pop)
        install_cuda_ubuntu
        ;;
    arch|manjaro|endeavouros|cachyos|garuda|artix)
        install_cuda_arch
        ;;
    *)
        log "ERROR: Unsupported OS ($OS_TYPE)"
        log "This script supports Ubuntu/Debian and Arch-based distributions"
        exit 1
        ;;
esac

log ""
log_ok "CUDA installation complete!"
log "Please run 'source ~/.bashrc' or restart your terminal to update PATH"

# Verify installation
if check_command nvcc; then
    log ""
    log "CUDA version:"
    nvcc --version
else
    log ""
    log "WARNING: nvcc not found in PATH. You may need to restart your terminal."
fi
