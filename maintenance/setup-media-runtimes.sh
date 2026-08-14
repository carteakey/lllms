#!/usr/bin/env bash
# Install and inspect the media runtimes declared by media-runtimes.json.
#
# This script is intentionally separate from llama-swap: media generation is
# a CLI/API workload and should not compete with the text-model listener on
# port 8080. It never prints API tokens or Hugging Face credentials.

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd -- "${SCRIPT_DIR}/.." && pwd)"

MEDIA_ROOT="${L3MS_MEDIA_ROOT:-${HOME}/models/media}"
AUDIO_CPP_ROOT="${L3MS_AUDIO_CPP_ROOT:-${HOME}/repos/audio.cpp}"
LTX_ROOT="${L3MS_LTX_ROOT:-${HOME}/repos/LTX-2}"
AUDIO_CPP_REF="${L3MS_AUDIO_CPP_REF:-release-0.6}"
LTX_REF="${L3MS_LTX_REF:-main}"
H3_PACKAGE_ID="${L3MS_H3_PACKAGE_ID:-minimax_h3_q4_k}"

log() { printf '%s\n' "-> $*"; }
ok() { printf '%s\n' "[OK] $*"; }
warn() { printf '%s\n' "[WARN] $*" >&2; }
die() { printf '%s\n' "[ERROR] $*" >&2; exit 1; }

usage() {
    cat <<'EOF'
Usage: maintenance/setup-media-runtimes.sh <command>

Commands:
  check              Report runtime, model, and authentication readiness
  install-audio-cpp Clone/build audio.cpp and install MiniMax-H3 Q4_K GGUF
  install-ltx        Clone LTX-2 and create its uv environment
  install-music-cli  Install the official mmx CLI with npm
  install            Run all three installers (LTX weights remain gated)

Environment:
  L3MS_MEDIA_ROOT       Shared media models/output root (default: ~/models/media)
  L3MS_AUDIO_CPP_ROOT   audio.cpp checkout (default: ~/repos/audio.cpp)
  L3MS_LTX_ROOT         LTX-2 checkout (default: ~/repos/LTX-2)
  LTX_DOWNLOAD=1        After install-ltx, download gated LTX-2.5 files
  HF_TOKEN               Optional token honored by the Hugging Face CLI
EOF
}

require_command() {
    command -v "$1" >/dev/null 2>&1 || die "required command not found: $1"
}

prepare_cuda_path() {
    # CachyOS keeps the toolkit under /opt/cuda; non-interactive SSH and
    # systemd shells do not necessarily source the user's interactive profile.
    if [[ -x /opt/cuda/bin/nvcc ]]; then
        PATH="/opt/cuda/bin:$PATH"
        export PATH
        LD_LIBRARY_PATH="/opt/cuda/lib64:${LD_LIBRARY_PATH:-}"
        export LD_LIBRARY_PATH
    fi
}

ensure_checkout() {
    local url="$1"
    local ref="$2"
    local destination="$3"

    if [[ -e "$destination" && ! -d "$destination/.git" ]]; then
        die "destination exists but is not a Git checkout: $destination"
    fi
    if [[ -d "$destination/.git" ]]; then
        log "Using existing checkout: $destination ($(git -C "$destination" rev-parse --short HEAD))"
        return
    fi
    mkdir -p "$(dirname "$destination")"
    log "Cloning $url at $ref into $destination"
    git clone --branch "$ref" --depth 1 "$url" "$destination"
}

audio_cli_path() {
    local candidate
    for candidate in \
        "$AUDIO_CPP_ROOT/build/linux-cuda-release/bin/audiocpp_cli" \
        "$AUDIO_CPP_ROOT/build/bin/audiocpp_cli" \
        "$AUDIO_CPP_ROOT/build/debug/bin/audiocpp_cli" \
        "$AUDIO_CPP_ROOT/build/release/bin/audiocpp_cli"; do
        if [[ -x "$candidate" ]]; then
            printf '%s\n' "$candidate"
            return 0
        fi
    done
    return 1
}

install_audio_cpp() {
    require_command git
    require_command cmake
    prepare_cuda_path
    ensure_checkout "https://github.com/0xShug0/audio.cpp.git" "$AUDIO_CPP_REF" "$AUDIO_CPP_ROOT"
    [[ -x "$AUDIO_CPP_ROOT/scripts/build_linux.sh" ]] || die "audio.cpp build helper is missing"

    if ! audio_cli_path >/dev/null; then
        log "Building audio.cpp CUDA CLI/server for minimax_h3"
        (
            cd "$AUDIO_CPP_ROOT"
            build_dir="${AUDIO_CPP_BUILD_DIR:-$AUDIO_CPP_ROOT/build/linux-cuda-release}"
            # audio.cpp release-0.6 needs CCCL 3.2 for its CUB top-k path.
            # CUDA 13.3 ships a newer system CCCL, so explicitly fetch the
            # compatible version instead of relying on the system headers.
            cmake -S . -B "$build_dir" \
                -DCMAKE_BUILD_TYPE=RelWithDebInfo \
                -DENGINE_ENABLE_CUDA=ON \
                -DENGINE_ENABLE_VULKAN=OFF \
                -DENGINE_ENABLE_HIP=OFF \
                -DENGINE_ENABLE_NATIVE_CPU=ON \
                -DENGINE_ENABLE_LLAMAFILE=ON \
                -DENGINE_BUILD_EXAMPLES=OFF \
                -DENGINE_BUILD_TESTS=OFF \
                -DENGINE_BUILD_WARMBENCH=OFF \
                -DAUDIOCPP_DEPLOYMENT_BUILD=ON \
                -DAUDIOCPP_MODEL_SET=custom \
                -DAUDIOCPP_MODELS=minimax_h3 \
                -DGGML_CUDA_CUB_3DOT2=ON \
                -DCMAKE_CUDA_ARCHITECTURES="${L3MS_CUDA_ARCHITECTURES:-89-real}"
            cmake --build "$build_dir" --parallel "$(nproc 2>/dev/null || echo 8)" \
                --target audiocpp_cli --target audiocpp_server
        )
    fi
    local cli
    cli="$(audio_cli_path)" || die "audio.cpp build completed without audiocpp_cli"
    ok "audio.cpp CLI: $cli"

    mkdir -p "$MEDIA_ROOT"
    local model_python="$AUDIO_CPP_ROOT/.venv/bin/python"
    if [[ ! -x "$model_python" ]]; then
        require_command python3
        log "Creating audio.cpp model-manager environment"
        python3 -m venv "$AUDIO_CPP_ROOT/.venv"
        "$model_python" -m pip install --quiet --upgrade pip huggingface_hub
    fi

    local h3_dir="$MEDIA_ROOT/MiniMax-H3-Q4-GGUF"
    if [[ ! -f "$h3_dir/dit.gguf" ]]; then
        log "Installing audio.cpp package $H3_PACKAGE_ID into $MEDIA_ROOT"
        (
            cd "$AUDIO_CPP_ROOT"
            "$model_python" tools/model_manager_v2.py install "$H3_PACKAGE_ID" --models-root "$MEDIA_ROOT"
        )
    fi
    [[ -f "$h3_dir/dit.gguf" ]] || die "MiniMax-H3 package is incomplete: $h3_dir/dit.gguf"
    [[ -f "$h3_dir/text_encoder_q4_k.gguf" ]] || die "MiniMax-H3 text encoder is missing"
    [[ -f "$h3_dir/audio_vae_folded_f16.gguf" ]] || die "MiniMax-H3 audio VAE is missing"
    [[ -f "$h3_dir/video_vae.gguf" ]] || die "MiniMax-H3 video VAE is missing"
    ok "MiniMax-H3 Q4_K package: $h3_dir"
}

download_ltx_models() {
    log "Downloading gated LTX-2.5 distilled BF16 components"
    (
        cd "$LTX_ROOT"
        # HF_TOKEN is consumed by huggingface_hub from the environment when set;
        # no token is placed in argv or output.
        uv run --with huggingface_hub hf download Lightricks/LTX-2.5 \
            diffusion_models/ltx-2.5-22b-distilled-transformer-bf16.safetensors \
            text_encoders/gemma4-12b-with-proj-ltx-2.5-bf16.safetensors \
            vae/ltx-2.5-video-vae-conv-bf16.safetensors \
            vae/ltx-2.5-audio-vae-bf16.safetensors \
            latent_upscale_models/ltx-2.5-latent-spatial-upscaler-x2-bf16-1.0.safetensors \
            --local-dir "$MEDIA_ROOT/ltx-2.5"
    )
}

install_ltx() {
    require_command git
    require_command uv
    ensure_checkout "https://github.com/Lightricks/LTX-2.git" "$LTX_REF" "$LTX_ROOT"
    log "Syncing LTX-2 Python environment"
    (cd "$LTX_ROOT" && uv sync --extra natten)
    ok "LTX-2 checkout: $LTX_ROOT"
    if [[ "${LTX_DOWNLOAD:-0}" == "1" ]]; then
        download_ltx_models
    else
        warn "LTX-2.5 weights are gated and were not downloaded; set LTX_DOWNLOAD=1 after accepting the model terms"
    fi
}

install_music_cli() {
    if [[ -s "${NVM_DIR:-${HOME}/.nvm}/nvm.sh" ]]; then
        # shellcheck disable=SC1090
        . "${NVM_DIR:-${HOME}/.nvm}/nvm.sh"
    fi
    require_command node
    require_command npm
    local node_major
    node_major="$(node -p 'process.versions.node.split(".")[0]')"
    ((node_major >= 18)) || die "mmx-cli requires Node.js 18+ (found $node_major)"
    log "Installing official MiniMax mmx CLI"
    npm install --global mmx-cli
    command -v mmx >/dev/null 2>&1 || die "npm install completed without mmx on PATH"
    ok "mmx CLI: $(command -v mmx)"
    warn "Authenticate once with: mmx auth login --api-key <key> (the key is never stored in L3MS files)"
}

check_file() {
    local label="$1"
    local path="$2"
    if [[ -f "$path" ]]; then
        ok "$label: $path"
    else
        warn "$label missing: $path"
        return 1
    fi
}

check_runtime() {
    local missing=0
    if [[ -s "${NVM_DIR:-${HOME}/.nvm}/nvm.sh" ]]; then
        # shellcheck disable=SC1090
        . "${NVM_DIR:-${HOME}/.nvm}/nvm.sh"
    fi
    if command -v nvidia-smi >/dev/null 2>&1; then
        nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader | sed 's/^/GPU: /'
    else
        warn "nvidia-smi is unavailable; CUDA runtimes cannot be verified"
        missing=1
    fi

    local cli
    if cli="$(audio_cli_path 2>/dev/null)"; then
        ok "audio.cpp CLI: $cli"
    else
        warn "audio.cpp CLI missing under $AUDIO_CPP_ROOT"
        missing=1
    fi
    local h3_dir="$MEDIA_ROOT/MiniMax-H3-Q4-GGUF"
    check_file "H3 DiT" "$h3_dir/dit.gguf" || missing=1
    check_file "H3 text encoder" "$h3_dir/text_encoder_q4_k.gguf" || missing=1
    check_file "H3 audio VAE" "$h3_dir/audio_vae_folded_f16.gguf" || missing=1
    check_file "H3 video VAE" "$h3_dir/video_vae.gguf" || missing=1

    if [[ -d "$LTX_ROOT" ]]; then
        ok "LTX-2 checkout: $LTX_ROOT"
        local ltx_dir="${L3MS_LTX_MODEL_DIR:-${MEDIA_ROOT}/ltx-2.5}"
        check_file "LTX transformer" "$ltx_dir/diffusion_models/ltx-2.5-22b-distilled-transformer-bf16.safetensors" || missing=1
        check_file "LTX text encoder" "$ltx_dir/text_encoders/gemma4-12b-with-proj-ltx-2.5-bf16.safetensors" || missing=1
        check_file "LTX video VAE" "$ltx_dir/vae/ltx-2.5-video-vae-conv-bf16.safetensors" || missing=1
        check_file "LTX audio VAE" "$ltx_dir/vae/ltx-2.5-audio-vae-bf16.safetensors" || missing=1
        check_file "LTX spatial upscaler" "$ltx_dir/latent_upscale_models/ltx-2.5-latent-spatial-upscaler-x2-bf16-1.0.safetensors" || missing=1
    else
        warn "LTX-2 checkout missing: $LTX_ROOT"
        missing=1
    fi

    if command -v mmx >/dev/null 2>&1; then
        ok "MiniMax mmx CLI: $(command -v mmx)"
    else
        warn "MiniMax mmx CLI missing"
        missing=1
    fi
    return "$missing"
}

command="${1:-}"
case "$command" in
    check)
        check_runtime
        ;;
    install-audio-cpp)
        install_audio_cpp
        ;;
    install-ltx)
        install_ltx
        ;;
    install-music-cli)
        install_music_cli
        ;;
    install)
        install_audio_cpp
        install_ltx
        install_music_cli
        ;;
    -h|--help|"")
        usage
        [[ -n "$command" ]] || exit 1
        ;;
    *)
        usage >&2
        exit 2
        ;;
esac
