#!/usr/bin/env bash
# Generate a short LTX-2.5 synchronized audio/video clip.
#
# The profile uses the official distilled BF16 checkpoint with fp8-cast and
# CPU offload, which is the practical starting point for Yeti's RTX 4070 12 GB.

set -euo pipefail

usage() {
    cat <<'EOF'
Usage:
  l3ms --media ltx-2.5 --extra '--prompt "a paper boat crossing a puddle"'

Options:
  --prompt TEXT             Text prompt (required)
  --output PATH             MP4 output path (default: L3MS_MEDIA_OUTPUT_DIR)
  --frames N                Number of frames (default: 121)
  --seed N                  Deterministic seed (default: 42)
  --quantization NAME       fp8-cast (default)
  --offload TARGET          cpu or disk (default: cpu)
  --help                    Show this help

Environment:
  L3MS_LTX_ROOT             LTX-2 checkout (default: $HOME/repos/LTX-2)
  L3MS_LTX_MODEL_DIR        LTX-2.5 model directory (default: $HOME/models/media/ltx-2.5)
  L3MS_MEDIA_OUTPUT_DIR     Output directory (default: $HOME/media-output)
EOF
}

ltx_root="${L3MS_LTX_ROOT:-${HOME}/repos/LTX-2}"
model_dir="${L3MS_LTX_MODEL_DIR:-${L3MS_MEDIA_ROOT:-${HOME}/models/media}/ltx-2.5}"
output_dir="${L3MS_MEDIA_OUTPUT_DIR:-${HOME}/media-output}"
prompt=""
output=""
frames=121
seed=42
quantization="fp8-cast"
offload="cpu"

while (($#)); do
    case "$1" in
        --prompt)
            (($# >= 2)) || { echo "Missing value for --prompt" >&2; exit 2; }
            prompt="$2"
            shift 2
            ;;
        --output|--output-path)
            (($# >= 2)) || { echo "Missing value for $1" >&2; exit 2; }
            output="$2"
            shift 2
            ;;
        --frames|--num-frames|--seed|--quantization|--offload)
            (($# >= 2)) || { echo "Missing value for $1" >&2; exit 2; }
            case "$1" in
                --frames|--num-frames) frames="$2" ;;
                --seed) seed="$2" ;;
                --quantization) quantization="$2" ;;
                --offload) offload="$2" ;;
            esac
            shift 2
            ;;
        --help|-h)
            usage
            exit 0
            ;;
        --)
            shift
            ;;
        *)
            echo "Unknown LTX-2.5 option: $1" >&2
            usage >&2
            exit 2
            ;;
    esac
done

if [[ -z "$prompt" ]]; then
    echo "LTX-2.5 requires --prompt TEXT" >&2
    exit 2
fi
[[ -d "$ltx_root" ]] || {
    echo "LTX-2 checkout not found: $ltx_root" >&2
    echo "Run maintenance/setup-media-runtimes.sh install-ltx" >&2
    exit 1
}

transformer="$model_dir/diffusion_models/ltx-2.5-22b-distilled-transformer-bf16.safetensors"
text_encoder="$model_dir/text_encoders/gemma4-12b-with-proj-ltx-2.5-bf16.safetensors"
video_vae="$model_dir/vae/ltx-2.5-video-vae-conv-bf16.safetensors"
audio_vae="$model_dir/vae/ltx-2.5-audio-vae-bf16.safetensors"
spatial_upscaler="$model_dir/latent_upscale_models/ltx-2.5-latent-spatial-upscaler-x2-bf16-1.0.safetensors"
for required in "$transformer" "$text_encoder" "$video_vae" "$audio_vae" "$spatial_upscaler"; do
    [[ -f "$required" ]] || {
        echo "LTX-2.5 file not found: $required" >&2
        echo "Accept the gated model terms and run the download command in docs/media-generation-runbook.md" >&2
        exit 1
    }
done

command -v uv >/dev/null 2>&1 || {
    echo "uv is required to run LTX-2.5" >&2
    exit 1
}

mkdir -p "$output_dir"
if [[ -z "$output" ]]; then
    output="$output_dir/ltx-2.5-$(date +%Y%m%d-%H%M%S).mp4"
fi

echo "LTX-2.5 distilled: ${frames} frames / ${quantization} / ${offload} offload -> $output"
cd "$ltx_root"
exec uv run python -m ltx_pipelines.distilled \
    --transformer-path "$transformer" \
    --text-encoder-path "$text_encoder" \
    --video-vae-path "$video_vae" \
    --audio-vae-path "$audio_vae" \
    --spatial-upsampler-path "$spatial_upscaler" \
    --num-frames "$frames" \
    --seed "$seed" \
    --quantization "$quantization" \
    --offload "$offload" \
    --output-path "$output" \
    --prompt "$prompt"
