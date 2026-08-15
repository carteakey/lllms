#!/usr/bin/env bash
# Generate a MiniMax Music track through the official hosted mmx CLI.

set -euo pipefail

usage() {
    cat <<'EOF'
Usage:
  l3ms --media minimax-music --extra '--prompt "dreamy ambient electronica" --instrumental'

Options:
  --prompt TEXT             Music style/scene prompt (required)
  --prompt-file PATH        Read the prompt from a UTF-8 text file
  --lyrics TEXT             Lyrics to arrange
  --lyrics-file PATH        Read lyrics from a local UTF-8 file
  --instrumental            Generate without vocals
  --lyrics-optimizer        Ask MiniMax to optimize supplied lyrics
  --output PATH             MP3 output path (default: L3MS_MEDIA_OUTPUT_DIR)
  --help                    Show this help

Authentication:
  Install mmx-cli and run `mmx auth login --api-key <key>` once. The key is
  kept in the mmx config; this wrapper never prints it.
EOF
}

prompt=""
prompt_file=""
lyrics=""
lyrics_file=""
instrumental=false
lyrics_optimizer=false
output_dir="${L3MS_MEDIA_OUTPUT_DIR:-${HOME}/media-output}"
output=""

if [[ -s "${NVM_DIR:-${HOME}/.nvm}/nvm.sh" ]]; then
    # SSH/systemd shells often do not source .zshrc, but the user-level Node
    # install is still a supported path for the official mmx CLI.
    # shellcheck disable=SC1090
    . "${NVM_DIR:-${HOME}/.nvm}/nvm.sh"
fi

while (($#)); do
    case "$1" in
        --prompt)
            (($# >= 2)) || { echo "Missing value for --prompt" >&2; exit 2; }
            prompt="$2"
            shift 2
            ;;
        --prompt-file)
            (($# >= 2)) || { echo "Missing value for --prompt-file" >&2; exit 2; }
            prompt_file="$2"
            shift 2
            ;;
        --lyrics)
            (($# >= 2)) || { echo "Missing value for --lyrics" >&2; exit 2; }
            lyrics="$2"
            shift 2
            ;;
        --lyrics-file)
            (($# >= 2)) || { echo "Missing value for --lyrics-file" >&2; exit 2; }
            lyrics_file="$2"
            shift 2
            ;;
        --instrumental)
            instrumental=true
            shift
            ;;
        --lyrics-optimizer)
            lyrics_optimizer=true
            shift
            ;;
        --output|--out)
            (($# >= 2)) || { echo "Missing value for $1" >&2; exit 2; }
            output="$2"
            shift 2
            ;;
        --out-dir)
            (($# >= 2)) || { echo "Missing value for --out-dir" >&2; exit 2; }
            output_dir="$2"
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
            echo "Unknown MiniMax Music option: $1" >&2
            usage >&2
            exit 2
            ;;
    esac
done

if [[ -n "$prompt_file" ]]; then
    [[ -f "$prompt_file" ]] || { echo "Prompt file not found: $prompt_file" >&2; exit 1; }
    [[ -z "$prompt" ]] || { echo "--prompt and --prompt-file are mutually exclusive" >&2; exit 2; }
    prompt="$(<"$prompt_file")"
fi
if [[ -n "$lyrics_file" ]]; then
    [[ -f "$lyrics_file" ]] || { echo "Lyrics file not found: $lyrics_file" >&2; exit 1; }
    lyrics="$(<"$lyrics_file")"
fi
if [[ -z "$prompt" ]]; then
    echo "MiniMax Music requires --prompt TEXT" >&2
    exit 2
fi
if [[ "$instrumental" == true && -n "$lyrics" ]]; then
    echo "--instrumental and --lyrics are mutually exclusive" >&2
    exit 2
fi
if [[ -z "$lyrics" && "$instrumental" == false && "$lyrics_optimizer" == false ]]; then
    # mmx requires one of these modes. Prompt-only calls get the useful,
    # deterministic default of having MiniMax write the lyrics.
    lyrics_optimizer=true
fi
command -v mmx >/dev/null 2>&1 || {
    echo "mmx CLI not found; run maintenance/setup-media-runtimes.sh install-music-cli" >&2
    exit 1
}

mkdir -p "$output_dir"
if [[ -z "$output" ]]; then
    output="$output_dir/minimax-music-$(date +%Y%m%d-%H%M%S).mp3"
fi

args=(music generate --prompt "$prompt" --out "$output")
if [[ -n "$lyrics" ]]; then
    args+=(--lyrics "$lyrics")
fi
if [[ "$instrumental" == true ]]; then
    args+=(--instrumental)
fi
if [[ "$lyrics_optimizer" == true ]]; then
    args+=(--lyrics-optimizer)
fi

echo "MiniMax Music API -> $output"
exec mmx "${args[@]}"
