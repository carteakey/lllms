#!/bin/bash
#
# build-gpt-oss-puzzle-llama-cpp.sh
# ---------------------------------
# Thin wrapper around llama-test-pr.sh for gpt-oss-puzzle validation on
# upstream llama.cpp PR flow.
#
# Default PR:
#   21032
#
# Environment:
#   GPT_OSS_PUZZLE_PR_NUMBER   (default: 21032)
#   Plus any env accepted by maintenance/llama-test-pr.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GPT_OSS_PUZZLE_PR_NUMBER="${GPT_OSS_PUZZLE_PR_NUMBER:-21032}"

if [ "$#" -gt 0 ]; then
    echo "build-gpt-oss-puzzle-llama-cpp.sh does not accept positional args." >&2
    echo "Set GPT_OSS_PUZZLE_PR_NUMBER (or use maintenance/llama-test-pr.sh directly) instead." >&2
    exit 1
fi

exec "${SCRIPT_DIR}/llama-test-pr.sh" "${GPT_OSS_PUZZLE_PR_NUMBER}"
