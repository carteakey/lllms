#!/bin/bash
#
# build-qwen-mtp-llama-cpp.sh
# ---------------------------------
# Thin wrapper around llama-test-pr.sh for Qwen MTP validation on
# upstream llama.cpp PR flow.
#
# Default PR:
#   22673
#
# Environment:
#   QWEN_MTP_PR_NUMBER   (default: 22673)
#   Plus any env accepted by maintenance/llama-test-pr.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
QWEN_MTP_PR_NUMBER="${QWEN_MTP_PR_NUMBER:-22673}"

if [ "$#" -gt 0 ]; then
    echo "build-qwen-mtp-llama-cpp.sh does not accept positional args." >&2
    echo "Set QWEN_MTP_PR_NUMBER (or use maintenance/llama-test-pr.sh directly) instead." >&2
    exit 1
fi

exec "${SCRIPT_DIR}/llama-test-pr.sh" "${QWEN_MTP_PR_NUMBER}"
