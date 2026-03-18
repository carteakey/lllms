#!/bin/bash
#
# build-sarvam-llama-cpp.sh
# -------------------------
# Thin wrapper around llama-test-pr.sh for Sarvam validation on upstream PR.
#
# Default PR:
#   20275
#
# Environment:
#   SARVAM_PR_NUMBER   (default: 20275)
#   Plus any env accepted by maintenance/llama-test-pr.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SARVAM_PR_NUMBER="${SARVAM_PR_NUMBER:-20275}"

if [ "$#" -gt 0 ]; then
    echo "build-sarvam-llama-cpp.sh does not accept positional args." >&2
    echo "Set SARVAM_PR_NUMBER (or use maintenance/llama-test-pr.sh directly) instead." >&2
    exit 1
fi

exec "${SCRIPT_DIR}/llama-test-pr.sh" "${SARVAM_PR_NUMBER}"
