#!/usr/bin/env bash
# run-l3ms-kpc.sh
# ----------------
# Launch the Rust L3MS binary from the KPC Git checkout.
#
# The llama-swap URL and API key are derived from the user service when they
# are not already supplied. Extra arguments are passed to the L3MS binary.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BIN="${L3MS_BIN:-${ROOT}/target/release/l3ms}"
SERVICE="${L3MS_LLAMA_SWAP_SERVICE:-llama-swap.service}"
DEFAULT_PORT="${L3MS_LLAMA_SWAP_PORT:-8001}"

if [ -f "${HOME}/.cargo/env" ]; then
  # shellcheck source=/dev/null
  . "${HOME}/.cargo/env"
fi

service_environment=""
if command -v systemctl >/dev/null 2>&1; then
  service_environment="$(
    systemctl --user show "${SERVICE}" --property=Environment --value 2>/dev/null || true
  )"
fi

service_listen="$(
  printf '%s\n' "${service_environment}" \
    | tr ' ' '\n' \
    | sed -n 's/^LLAMA_SWAP_LISTEN=//p' \
    | sed -n '1p'
)"

if [ -z "${LLAMA_SWAP_URL:-}" ]; then
  port="${DEFAULT_PORT}"
  case "${service_listen}" in
    :*) port="${service_listen#:}" ;;
    *:*) port="${service_listen##*:}" ;;
  esac
  export LLAMA_SWAP_URL="http://127.0.0.1:${port}"
fi

if [ -z "${LLAMA_SWAP_API_KEY:-}" ]; then
  service_key="$(
    printf '%s\n' "${service_environment}" \
      | tr ' ' '\n' \
      | sed -n 's/^LLAMA_SWAP_API_KEY=//p' \
      | sed -n '1p'
  )"
  if [ -n "${service_key}" ]; then
    export LLAMA_SWAP_API_KEY="${service_key}"
  fi
fi

if [ -z "${L3MS_DOWNLOADER_PYTHON:-}" ]; then
  if [ -x "${ROOT}/.venv/bin/python3" ]; then
    export L3MS_DOWNLOADER_PYTHON="${ROOT}/.venv/bin/python3"
  elif command -v python3 >/dev/null 2>&1; then
    export L3MS_DOWNLOADER_PYTHON="$(command -v python3)"
  else
    printf '%s\n' "Python 3 is required for the downloader compatibility boundary." >&2
    exit 1
  fi
fi

if [ ! -x "${BIN}" ]; then
  printf 'Missing executable: %s\n' "${BIN}" >&2
  printf 'Build it with: cargo build --release --locked\n' >&2
  exit 1
fi

exec "${BIN}" "$@"
