#!/usr/bin/env bash
# update-llama-stack.sh
# ---------------------
# Update the local llama-swap binary and rebuild the mainline llama.cpp tree.
#
# Defaults are conservative:
# - snapshot current binary + git/build metadata before writes
# - stop llama-swap.service while replacing binaries
# - restart llama-swap.service only if it was active before the update
#
# Env overrides:
#   UPDATE_LLAMA_SWAP=0       skip llama-swap binary update
#   UPDATE_LLAMA_CPP=0        skip llama.cpp pull/build
#   RESTART_LLAMA_SWAP=0      never restart llama-swap.service
#   STOP_LLAMA_SWAP=0         do not stop llama-swap.service before updating
#   DRY_RUN=1                 print actions without changing files
#   FORCE=1                   passed through to install-llama-swap.sh
#   LLAMA_SWAP_VERSION=vNNN   pin llama-swap release instead of latest
#   CUDA_ARCH=89              passed through to build-llama-cpp.sh

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SNAPSHOT_ROOT="${ROOT}/.toolkit/maintenance_versions"
STAMP="$(date +%Y%m%d-%H%M%S)"

UPDATE_LLAMA_SWAP="${UPDATE_LLAMA_SWAP:-1}"
UPDATE_LLAMA_CPP="${UPDATE_LLAMA_CPP:-1}"
RESTART_LLAMA_SWAP="${RESTART_LLAMA_SWAP:-1}"
STOP_LLAMA_SWAP="${STOP_LLAMA_SWAP:-1}"
DRY_RUN="${DRY_RUN:-0}"

log() {
  printf '%s\n' "-> $*"
}

warn() {
  printf 'WARN: %s\n' "$*" >&2
}

run() {
  if [ "${DRY_RUN}" = "1" ]; then
    printf '[dry-run] %q' "$1"
    shift
    for arg in "$@"; do
      printf ' %q' "$arg"
    done
    printf '\n'
    return 0
  fi
  "$@"
}

have_systemd_user() {
  systemctl --user is-active --quiet basic.target >/dev/null 2>&1
}

service_active() {
  systemctl --user is-active --quiet llama-swap.service >/dev/null 2>&1
}

snapshot_state() {
  local dir="${SNAPSHOT_ROOT}/${STAMP}"
  log "Writing snapshot: ${dir}"
  run mkdir -p "${dir}"

  if [ "${DRY_RUN}" = "1" ]; then
    return 0
  fi

  {
    printf 'timestamp=%s\n' "${STAMP}"
    printf 'root=%s\n' "${ROOT}"
    printf 'llama_swap_version='
    if [ -x "${HOME}/bin/llama-swap" ]; then
      "${HOME}/bin/llama-swap" -version 2>/dev/null || "${HOME}/bin/llama-swap" --version 2>/dev/null || true
    else
      printf 'missing\n'
    fi
    printf 'llama_cpp_head='
    git -C "${ROOT}/vendor/llama.cpp" rev-parse HEAD 2>/dev/null || true
    printf 'llama_cpp_branch='
    git -C "${ROOT}/vendor/llama.cpp" branch --show-current 2>/dev/null || true
    printf 'llama_cpp_status=\n'
    git -C "${ROOT}/vendor/llama.cpp" status --short 2>/dev/null || true
  } > "${dir}/state.txt"

  if [ -x "${HOME}/bin/llama-swap" ]; then
    cp -p "${HOME}/bin/llama-swap" "${dir}/llama-swap"
  fi

  if [ -d "${ROOT}/vendor/llama.cpp/build/bin" ]; then
    mkdir -p "${dir}/llama.cpp-build-bin"
    find "${ROOT}/vendor/llama.cpp/build/bin" -maxdepth 1 -type f -name 'llama-*' \
      -exec cp -p {} "${dir}/llama.cpp-build-bin/" \;
  fi
}

validate_llama_swap_config() {
  if [ ! -x "${HOME}/bin/llama-swap" ]; then
    warn "Skipping config validation: ${HOME}/bin/llama-swap is missing"
    return 0
  fi

  log "Validating llama-swap config"
  if [ "${DRY_RUN}" = "1" ]; then
    log "[dry-run] L3MS_ROOT=${ROOT} ${HOME}/bin/llama-swap -config ${ROOT}/llama-swap.yaml -watch-config"
    return 0
  fi

  local log_file
  log_file="$(mktemp)"
  set +e
  L3MS_ROOT="${ROOT}" timeout 5 "${HOME}/bin/llama-swap" \
    -config "${ROOT}/llama-swap.yaml" \
    -listen 127.0.0.1:0 \
    -watch-config >"${log_file}" 2>&1
  local status=$?
  set -e

  if [ "${status}" -ne 0 ] && [ "${status}" -ne 124 ]; then
      cat "${log_file}" >&2
      rm -f "${log_file}"
      return 1
  fi
  rm -f "${log_file}"
}

was_active=0
if have_systemd_user && service_active; then
  was_active=1
fi

snapshot_state

if [ "${STOP_LLAMA_SWAP}" = "1" ] && [ "${was_active}" = "1" ]; then
  log "Stopping llama-swap.service before binary updates"
  run systemctl --user stop llama-swap.service
fi

if [ "${UPDATE_LLAMA_SWAP}" = "1" ]; then
  log "Updating llama-swap"
  run env FORCE="${FORCE:-1}" LLAMA_SWAP_VERSION="${LLAMA_SWAP_VERSION:-latest}" \
    bash "${ROOT}/maintenance/install-llama-swap.sh"
else
  log "Skipping llama-swap update"
fi

if [ "${UPDATE_LLAMA_CPP}" = "1" ]; then
  log "Updating and rebuilding llama.cpp"
  run bash "${ROOT}/maintenance/build-llama-cpp.sh"
else
  log "Skipping llama.cpp update"
fi

validate_llama_swap_config

if [ "${RESTART_LLAMA_SWAP}" = "1" ] && have_systemd_user; then
  if [ "${was_active}" = "1" ]; then
    log "Restarting llama-swap.service"
    run systemctl --user restart llama-swap.service
  else
    log "llama-swap.service was not active before update; leaving it stopped"
  fi
elif [ "${RESTART_LLAMA_SWAP}" = "1" ]; then
  warn "systemd user bus unavailable; start/restart llama-swap.service manually if needed"
fi

log "Done"
