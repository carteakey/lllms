#!/usr/bin/env bash
# install-llama-swap.sh
# ----------------------
# Fetch the llama-swap release binary into ~/bin/ (or LLAMA_SWAP_BIN_DIR).
#
# Env overrides:
#   LLAMA_SWAP_VERSION   pin a release tag (e.g. v143); default: latest
#   LLAMA_SWAP_BIN_DIR   install target dir;            default: $HOME/bin
#   LLAMA_SWAP_OS        override detected os  (linux|darwin)
#   LLAMA_SWAP_ARCH      override detected arch (amd64|arm64)
#   FORCE=1              overwrite an existing binary without prompting

set -euo pipefail

BIN_DIR="${LLAMA_SWAP_BIN_DIR:-${HOME}/bin}"
VERSION="${LLAMA_SWAP_VERSION:-latest}"
REPO="mostlygeek/llama-swap"

uname_os() {
  case "$(uname -s)" in
    Linux)  echo "linux" ;;
    Darwin) echo "darwin" ;;
    *)      echo "unsupported: $(uname -s)" >&2; exit 2 ;;
  esac
}

uname_arch() {
  case "$(uname -m)" in
    x86_64|amd64)  echo "amd64" ;;
    aarch64|arm64) echo "arm64" ;;
    *)             echo "unsupported: $(uname -m)" >&2; exit 2 ;;
  esac
}

OS="${LLAMA_SWAP_OS:-$(uname_os)}"
ARCH="${LLAMA_SWAP_ARCH:-$(uname_arch)}"

if [ "${VERSION}" = "latest" ]; then
  URL="https://github.com/${REPO}/releases/latest/download/llama-swap_${OS}_${ARCH}.tar.gz"
else
  URL="https://github.com/${REPO}/releases/download/${VERSION}/llama-swap_${OS}_${ARCH}.tar.gz"
fi

TARGET="${BIN_DIR}/llama-swap"

if [ -x "${TARGET}" ] && [ "${FORCE:-0}" != "1" ]; then
  current="$("${TARGET}" --version 2>/dev/null || echo 'unknown')"
  printf "%s already exists (%s). Re-run with FORCE=1 to overwrite.\n" "${TARGET}" "${current}"
  exit 0
fi

mkdir -p "${BIN_DIR}"
tmp="$(mktemp -d)"
trap 'rm -rf "${tmp}"' EXIT

echo "Downloading ${URL}"
if command -v curl >/dev/null 2>&1; then
  curl -fL --retry 3 -o "${tmp}/llama-swap.tar.gz" "${URL}"
elif command -v wget >/dev/null 2>&1; then
  wget -O "${tmp}/llama-swap.tar.gz" "${URL}"
else
  echo "Need curl or wget" >&2
  exit 2
fi

tar -xzf "${tmp}/llama-swap.tar.gz" -C "${tmp}"

# The tarball may contain just 'llama-swap' or a nested dir — find it.
binary="$(find "${tmp}" -name llama-swap -type f -perm -u+x | head -n 1 || true)"
if [ -z "${binary}" ]; then
  binary="$(find "${tmp}" -name llama-swap -type f | head -n 1 || true)"
fi
if [ -z "${binary}" ] || [ ! -f "${binary}" ]; then
  echo "llama-swap binary not found inside tarball" >&2
  exit 1
fi

install -m 0755 "${binary}" "${TARGET}"
echo "Installed: ${TARGET}"
"${TARGET}" --version || echo "(note: --version not supported by this build)"

case ":${PATH}:" in
  *":${BIN_DIR}:"*) ;;
  *) echo "Reminder: ${BIN_DIR} is not in \$PATH. Add it to your shell rc." ;;
esac
