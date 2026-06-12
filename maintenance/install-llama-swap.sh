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

http_get() {
  local url="$1"
  if command -v curl >/dev/null 2>&1; then
    curl -fsSL --retry 3 "${url}"
  elif command -v wget >/dev/null 2>&1; then
    wget -qO- "${url}"
  else
    echo "Need curl or wget" >&2
    exit 2
  fi
}

download_to_file() {
  local url="$1"
  local out="$2"
  if command -v curl >/dev/null 2>&1; then
    curl -fL --retry 3 -o "${out}" "${url}"
  elif command -v wget >/dev/null 2>&1; then
    wget -O "${out}" "${url}"
  else
    echo "Need curl or wget" >&2
    exit 2
  fi
}

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
  release_api="https://api.github.com/repos/${REPO}/releases/latest"
else
  release_api="https://api.github.com/repos/${REPO}/releases/tags/${VERSION}"
fi

release_json="$(http_get "${release_api}")"
if command -v jq >/dev/null 2>&1; then
  URL="$(
    printf '%s' "${release_json}" | jq -r --arg os "${OS}" --arg arch "${ARCH}" '
      .assets[]?.browser_download_url
      | select(test("llama-swap(_[0-9]+)?_" + $os + "_" + $arch + "\\.tar\\.gz$"))
    ' | head -n 1
  )"
else
  URL="$(
    printf '%s' "${release_json}" \
      | tr ',' '\n' \
      | sed -n 's/.*"browser_download_url"[[:space:]]*:[[:space:]]*"\([^"]*\)".*/\1/p' \
      | grep -E "llama-swap(_[0-9]+)?_${OS}_${ARCH}\.tar\.gz$" \
      | head -n 1 || true
  )"
fi

if [ -z "${URL}" ]; then
  echo "Could not find a llama-swap release asset for ${OS}/${ARCH} (version=${VERSION})." >&2
  exit 1
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
download_to_file "${URL}" "${tmp}/llama-swap.tar.gz"

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
