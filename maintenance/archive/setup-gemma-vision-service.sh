#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
UNIT_SRC="${REPO_DIR}/maintenance/systemd/gemma-vision.service"
UNIT_DST_DIR="${HOME}/.config/systemd/user"
UNIT_DST="${UNIT_DST_DIR}/gemma-vision.service"

usage() {
  cat <<'EOF'
Usage: maintenance/setup-gemma-vision-service.sh <command>

Commands:
  install   Install/update the user service, daemon-reload, enable, and start
  start     Start service
  stop      Stop service
  restart   Restart service
  enable    Enable on login/startup
  disable   Disable startup
  status    Show service status
  logs      Follow service logs (journalctl -f)
EOF
}

if [ ! -f "${UNIT_SRC}" ]; then
  echo "Service unit not found: ${UNIT_SRC}" >&2
  exit 1
fi

cmd="${1:-}"
if [ -z "${cmd}" ]; then
  usage
  exit 1
fi

mkdir -p "${UNIT_DST_DIR}"

case "${cmd}" in
  install)
    cp "${UNIT_SRC}" "${UNIT_DST}"
    systemctl --user daemon-reload
    systemctl --user enable --now gemma-vision.service
    systemctl --user status --no-pager gemma-vision.service || true
    ;;
  start)
    systemctl --user start gemma-vision.service
    ;;
  stop)
    systemctl --user stop gemma-vision.service
    ;;
  restart)
    systemctl --user restart gemma-vision.service
    ;;
  enable)
    systemctl --user enable gemma-vision.service
    ;;
  disable)
    systemctl --user disable gemma-vision.service
    ;;
  status)
    systemctl --user status --no-pager gemma-vision.service
    ;;
  logs)
    exec journalctl --user-unit gemma-vision.service -n 200 -f
    ;;
  *)
    usage
    exit 1
    ;;
esac
