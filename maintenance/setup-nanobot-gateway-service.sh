#!/usr/bin/env bash
# setup-nanobot-gateway-service.sh
# ----------------------
# Installer/controller for the nanobot-gateway user service.

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="${SCRIPT_DIR}" # Since this is in maintenance/, REPO_DIR should be the parent
if [[ "$(basename "${SCRIPT_DIR}")" == "maintenance" ]]; then
  REPO_DIR="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
fi

UNIT_SRC="${REPO_DIR}/maintenance/systemd/nanobot-gateway.service"
UNIT_DST_DIR="${HOME}/.config/systemd/user"
UNIT_DST="${UNIT_DST_DIR}/nanobot-gateway.service"

usage() {
  cat <<EOF
Usage: $(basename "$0") <command>

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
    systemctl --user enable --now nanobot-gateway.service
    systemctl --user status --no-pager nanobot-gateway.service || true
    ;;
  start)
    systemctl --user start nanobot-gateway.service
    ;;
  stop)
    systemctl --user stop nanobot-gateway.service
    ;;
  restart)
    systemctl --user restart nanobot-gateway.service
    ;;
  enable)
    systemctl --user enable nanobot-gateway.service
    ;;
  disable)
    systemctl --user disable nanobot-gateway.service
    ;;
  status)
    systemctl --user status --no-pager nanobot-gateway.service
    ;;
  logs)
    exec journalctl --user-unit nanobot-gateway.service -n 200 -f
    ;;
  *)
    usage
    exit 1
    ;;
esac
