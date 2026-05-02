#!/usr/bin/env bash
set -euo pipefail

# 与 vpn_start.md 一致的 exit node 地址
EXIT_NODE_IP="100.98.176.27"

usage() {
  echo "用法: $(basename "$0") {start|stop}" >&2
}

status() {
  tailscale status
  echo
  echo "exit ip: `curl -s ifconfig.me`"
}

case "${1:-}" in
  start)
    sudo tailscale up --exit-node="$EXIT_NODE_IP" --exit-node-allow-lan-access
    ;;
  stop)
    sudo tailscale up --reset
    sudo tailscale set --exit-node=
    ;;
  *)
    usage
    ;;
esac

echo 
status
echo
