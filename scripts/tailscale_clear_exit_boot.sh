#!/usr/bin/env bash
# 开机或服务启动时清空 Tailscale exit node（仅占位这两条，勿依赖仓库路径）。
# 建议：sudo install -Dm755 本脚本 /usr/local/sbin/tailscale_clear_exit_boot.sh
set -euo pipefail

sudo tailscale up --reset
sudo tailscale set --exit-node=
