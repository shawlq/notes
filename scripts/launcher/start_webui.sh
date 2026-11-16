#!/bin/bash
# Open WebUI → 本机 vLLM (:8000) + 远程 qwen-9B (:8001) + Cursor ACP API (:18791)

set -euo pipefail

CONTAINER_NAME="${CONTAINER_NAME:-open-webui}"
HOST_PORT="${HOST_PORT:-13000}"

# Cursor ACP OpenAI-compatible API token (from qiu-acp-bridge/.env); do not commit.
ROOT_ENV="${ROOT_ENV:-/home/atom/workdir/qiu-acp-bridge/.env}"
if [[ -z "${ACP_API_TOKEN:-}" && -f "${ROOT_ENV}" ]]; then
  ACP_API_TOKEN="$(grep -E '^API_TOKEN=' "${ROOT_ENV}" | head -1 | cut -d= -f2-)"
fi
ACP_API_TOKEN="${ACP_API_TOKEN:-sk-acp-missing}"

# 分号分隔多个 OpenAI 兼容端点
OPENAI_API_BASE_URLS="${OPENAI_API_BASE_URLS:-http://host.docker.internal:8000/v1;http://100.104.124.30:8001/v1;http://host.docker.internal:18791/v1}"
OPENAI_API_KEYS="${OPENAI_API_KEYS:-sk-anything;sk-anything;${ACP_API_TOKEN}}"

if docker ps -a --format '{{.Names}}' | grep -qx "${CONTAINER_NAME}"; then
  echo "移除旧容器 ${CONTAINER_NAME}（保留 volume open-webui）..."
  docker rm -f "${CONTAINER_NAME}" >/dev/null
fi

echo "启动 ${CONTAINER_NAME} → ${OPENAI_API_BASE_URLS}"
docker run -d \
  --name "${CONTAINER_NAME}" \
  --restart unless-stopped \
  -p "${HOST_PORT}:8080" \
  --add-host=host.docker.internal:host-gateway \
  -v open-webui:/app/backend/data \
  -e OPENAI_API_BASE_URLS="${OPENAI_API_BASE_URLS}" \
  -e OPENAI_API_KEYS="${OPENAI_API_KEYS}" \
  -e WEBUI_AUTH="${WEBUI_AUTH:-true}" \
  ghcr.io/open-webui/open-webui:main

echo "WebUI: http://127.0.0.1:${HOST_PORT}"
echo "模型侧: ${OPENAI_API_BASE_URLS}"
echo "提示: Open WebUI 连接列表缓存在 volume webui.db；若模型缺失，到 设置→连接→OpenAI 手动补 host.docker.internal:18791/v1（密钥=API_TOKEN），或见 qiu-acp-bridge/docs/cursor-acp-webui.md"
