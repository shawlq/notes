#!/bin/bash
# Open WebUI → 本机 vLLM (:8000) + 远程 qwen-9B (100.104.124.30:8001)

set -euo pipefail

CONTAINER_NAME="${CONTAINER_NAME:-open-webui}"
HOST_PORT="${HOST_PORT:-13000}"
# 分号分隔多个 OpenAI 兼容端点
OPENAI_API_BASE_URLS="${OPENAI_API_BASE_URLS:-http://host.docker.internal:8000/v1;http://100.104.124.30:8001/v1}"
OPENAI_API_KEYS="${OPENAI_API_KEYS:-sk-anything;sk-anything}"

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
