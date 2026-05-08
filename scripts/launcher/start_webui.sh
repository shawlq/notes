#!/bin/bash

CONTAINER_NAME="open-webui"

# 检查容器是否存在
if docker ps -a --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}$"; then
    echo "容器 ${CONTAINER_NAME} 已存在，正在启动..."
    docker start ${CONTAINER_NAME}
else
    echo "容器 ${CONTAINER_NAME} 不存在，正在创建并启动..."
    docker run -d \
      -p 13000:8080 \
      -v open-webui:/app/backend/data \
      -e OPENAI_API_BASE_URL=http://host.docker.internal:30000/v1 \
      -e OPENAI_API_KEY=sk-anything \
      --name ${CONTAINER_NAME} \
      ghcr.io/open-webui/open-webui:main
fi
