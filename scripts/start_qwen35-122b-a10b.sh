#!/bin/bash

CONTAINER_NAME="Qwen35-122B-A10B"

# 检查容器是否存在
if docker ps -a --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}$"; then
    echo "容器 ${CONTAINER_NAME} 已存在，正在启动..."
    docker start ${CONTAINER_NAME}
else
    echo "容器 ${CONTAINER_NAME} 不存在，正在创建并启动..."

    docker run -d \
        --name ${CONTAINER_NAME} \
        --gpus all \
        --network host \
        --ipc=host -v /data/models/modelscope:/root/.cache/modelscope/hub/models -e SGLANG_USE_MODELSCOPE=true \
        lmsysorg/sglang:nightly-dev-cu13-20260416-a4cf2ea1 \
            python3 -m sglang.launch_server \
            --model-path /root/.cache/modelscope/hub/models/Qwen3.5-122B-A10B-NVFP4 \
            --host 0.0.0.0 \
            --port 30000 \
            --trust-remote-code \
            --tp 1 \
            --mem-fraction-static 0.85 \
            --sleep-on-idle

fi
