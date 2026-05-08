#!/bin/bash

CONTAINER_NAME="nemotron-120b-a12b"
MODE_DIR_NAME="NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4"
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
            sglang serve \
            --model-path /root/.cache/modelscope/hub/models/${MODE_DIR_NAME} \
            --served-model-name nvidia/nemotron-3-super \
	    --host 0.0.0.0 \
            --port 30000 \
            --trust-remote-code \
            --tp 1 \
            --mem-fraction-static 0.85 \
            --sleep-on-idle \
            --disable-piecewise-cuda-graph \
            --quantization modelopt_fp4 \
   	    --attention-backend flashinfer \
            --fp4-gemm-backend flashinfer_cudnn \
	    --enable-flashinfer
            

fi
