#!/bin/bash

CONTAINER_NAME="vllm-qwen3.6-35b"
MODE_DIR_NAME="Qwen3-6-35B-A3"
# 检查容器是否存在
if docker ps -a --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}$"; then
    echo "容器 ${CONTAINER_NAME} 已存在，正在启动..."
    docker start ${CONTAINER_NAME}
else
    echo "容器 ${CONTAINER_NAME} 不存在，正在创建并启动..."



docker run -d --gpus all \
  --name ${CONTAINER_NAME} \
  -e VLLM_NVFP4_GEMM_BACKEND=marlin \
  -e VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 \
  -e VLLM_FLASHINFER_ALLREDUCE_BACKEND=trtllm \
  -e VLLM_USE_FLASHINFER_MOE_FP4=0 \
  -e HF_TOKEN=$HF_TOKEN \
  -e VLLM_USE_MODELSCOPE=True \
  -v /data/models/modelscope:/root/.cache/modelscope/hub/models \
  -p 8000:8000 \
  vllm/vllm-openai:v0.18.1-cu130 \
    --model /root/.cache/modelscope/hub/models/${MODE_DIR_NAME} \
    --served-model-name qwen36-35b-vllm \
    --host 0.0.0.0 \
    --port 8000 \
    --async-scheduling \
    --dtype auto \
    --kv-cache-dtype fp8 \
    --tensor-parallel-size 1 \
    --pipeline-parallel-size 1 \
    --data-parallel-size 1 \
    --trust-remote-code \
    --gpu-memory-utilization 0.60 \
    --enable-chunked-prefill \
    --max-num-seqs 8 \
    --max-model-len 1000000 \
    --moe-backend triton \
    --mamba_ssm_cache_dtype float32 \
    --enable-flashinfer \
    --enable-auto-tool-choice \
    --tool-call-parser hermes

fi
#echo "  curl -sS http://localhost:8000/v1/chat/completions \
#    -H "Content-Type: application/json" \
#    -d '{
#      "model": "qwen36-35b-vllm",
#      "messages": [{"role": "user", "content": "为什么还有thinking"}],
#      "max_tokens": 2046,
#      "temperature": 0.6,
#      "chat_template_kwargs": {"enable_thinking": false}
#    }'"

