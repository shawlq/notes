#!/bin/bash
# 启动 HauhauCS Aggressive（BF16 safetensors 重建版）到本地 vLLM :8000
# 说明：官方 HauhauCS/Qwen3.6-35B-A3B-Uncensored-HauhauCS-Aggressive 仅提供 GGUF，
#       vLLM 无法直接加载；此处使用 ericpandev 的 BF16 safetensors 重建仓。

set -euo pipefail

CONTAINER_NAME="${CONTAINER_NAME:-vllm-qwen36-35b-hauhau-uncensored}"
HOST_HF_CACHE="${HOST_HF_CACHE:-/home/atom/.cache/huggingface}"
MODEL_DIR="${MODEL_DIR:-${HOST_HF_CACHE}/hub/HauhauCS-Aggressive-bf16}"
CONTAINER_MODEL_PATH="/root/.cache/huggingface/hub/HauhauCS-Aggressive-bf16"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-HauhauCS/Qwen3.6-35B-A3B-Uncensored-HauhauCS-Aggressive}"
IMAGE="${IMAGE:-vllm/vllm-openai:v0.28.0}"

# BF16 ~70GB，Spark/GB10 统一内存需控制上下文与并发
MAX_MODEL_LEN="${MAX_MODEL_LEN:-65536}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.75}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-4}"

if [[ ! -f "${MODEL_DIR}/config.json" ]]; then
  echo "错误: 未找到模型目录 ${MODEL_DIR}" >&2
  exit 1
fi

# 至少应有 17 个 shard
shard_count=$(ls -1 "${MODEL_DIR}"/model-*-of-*.safetensors 2>/dev/null | wc -l)
if [[ "${shard_count}" -lt 17 ]]; then
  echo "错误: safetensors 分片不完整 (${shard_count}/17)，请等待下载完成。" >&2
  exit 1
fi

# 停掉同端口旧服务（若存在）
if docker ps -a --format '{{.Names}}' | grep -qx 'vllm-qwen36-35b-a3b-nvfp4'; then
  docker stop vllm-qwen36-35b-a3b-nvfp4 >/dev/null 2>&1 || true
fi

if docker ps -a --format '{{.Names}}' | grep -qx "${CONTAINER_NAME}"; then
  echo "移除旧容器 ${CONTAINER_NAME} ..."
  docker rm -f "${CONTAINER_NAME}" >/dev/null
fi

echo "启动 ${CONTAINER_NAME} ..."
docker run -d \
  --name "${CONTAINER_NAME}" \
  --gpus all \
  --network host \
  --ipc host \
  --shm-size 16g \
  --restart unless-stopped \
  --security-opt label=disable \
  -e HF_HOME=/root/.cache/huggingface \
  -e HUGGINGFACE_HUB_CACHE=/root/.cache/huggingface/hub \
  -e HF_HUB_OFFLINE=1 \
  -e TRANSFORMERS_OFFLINE=1 \
  -e HF_ENDPOINT=https://hf-mirror.com \
  -v "${HOST_HF_CACHE}:/root/.cache/huggingface" \
  "${IMAGE}" \
  "${CONTAINER_MODEL_PATH}" \
  --host 0.0.0.0 \
  --port 8000 \
  --served-model-name "${SERVED_MODEL_NAME}" \
  --trust-remote-code \
  --tensor-parallel-size 1 \
  --dtype bfloat16 \
  --kv-cache-dtype fp8 \
  --attention-backend flashinfer \
  --moe-backend triton \
  --gpu-memory-utilization "${GPU_MEM_UTIL}" \
  --max-model-len "${MAX_MODEL_LEN}" \
  --max-num-seqs "${MAX_NUM_SEQS}" \
  --max-num-batched-tokens 8192 \
  --enable-chunked-prefill \
  --async-scheduling \
  --enable-prefix-caching \
  --default-chat-template-kwargs '{"enable_thinking": false}' \
  --reasoning-parser qwen3 \
  --enable-auto-tool-choice \
  --tool-call-parser qwen3_xml

echo "OpenAI 兼容接口: http://127.0.0.1:8000/v1"
echo "served-model-name: ${SERVED_MODEL_NAME}"
echo "日志: docker logs -f ${CONTAINER_NAME}"
