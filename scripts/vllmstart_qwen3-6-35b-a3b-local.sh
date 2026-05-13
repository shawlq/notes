#!/bin/bash
# 使用本机 ModelScope 缓存目录启动 vLLM OpenAI API，模型：Qwen3-6-35B-A3B（MoE）
# 默认宿主机 hub 根目录与 Docker 内路径对齐，便于直接读本地权重。

set -euo pipefail

CONTAINER_NAME="${CONTAINER_NAME:-vllm-qwen3-6-35b-a3b-local}"
# 宿主机上「hub」目录：其下应有 models/qwen/Qwen3-6-35B-A3B
HOST_MODELSCOPE_HUB="${HOST_MODELSCOPE_HUB:-${HOME}/.cache/modelscope/hub}"
CONTAINER_MODEL_PATH="/root/.cache/modelscope/hub/models/qwen/Qwen3-6-35B-A3B"
HOST_PORT="${HOST_PORT:-8000}"

# ---------- Docker 可调参数（常用）----------
# --gpus all              使用全部 GPU
# --cpus N                容器可用 CPU「总 vCPU 预算」（此处为 8）
# --cpuset-cpus 0-7       若希望绑定具体物理核，可改用此参数代替 --cpus（二者按需二选一）
# --memory / --memory-swap  限制内存（示例：--memory 64g）
# --shm-size              共享内存，vLLM/NCCL 建议加大（示例：16g）
# --ulimit memlock=-1     部分环境利于大模型/多卡通信
# --name / -p             容器名与端口映射
# --restart unless-stopped  需要常驻可加
#
# ---------- vLLM 可调参数（常用，见下方 docker run 内注释）----------
# --tensor-parallel-size   多卡张数并行（单卡填 1）
# --pipeline-parallel-size / --data-parallel-size
# --dtype auto|bfloat16|float16
# --max-model-len         最大上下文，显存/内存不够时优先调小
# --gpu-memory-utilization  KV 等占 GPU 显存比例上限（OOM 可调低，如 0.7）
# --max-num-seqs            最大并发序列数
# --enable-chunked-prefill  长上下文 prefill 分块，利于显存
# --trust-remote-code       自定义模型架构时需要

if [[ ! -d "${HOST_MODELSCOPE_HUB}/models/qwen/Qwen3-6-35B-A3B" ]]; then
  echo "错误: 未找到本地模型目录：" >&2
  echo "  ${HOST_MODELSCOPE_HUB}/models/qwen/Qwen3-6-35B-A3B" >&2
  echo "可通过环境变量 HOST_MODELSCOPE_HUB 指定正确的 hub 路径。" >&2
  exit 1
fi

if docker ps -a --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}$"; then
  echo "容器 ${CONTAINER_NAME} 已存在，正在启动..."
  docker start "${CONTAINER_NAME}"
else
  echo "容器 ${CONTAINER_NAME} 不存在，正在创建并启动..."

  docker run -d \
    --gpus all \
    --cpus 8 \
    --shm-size 16g \
    --name "${CONTAINER_NAME}" \
    -p "${HOST_PORT}:8000" \
    -e VLLM_USE_MODELSCOPE=False \
    -e HF_HUB_OFFLINE=1 \
    -v "${HOST_MODELSCOPE_HUB}:/root/.cache/modelscope/hub:ro" \
    vllm/vllm-openai:v0.18.1-cu130 \
    --model "${CONTAINER_MODEL_PATH}" \
    --served-model-name "${SERVED_MODEL_NAME:-qwen3-6-35b-a3b}" \
    --host 0.0.0.0 \
    --port 8000 \
    --dtype auto \
    --trust-remote-code \
    --tensor-parallel-size "${TP:-1}" \
    --gpu-memory-utilization "${GPU_MEM_UTIL:-0.85}" \
    --max-model-len "${MAX_MODEL_LEN:-131072}" \
    --max-num-seqs "${MAX_NUM_SEQS:-8}" \
    --enable-chunked-prefill \
    --async-scheduling \
    --moe-backend triton \
    --mamba_ssm_cache_dtype float32 \
    --enable-auto-tool-choice \
    --tool-call-parser hermes
fi

echo "OpenAI 兼容接口: http://127.0.0.1:${HOST_PORT}/v1"
echo "查看日志: docker logs -f ${CONTAINER_NAME}"
