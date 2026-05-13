#!/bin/bash
# Ollama：启动服务后 pull 指定 GGUF 模型并做一次 API 预热，模型才会进显存。
#
# 本机 ModelScope 目录里是 HuggingFace 分片 .safetensors，Ollama 不能当模型直接加载；
# 若要「用这份本地权重」跑推理，请用同目录下的 vllmstart_qwen3-6-35b-a3b-local.sh（OpenAI 兼容 :8000）。

set -euo pipefail

OLLAMA_MODEL="${OLLAMA_MODEL:-qwen3:30b-a3b}"

if docker ps -a --format '{{.Names}}' | grep -qx ollama; then
  docker start ollama
else
  docker run -d \
    --gpus all \
    --cpus 8 \
    --name ollama \
    -e OLLAMA_HOST=0.0.0.0:11434 \
    -p 11434:11434 \
    -v ollama:/root/.ollama \
    -v "${HOME}/.cache/modelscope/hub/models/qwen/Qwen3-6-35B-A3B:/mnt/model:ro" \
    ollama/ollama
fi

echo "等待 Ollama API..."
for _ in $(seq 1 60); do
  if curl -sf http://127.0.0.1:11434/api/tags >/dev/null; then
    break
  fi
  sleep 1
done

echo "拉取模型（已缓存则很快）: ${OLLAMA_MODEL}"
docker exec ollama ollama pull "${OLLAMA_MODEL}"

echo "预热加载到显存..."
curl -sfS http://127.0.0.1:11434/api/generate \
  -H 'Content-Type: application/json' \
  -d "$(printf '%s' '{"model":"'"${OLLAMA_MODEL}"'","prompt":"ping","stream":false}')" \
  >/dev/null

echo "Ollama API: http://127.0.0.1:11434"
echo "对话: docker exec -it ollama ollama run ${OLLAMA_MODEL}"
echo "其它模型: OLLAMA_MODEL=qwen3:32b $0"
