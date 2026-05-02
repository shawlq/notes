# 官方教程
参考 https://developer.nvidia.cn/build-spark/vllm#i7njvai  
容器更新为 vllm/vllm-openai:v0.18.1-cu130 

# 容器下载
```bash
docker pull vllm/vllm-openai:v0.18.1-cu130 
```

# 模型下载
```bash
sudo modelscope download --model nv-community/NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4 --local_dir /data/models/modelscope/NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4/
```

推理解析文件下载：
```
curl -O https://huggingface.co/nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4/raw/main/super_v3_reasoning_parser.py
```


# 启动命令

官方教程太简单，直接参考魔塔社区：https://www.modelscope.cn/models/nv-community/NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4#vllm-on-dgx-spark
去掉了reasoning parser 和 tool

```bash
#!/bin/bash

CONTAINER_NAME="vllm-nemotron-120b-a12b"
MODE_DIR_NAME="NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4"
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
    --served-model-name nemotron-3-super_vllm \
    --host 0.0.0.0 \
    --port 8000 \
    --async-scheduling \
    --dtype auto \
    --kv-cache-dtype fp8 \
    --tensor-parallel-size 1 \
    --pipeline-parallel-size 1 \
    --data-parallel-size 1 \
    --trust-remote-code \
    --gpu-memory-utilization 0.90 \
    --enable-chunked-prefill \
    --max-num-seqs 4 \
    --max-model-len 1000000 \
    --moe-backend marlin \
    --mamba_ssm_cache_dtype float32 \
    --quantization fp4 \
    --enable-flashinfer
fi
```

# 测试
## 本地测试
1. 查询模型
```
atom@spark-c3b6:~/workdir$ curl  http://localhost:8000/v1/models
{"object":"list","data":[{"id":"nemotron-3-super_vllm","object":"model","created":1776821398,"owned_by":"vllm","root":"/root/.cache/modelscope/hub/models/NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4","parent":null,"max_model_len":1000000,"permission":[{"id":"modelperm-b1f4e4bffa907890","object":"model_permission","created":1776821398,"allow_create_engine":false,"allow_sampling":true,"allow_logprobs":true,"allow_search_indices":false,"allow_view":true,"allow_fine_tuning":false,"organization":"*","group":null,"is_blocking":false}]}]}
```

2. 本地请求，关闭推理
```
atom@spark-c3b6:~/workdir$ curl http://localhost:8000/v1/chat/completions -H "Content-Type: application/json" -d '{
    "model": "nemotron-3-super_vllm",
    "messages": [{"role": "user", "content": "你好啊"}],
    "max_tokens": 500,
    "chat_template_kwargs": {"enable_thinking": false}
}'
{"id":"chatcmpl-9013b3edb886bd63","object":"chat.completion","created":1776821576,"model":"nemotron-3-super_vllm","choices":[{"index":0,"message":{"role":"assistant","content":"你好！😊 有什么我可以帮你的吗？无论是聊天、解答问题、写故事、写代码，还是只是需要一个倾听的朋友，我都在这里！ 🌟","refusal":null,"annotations":null,"audio":null,"function_call":null,"tool_calls":[],"reasoning":null},"logprobs":null,"finish_reason":"stop","stop_reason":null,"token_ids":null}],"service_tier":null,"system_fingerprint":null,"usage":{"prompt_tokens":18,"total_tokens":68,"completion_tokens":50,"prompt_tokens_details":null},"prompt_logprobs":null,"prompt_token_ids":null,"kv_transfer_params":null}
```

## open webui测试
1. 配置openai
- 设置-外部连接-新增，填写ttp://192.168.1.23:8000/v1，密钥为api-nothing（不会用到）

2. 设置thinking off
- 模型-高级参数，增加chat_template_kwargs，值为{"enable_thinking": false}

