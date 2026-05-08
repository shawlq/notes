# benchmark测试

## 测试一

启动后端容器
```
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
    --gpu-memory-utilization 0.90 \
    --enable-chunked-prefill \
    --max-num-seqs 8 \
    --max-model-len 1000000 \
    --moe-backend triton \
    --mamba_ssm_cache_dtype float32
```

在容器内，执行
```
vllm bench serve \
  --backend vllm \
  --num-prompts 100 \
  --dataset-name random \
  --request-rate inf \
  --base-url http://127.0.0.1:8000 \
  --endpoint /v1/completions \
  --temperature 0
```

结果：

```
============ Serving Benchmark Result ============
Successful requests:                     100
Failed requests:                         0
Benchmark duration (s):                  153.35
Total input tokens:                      102400
Total generated tokens:                  12800
Request throughput (req/s):              0.65
Output token throughput (tok/s):         83.47
Peak output token throughput (tok/s):    128.00
Peak concurrent requests:                100.00
Total token throughput (tok/s):          751.21
---------------Time to First Token----------------
Mean TTFT (ms):                          74889.94
Median TTFT (ms):                        78476.66
P99 TTFT (ms):                           146863.84
-----Time per Output Token (excl. 1st token)------
Mean TPOT (ms):                          85.89
Median TPOT (ms):                        84.89
P99 TPOT (ms):                           136.35
---------------Inter-token Latency----------------
Mean ITL (ms):                           85.89
Median ITL (ms):                         72.84
P99 ITL (ms):                            346.76
==================================================
```


-------------------------------


## 测试二

启动容器，增加参数 --enable-flashinfer，没有fp4不能用。
```
...
    --enable-flashinfer
```

结果如下：

```
============ Serving Benchmark Result ============
Successful requests:                     100
Failed requests:                         0
Benchmark duration (s):                  151.35
Total input tokens:                      102400
Total generated tokens:                  12800
Request throughput (req/s):              0.66
Output token throughput (tok/s):         84.57
Peak output token throughput (tok/s):    120.00
Peak concurrent requests:                100.00
Total token throughput (tok/s):          761.16
---------------Time to First Token----------------
Mean TTFT (ms):                          73163.41
Median TTFT (ms):                        76239.29
P99 TTFT (ms):                           144582.91
-----Time per Output Token (excl. 1st token)------
Mean TPOT (ms):                          84.95
Median TPOT (ms):                        84.00
P99 TPOT (ms):                           137.01
---------------Inter-token Latency----------------
Mean ITL (ms):                           84.95
Median ITL (ms):                         72.04
P99 ITL (ms):                            347.54
==================================================
```
