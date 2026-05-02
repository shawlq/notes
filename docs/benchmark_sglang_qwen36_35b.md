# benchmark测试

## 测试一

启动后端容器
```
docker run  -d \
  --gpus all \
  --rm \
  --network host \
  --ipc=host -v /data/models/modelscope:/root/.cache/modelscope/hub/models -e SGLANG_USE_MODELSCOPE=true  \
  --name ${CONTAINER_NAME} lmsysorg/sglang:nightly-dev-cu13-20260416-a4cf2ea1 \
    python3 -m sglang.launch_server  \
    --model-path /root/.cache/modelscope/hub/models/Qwen3-6-35B-A3 \
    --host 0.0.0.0 \
    --port 30000 \
    --trust-remote-code \
    --tp 1 \
    --mem-fraction-static 0.85 \
    --context-length 131072 \
    --sleep-on-idle
```

在容器内，执行
```
python3 -m sglang.bench_serving \
  --model /root/.cache/modelscope/hub/models/Qwen3-6-35B-A3 \
  --backend sglang \
  --num-prompts 100 \
  --dataset-name random \
  --dataset-path /root/.cache/modelscope/hub/models/ShareGPT_V3_unfiltered_cleaned_split.json \
  --request-rate inf \
  --host 127.0.0.1 \
  --port 30000
```

**注意**：ShareGPT_V3_unfiltered_cleaned_split.json在spark的workdir上，需要自己去外网下载，否则容器内很难下载，烦死了。


结果：

```
============ Serving Benchmark Result ============
Backend:                                 sglang
Traffic request rate:                    inf
Max request concurrency:                 not set
Successful requests:                     100
Benchmark duration (s):                  314.67
Total input tokens:                      50561
Total input text tokens:                 50561
Total generated tokens:                  52444
Total generated tokens (retokenized):    52184
Request throughput (req/s):              0.32
Input token throughput (tok/s):          160.68
Output token throughput (tok/s):         166.66
Peak output token throughput (tok/s):    260.00
Peak concurrent requests:                100
Total token throughput (tok/s):          327.34
Concurrency:                             59.47
----------------End-to-End Latency----------------
Mean E2E Latency (ms):                   187147.93
Median E2E Latency (ms):                 203659.78
P90 E2E Latency (ms):                    291033.19
P99 E2E Latency (ms):                    313782.80
---------------Time to First Token----------------
Mean TTFT (ms):                          57118.81
Median TTFT (ms):                        5916.51
P99 TTFT (ms):                           194098.96
-----Time per Output Token (excl. 1st token)------
Mean TPOT (ms):                          254.32
Median TPOT (ms):                        267.04
P99 TPOT (ms):                           282.13
---------------Inter-Token Latency----------------
Mean ITL (ms):                           248.41
Median ITL (ms):                         245.76
P95 ITL (ms):                            261.06
P99 ITL (ms):                            668.85
Max ITL (ms):                            6027.52
==================================================
```



## 测试二
启动参数新增 `--disable-piecewise-cuda-graph --quantization modelopt_fp4 --attention-backend flashinfer --fp4-gemm-backend flashinfer_cudnn --enable-flashinfer`

```
docker run  -d \
  --gpus all \
  --rm \
  --network host \
  --ipc=host -v /data/models/modelscope:/root/.cache/modelscope/hub/models -e SGLANG_USE_MODELSCOPE=true  \
  --name ${CONTAINER_NAME} lmsysorg/sglang:nightly-dev-cu13-20260416-a4cf2ea1 \
    python3 -m sglang.launch_server  \
    --model-path Qwen3-6-35B-A3 \
    --host 0.0.0.0 \
    --port 30000 \
    --trust-remote-code \
    --tp 1 \
    --mem-fraction-static 0.85 \
    --context-length 131072 \
    --sleep-on-idle \
    --disable-piecewise-cuda-graph \
    --quantization modelopt_fp4 \
    --attention-backend flashinfer \
    --fp4-gemm-backend flashinfer_cudnn \
    --enable-flashinfer  
```

结果： 加了新参数模型加载，失败，sglang太难用了，不测了。


