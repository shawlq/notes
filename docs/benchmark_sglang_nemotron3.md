# benchmark测试

## 测试一

启动后端容器
```
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
```

在容器内，执行
```
python3 -m sglang.bench_serving
  --model /root/.cache/modelscope/hub/models/NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4 \
  --backend sglang \
  --num-prompts 100 \
  --dataset-name random \
  --dataset-path /sgl-workspace/sglang/ShareGPT_V3_unfiltered_cleaned_split.json \
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
Benchmark duration (s):                  695.79
Total input tokens:                      50561
Total input text tokens:                 50561
Total generated tokens:                  52444
Total generated tokens (retokenized):    51734
Request throughput (req/s):              0.14
Input token throughput (tok/s):          72.67
Output token throughput (tok/s):         75.37
Peak output token throughput (tok/s):    105.00
Peak concurrent requests:                100
Total token throughput (tok/s):          148.04
Concurrency:                             53.93
----------------End-to-End Latency----------------
Mean E2E Latency (ms):                   375235.42
Median E2E Latency (ms):                 374078.00
P90 E2E Latency (ms):                    635501.97
P99 E2E Latency (ms):                    691654.68
---------------Time to First Token----------------
Mean TTFT (ms):                          244508.05
Median TTFT (ms):                        240657.66
P99 TTFT (ms):                           545801.63
-----Time per Output Token (excl. 1st token)------
Mean TPOT (ms):                          252.11
Median TPOT (ms):                        256.12
P99 TPOT (ms):                           271.37
---------------Inter-Token Latency----------------
Mean ITL (ms):                           249.75
Median ITL (ms):                         240.76
P95 ITL (ms):                            260.98
P99 ITL (ms):                            792.08
Max ITL (ms):                            6086.24
==================================================
```

分析：

| 指标                 | sglang     | vLLM（第二次）  | 差距        |
| ------------------ | ---------- | ---------- | --------- |
| Request throughput | 0.14 req/s | 0.47 req/s | ❌ 慢 3.3 倍 |
| Output tok/s       | 75.37      | 60.37      | ✅ 单点略高    |
| Total tok/s        | 148        | 543        | ❌ 总吞吐低很多  |
| Concurrency        | 53.9       | 100        | ❌ 并发没跑满   |

👉 关键点：

sglang 没有把 100 并发吃满（只有 ~54）
vLLM 是 fully packed（100）

📌 这说明：

sglang scheduler 没把请求有效 batch 起来

-----------

| 指标        | sglang    | vLLM  |
| --------- | --------- | ----- |
| Mean TTFT | **244 秒** | 98 秒  |
| P99 TTFT  | **545 秒** | 201 秒 |
👉 结论：

已经不是“慢”，是严重异常
4分钟才出第一个 token

📌 这通常只可能是：

🚨 三种情况之一：
prefill 被严重串行化（没有 batching）
KV cache / graph 没复用
scheduler 等待队列策略有问题（卡住）

-----------

| 指标        | sglang | vLLM   |
| --------- | ------ | ------ |
| Mean TPOT | 252 ms | 117 ms |

👉 慢了 2 倍以上

这就说明：

❌ 不只是 TTFT 问题，连 decode 都没优化好

-----------

| 指标       | 数值               |
| -------- | ---------------- |
| Mean ITL | 249 ms           |
| P99 ITL  | 792 ms           |
| Max ITL  | **6086 ms (!!)** |

👉 这个非常关键：

出现 6秒卡顿
说明：
GPU kernel 被打断
或 scheduler stall
或 batch rebuild 很慢

📌 这在健康系统里不应该出现

-----------

| 能力                  | vLLM   | sglang 当前   |
| ------------------- | ------ | ----------- |
| continuous batching | ✅ 强    | ⚠️ 可能没生效    |
| KV cache 管理         | ✅ 高效   | ⚠️ 可能重复构建   |
| 调度器成熟度              | ✅ 很高   | ❌ 当前配置明显有问题 |
| NVFP4 支持            | ⚠️ 有路径 | ❌ 基本没用上     |
