# AI-Infra + vLLM 8 周学习计划 · 每日阅读资料

> 本文档整理了 56 天学习中每天需要阅读的论文、文档、源码和仓库链接。
> 每天早上 5:00 定时推送时会同步发送当日资料。

---

## Week 1：量化数学手感 + GPU 硬件底座

### Day 1：环境准备与 Infra 全景扫描

**仓库克隆：**
- vLLM：https://github.com/vllm-project/vllm
- AutoAWQ：https://github.com/casper-hansen/AutoAWQ
- AI-Infra-from-Zero-to-Hero：https://github.com/HuaizhengZhang/AI-Infra-from-Zero-to-Hero

**阅读：**
- AI-Infra 仓库 `README.md` — 重点看 "System for AI (Ordered by Category)" 部分
- 快速浏览目录：`inference.md`、`llm_serving.md` 的论文列表（只看标题）

---

### Day 2：量化数学与 Scale/Zero-point

**论文：**
- Jacob Devlin 等, *"Quantization and Training of Neural Networks for Efficient Integer-Arithmetic-Only Inference"* (2017)
  - PDF：https://arxiv.org/abs/1712.05877
  - 重点：前 5 页，理解 scale / zero-point 公式推导

**参考文档：**
- PyTorch 量化官方文档：https://pytorch.org/docs/stable/quantization.html
- 博客：*"A Survey of Quantization Methods for Efficient Neural Network Inference"* — 理解 PTQ vs QAT 的区别

---

### Day 3：手写 SymmetricQuantizer

**参考代码：**
- PyTorch 量化 API 源码参考：`torch.quantization` 模块
- 简单量化示例：https://pytorch.org/tutorials/advanced/static_quantization_tutorial.html

**思考参考：**
- 博客：*"Understanding Quantization: From Floating Point to Integer"* — 帮助理解 Outlier 对量化误差的影响

---

### Day 4：GPU 硬件感知（结合 C++ 背景）

**阅读：**
- AI-Infra 仓库中硬件架构相关内容（搜索 "GPU architecture" 或 "memory hierarchy"）
- NVIDIA Hopper 架构白皮书（4090 基于 Ada Lovelace，但 Hopper 白皮书的内存层次描述通用）：
  - https://resources.nvidia.com/en-us-tensor-core/nvidia-tensor-core-gpu-whitepaper
- 博客推荐：*"GPU Memory Hierarchy Explained"* — 理解 Register → Shared Memory (SRAM) → L2 → Global Memory (DRAM) 的带宽差异

**关键数据（4090 / Ada Lovelace）：**
| 层级 | 带宽 | 容量 |
|------|------|------|
| Register | ~20 TB/s | ~256 KB/SM |
| Shared Memory (SRAM) | ~19 TB/s | ~228 KB/SM |
| L2 Cache | ~5 TB/s | 72 MB |
| Global Memory (DRAM) | ~1 TB/s | 24 GB |

---

### Day 5：量化与硬件的化学反应

**阅读：**
- 论文：*"GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers"* — 先看 Introduction，理解量化加速的本质
  - https://arxiv.org/abs/2210.17323
- 博客：*"Why Quantization Works: A Memory Bandwidth Perspective"* — 理解 DRAM 带宽瓶颈

**思考参考：**
- INT8 矩阵乘法 vs FP32 的理论加速比 = min(算力加速比, 带宽加速比)
- 在 memory-bound 场景下，加速比 ≈ FP32 位数 / INT8 位数 = 4x

---

### Day 6：inference.md 精读

**阅读：**
- AI-Infra 仓库 `inference.md` — Inference System 小节精读
  - 重点系统：Clockwork、Orca、Cloudburst、Swayam
  - 每个系统只读 Abstract + Introduction

**论文推荐（选读 1-2 篇）：**
- *"Clockwork: A Timeout-Aware Resource Management System for ML Inference"* — https://arxiv.org/abs/2011.09011
- *"Orca: A Distributed Serving System for Transformer-Based Generative Models"* — https://arxiv.org/abs/2304.02122

---

### Day 7：Week 1 总结

**输出任务：**
- 撰写 Markdown 笔记：《INT8 对称量化公式推导与 GPU 带宽瓶颈》
- 整理本周所有代码和实验数据

**预习：**
- 提前浏览 AWQ 论文：https://arxiv.org/abs/2306.00978
- 提前下载 AutoAWQ / AutoGPTQ 仓库

---

## Week 2：LLM 量化实战 (AWQ/GPTQ) + Serving 系统

### Day 8：LLM 推理瓶颈初探

**阅读：**
- AI-Infra 仓库 `llm_serving.md` — 全文扫读
  - 重点标记：Sarathi-Serve、DistServe、Llumnix、InfiniGen
  - 每篇只读 Abstract + Introduction

**论文（快速浏览）：**
- *"Sarathi-Serve: Efficient LLM Serving with Lazy Orchestration"* — https://arxiv.org/abs/2403.02313
- *"DistServe: Disaggregating Prefill and Decoding for Goodput-optimized Large Language Model Serving"* — https://arxiv.org/abs/2401.09670
- *"Llumnix: Efficient and Elastic LLM Serving with Dynamic Batching and Streaming"* — https://arxiv.org/abs/2402.01880

**关键概念：**
- Prefill 阶段：compute-bound（算力瓶颈）
- Decode 阶段：memory-bound（显存带宽瓶颈）
- KV Cache 是 Decode 阶段的显存大户

---

### Day 9：AWQ 论文精读

**论文：**
- *"AWQ: Activation-aware Weight Quantization for LLM Compression and Acceleration"* (2023)
  - https://arxiv.org/abs/2306.00978
  - **精读全文**，重点：
    - Section 3: 为什么激活值很重要
    - Section 4: 保护显著权重通道（salient weight channels）的算法
    - Figure 2: AWQ vs 均匀量化的精度对比

**核心问题：**
- AWQ 说"保护 1% 的显著权重通道"就能大幅降低精度损失，为什么？
- AWQ 的搜索算法如何找到最优缩放因子 s？
- AWQ vs GPTQ 的本质区别：AWQ 看激活，GPTQ 看 Hessian

---

### Day 10：GPTQ 论文精读

**论文：**
- *"GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers"* (2022)
  - https://arxiv.org/abs/2210.17323
  - **精读全文**，重点：
    - Section 2: 基于 OBS（Optimal Brain Surgeon）的近似
    - Section 3: 逐层量化算法（Algorithm 1）
    - Appendix: 为什么可以一次性量化整个模型

**对比参考：**
- AWQ vs GPTQ 对比表：
  | 维度 | AWQ | GPTQ |
  |------|-----|------|
  | 核心思想 | 保护显著权重通道 | 最小化逐层量化误差 |
  | 依赖信息 | 激活值统计 | Hessian 矩阵（二阶信息） |
  | 量化速度 | 较快 | 较慢（需要逐列处理） |
  | 精度 | 略优 | 好 |
  | 推理加速 | 好（支持 group size） | 好 |

---

### Day 11：AutoAWQ 4090 实战

**仓库：**
- AutoAWQ：https://github.com/casper-hansen/AutoAWQ
- 安装：`pip install autoawq`

**文档：**
- AutoAWQ 官方 README 中的量化示例
- 模型选择：Meta-Llama-3-8B（https://huggingface.co/meta-llama/Meta-Llama-3-8B）

**代码参考：**
```python
from awq import AutoAWQForCausalLM
from transformers import AutoTokenizer

model_path = "meta-llama/Meta-Llama-3-8B"
quant_path = "Meta-Llama-3-8B-AWQ-4bit"

tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
model = AutoAWQForCausalLM.from_pretrained(model_path, trust_remote_code=True)

model.quantize(tokenizer, quant_config={
    "zero_point": True,
    "q_group_size": 128,
    "w_bit": 4,
    "version": "GEMM"
})

model.save_quantized(quant_path)
tokenizer.save_pretrained(quant_path)
```

---

### Day 12：vLLM 加载 AWQ 模型测试

**文档：**
- vLLM 官方量化支持文档：https://docs.vllm.ai/en/latest/models/supported_models.html
- vLLM AWQ 加载示例：https://docs.vllm.ai/en/latest/quantization/awq.html

**代码参考：**
```python
from vllm import LLM, SamplingParams

llm = LLM(model="Meta-Llama-3-8B-AWQ-4bit", quantization="awq")
params = SamplingParams(temperature=0.0, max_tokens=256)
outputs = llm.generate(["Hello, my name is"], params)
```

**监控命令：**
```bash
watch -n 0.5 nvidia-smi
```

---

### Day 13：GPTQ 实战对比

**仓库：**
- AutoGPTQ：https://github.com/AutoGPTQ/AutoGPTQ
- 安装：`pip install auto-gptq`

**文档：**
- AutoGPTQ 官方 README 中的量化示例
- vLLM GPTQ 加载文档：https://docs.vllm.ai/en/latest/quantization/gptq.html

**对比实验要点：**
- 使用相同的 prompt 和生成参数（temperature=0.0, top_p=1.0, max_tokens=256）
- 记录：tokens/s、显存峰值、模型文件大小
- （选做）Perplexity 评估：使用 `lm-eval-harness`

---

### Day 14：推理系统全局观 + Week 2 总结

**阅读：**
- AI-Infra 仓库 `inference.md` — Model Serving / Serverless / Autoscaling 部分
  - 重点系统：Clockwork、Orca、RedisAI、Cloudburst

**论文（选读）：**
- *"Clockwork: A Timeout-Aware Resource Management System"* — https://arxiv.org/abs/2011.09011
- *"Orca: A Distributed Serving System"* — https://arxiv.org/abs/2304.02122

**输出任务：**
- 撰写对比报告：《4090 显卡下 FP16 vs AWQ-4bit vs GPTQ-4bit 性能与精度分析》

**预习：**
- Python asyncio 官方文档：https://docs.python.org/3/library/asyncio.html

---

## Week 3：异步编程 + vLLM API Server / LLM Engine

### Day 15：Python asyncio 基础复习

**文档：**
- Python asyncio 官方文档：https://docs.python.org/3/library/asyncio.html
- 重点章节：
  - Coroutines and Tasks
  - asyncio.Queue
  - asyncio.gather / asyncio.create_task

**教程推荐：**
- Real Python: *"Async IO in Python: A Complete Walkthrough"* — https://realpython.com/async-io-python/

---

### Day 16：vLLM 架构概览

**文档：**
- vLLM Architecture Overview：https://docs.vllm.ai/en/latest/getting_started/architecture.html
- 重点组件：AsyncLLMEngine、LLMEngine、Worker、ModelRunner、Scheduler

**源码入口：**
- `vllm/engine/async_llm_engine.py`
- `vllm/engine/llm_engine.py`
- `vllm/worker/worker.py`

---

### Day 17：vllm/entrypoints/api_server.py 源码阅读

**源码：**
- `vllm/entrypoints/openai/api_server.py`（vLLM 使用 OpenAI 兼容 API）
- 重点看路由定义和请求处理流程

**参考：**
- FastAPI 文档：https://fastapi.tiangolo.com/
- 理解 FastAPI 的依赖注入和异步路由

---

### Day 18：AsyncLLMEngine 源码阅读

**源码：**
- `vllm/engine/async_llm_engine.py`
- 重点函数：
  - `add_request()` — 添加推理请求
  - `step()` — 执行一步推理
  - `generate()` — 生成完整响应
  - `_process_model_requests()` — 请求调度循环

---

### Day 19：异步生产者-消费者实战

**参考代码：**
- Python asyncio.Queue 官方示例
- vLLM 中 AsyncLLMEngine 的请求队列实现

**练习目标：**
- 生产者：异步从文件读请求 → asyncio.Queue
- 消费者：模拟 GPU → await asyncio.sleep → 打印 batch
- 支持多消费者并发

---

### Day 20：DistServe / Llumnix 论文精读

**论文（选一篇精读）：**
- *"DistServe: Disaggregating Prefill and Decoding for Goodput-optimized LLM Serving"*
  - https://arxiv.org/abs/2401.09670
  - 重点：Prefill/Decode 分离架构、调度策略
- *"Llumnix: Efficient and Elastic LLM Serving with Dynamic Batching and Streaming"*
  - https://arxiv.org/abs/2402.01880
  - 重点：动态批处理、实例迁移

**对比 vLLM：**
- vLLM 的 continuous batching vs DistServe 的 prefill-decode 分离
- 各自的优缺点和适用场景

---

### Day 21：Week 3 总结

**输出任务：**
- HTTP 请求 → AsyncLLMEngine → Worker → ModelRunner 流程图
- 异步生产者-消费者脚本
- DistServe/Llumnix 论文笔记

**预习：**
- PagedAttention 论文：https://arxiv.org/abs/2309.06180

---

## Week 4：PagedAttention + Inference System 理论

### Day 22：PagedAttention 论文精读（上）

**论文：**
- *"Efficient Memory Management for Large Language Model Serving with PagedAttention"* (vLLM 团队, 2023)
  - https://arxiv.org/abs/2309.06180
  - **精读 Section 1-3**：
    - Section 1: 问题——传统 KV cache 的内存浪费
    - Section 2: 方案——虚拟内存分页思想
    - Section 3: PagedAttention 算法

---

### Day 23：PagedAttention 论文精读（下）

**论文（续）：**
- PagedAttention 论文 **Section 4-6**：
  - Section 4: 系统实现（vLLM）
  - Section 5: 实验结果
  - Section 6: 结论

**重点理解：**
- Block table 数据结构
- Copy-on-Write 机制（beam search / parallel sampling）
- 内存分配/回收策略

---

### Day 24：vLLM PagedAttention 源码 — 内存管理

**源码：**
- `vllm/attention/` 目录
- `vllm/core/block_manager.py` — Block 管理器
- `vllm/worker/model_runner.py` — 看如何构建 block table

**重点：**
- Block 大小如何确定（通常 16 个 token）
- Block table 如何维护（逻辑 → 物理映射）
- 序列创建到完成的 block 分配/释放过程

---

### Day 25：vLLM PagedAttention 源码 — CUDA Kernel

**源码：**
- `vllm/attention/ops/` — PagedAttention CUDA kernel
- 重点看 `block_table` 和 `slot_mapping` 如何传给 kernel
- 理解 kernel 如何通过 block table 间接访问 KV cache

**参考：**
- FlashAttention 论文：https://arxiv.org/abs/2205.14135
- FlashAttention-2：https://arxiv.org/abs/2307.08691

---

### Day 26：源码加日志 — 追踪 KV Block 分配

**实操：**
- 在 `vllm/core/block_manager.py` 中添加日志
- 运行简单推理请求，收集日志
- 根据日志画出 block 分配流程图

---

### Day 27：Inference System 理论对照

**阅读：**
- AI-Infra `inference.md` — 重读 Inference System 部分
- AI-Infra `llm_serving.md` — 重读 continuous batching / prefill-decode 分离

**输出：**
- 论文方案 vs vLLM 实现对照表

---

### Day 28：Week 4 总结

**输出任务：**
- 逻辑 KV block → 物理 block 映射图
- 源码级笔记：token 生成时的 block 分配/释放流程
- 论文方案 vs vLLM 实现对照表

**预习：**
- Pybind11 官方文档：https://pybind11.readthedocs.io/

---

## Week 5：C++/Pybind11 + 训练系统概览

### Day 29：C++ 基础复习

**参考：**
- C++ Reference：https://en.cppreference.com/
- 重点复习：智能指针、RAII、模板、移动语义

---

### Day 30：Pybind11 入门

**文档：**
- Pybind11 官方文档 — Basics：https://pybind11.readthedocs.io/en/stable/basics.html
- Pybind11 官方文档 — Functions：https://pybind11.readthedocs.io/en/stable/advanced/functions.html

**安装：**
- `pip install pybind11`

---

### Day 31：Pybind11 进阶 — Buffer Protocol

**文档：**
- Pybind11 官方文档 — Buffers：https://pybind11.readthedocs.io/en/stable/advanced/pycpp/numpy.html
- NumPy C-API 文档：https://numpy.org/doc/stable/reference/c-api/

---

### Day 32：vLLM pybind.cpp 源码阅读

**源码：**
- `vllm/csrc/pybind.cpp` — C++ 算子暴露给 Python 的入口
- 追踪 paged_attention 从 Python 到 C++ 的调用链

---

### Day 33：C++ 矩阵加法库 + Pybind11 封装

**实操：**
- 写 C++ 矩阵加法库
- 用 Pybind11 封装
- Benchmark 对比：torch.add / numpy.add / C++ 实现

---

### Day 34：training.md 扫读 — 并行策略

**阅读：**
- AI-Infra 仓库 `training.md` — Training System 部分
- 重点理解：
  - 数据并行（Data Parallelism）
  - 模型并行（Tensor Parallelism）
  - 流水线并行（Pipeline Parallelism）
  - ZeRO、PipeDream、Mesh-TensorFlow

**论文推荐（选读）：**
- *"ZeRO: Memory Optimizations Toward Training Trillion Parameter Models"* — https://arxiv.org/abs/1910.02054
- *"PipeDream: Generalized Pipeline Parallelism for DNN Training"* — https://arxiv.org/abs/1806.03377

---

### Day 35：Week 5 总结

**输出任务：**
- vllm-cpp-playground 仓库（Pybind11 示例）
- vLLM Python → C++ 调用链笔记
- 并行策略对比表

**预习：**
- FP8 格式介绍
- Nsight Systems 安装

---

## Week 6：FP8 量化 + 4090 Profiling

### Day 36：FP8 格式学习

**文档：**
- vLLM FP8 文档：https://docs.vllm.ai/en/latest/quantization/fp8.html
- NVIDIA FP8 规范：https://arxiv.org/abs/2209.05433 (FP8 Formats for Deep Learning)

**关键概念：**
- E4M3（4 位指数 + 3 位尾数）：用于前向/推理，精度更高
- E5M2（5 位指数 + 2 位尾数）：用于反向/训练，范围更大

**浮点格式对比：**
| 格式 | 总位数 | 指数 | 尾数 | 范围 | 精度 |
|------|--------|------|------|------|------|
| FP32 | 32 | 8 | 23 | ±3.4e38 | 高 |
| FP16 | 16 | 5 | 10 | ±65504 | 中 |
| BF16 | 16 | 8 | 7 | ±3.4e38 | 低 |
| FP8 E4M3 | 8 | 4 | 3 | ±448 | 中低 |
| FP8 E5M2 | 8 | 5 | 2 | ±57344 | 低 |

---

### Day 37：compressed-tensors + llm-compressor 入门

**仓库：**
- compressed-tensors：https://github.com/vllm-project/compressed-tensors
- llm-compressor：https://github.com/vllm-project/llm-compressor

**安装：**
- `pip install llmcompressor`

**文档：**
- compressed-tensors README
- llm-compressor Quick Start

---

### Day 38：FP8 模型制作

**文档：**
- llm-compressor FP8 量化示例
- compressed-tensors FP8 规范

**代码参考：**
```python
from llmcompressor.modifiers.quantization import QuantizationModifier
from llmcompressor import oneshot

model_id = "meta-llama/Meta-Llama-3-8B"
recipe = QuantizationModifier(targets="Linear", scheme="FP8_DYNAMIC")
oneshot(model=model_id, recipe=recipe, output_dir="Meta-Llama-3-8B-FP8")
```

---

### Day 39：vLLM 加载 FP8 模型 + 性能测试

**文档：**
- vLLM FP8 加载文档：https://docs.vllm.ai/en/latest/quantization/fp8.html

**代码参考：**
```python
from vllm import LLM, SamplingParams

llm = LLM(model="Meta-Llama-3-8B-FP8", quantization="fp8")
params = SamplingParams(temperature=0.0, max_tokens=256)
outputs = llm.generate(["Hello, my name is"], params)
```

---

### Day 40：Nsight Systems 入门

**工具：**
- Nsight Systems (nsys) 文档：https://docs.nvidia.com/nsight-systems/
- 安装：`sudo apt install nsight-systems-cli` 或从 NVIDIA 官网下载

**教程：**
- NVIDIA Nsight Systems Quick Start Guide
- 博客：*"Profiling PyTorch with Nsight Systems"*

**命令参考：**
```bash
nsys profile -t cuda,nvtx -o matmul python matmul.py
nsys-ui matmul.nsys-rep  # GUI 查看
```

---

### Day 41：vLLM Profiling 实战

**实操：**
```bash
nsys profile -t cuda,nvtx -o vllm_profile python benchmark.py
```

**分析要点：**
- GPU Kernel 执行时间分布
- CPU-GPU 交互（数据传输）
- 计算瓶颈 vs 内存/通信瓶颈

---

### Day 42：Week 6 总结

**输出任务：**
- FP16 / AWQ-4bit / FP8 对比报告
- nsys 截图 + 性能分析

**预习：**
- NCCL 文档：https://docs.nvidia.com/deeplearning/nccl/
- Ring AllReduce 算法

---

## Week 7：分布式通信 + Good First Issue

### Day 43：分布式通信基础

**文档：**
- NCCL 官方文档：https://docs.nvidia.com/deeplearning/nccl/
- MPI 基础概念（了解即可）

**教程推荐：**
- *"Ring AllReduce 原理详解"* — 理解分布式训练中的通信原语
- *"Introduction to Distributed Training"* — 理解 AllReduce、AllGather、ReduceScatter

**通信原语对比：**
| 原语 | 数据量 | 适用场景 |
|------|--------|----------|
| Broadcast | O(model) | 广播参数 |
| AllReduce | O(model) | 梯度同步 |
| AllGather | O(model) | 收集分片参数 |
| ReduceScatter | O(model) | 分散梯度 |

---

### Day 44：vLLM Tensor Parallel 源码阅读

**文档：**
- vLLM 分布式推理文档：https://docs.vllm.ai/en/latest/serving/distributed_inference.html

**源码：**
- `vllm/distributed/` 目录
- `vllm/model_executor/parallel_utils/` — 并行工具

**重点理解：**
- 列并行（Column Parallel）：权重按列切分，输出需要 AllReduce
- 行并行（Row Parallel）：权重按行切分，输入需要 AllGather

---

### Day 45：vLLM Tensor Parallel 实验

**实操：**
- 多卡：`vllm serve model --tensor-parallel-size 2`
- 单卡：阅读 TP 代码逻辑
- Profiling：`nsys profile` 观察 TP 通信开销

---

### Day 46：浏览 vLLM Good First Issues

**链接：**
- vLLM Good First Issues：https://github.com/vllm-project/vllm/labels/good%20first%20issue
- vLLM Quantization Issues：https://github.com/vllm-project/vllm/labels/quantization

**筛选标准：**
- 难度适合你的背景（C++ / Python / 量化）
- 需要了解的代码模块可控
- 预计 1-2 天可以完成

---

### Day 47：认领 Issue + 本地复现

**实操：**
- 在 GitHub 上留言表示正在处理
- Fork 仓库，创建本地分支
- 复现问题，理解根因

---

### Day 48：实现修复/改进

**实操：**
- 编写修复代码
- 补充/更新单测
- 本地运行测试

---

### Day 49：Week 7 总结 + PR 准备

**输出任务：**
- 分布式通信笔记
- Issue 修复的本地分支
- 修复说明文档
- PR 描述草稿

---

## Week 8：提交 PR + 总结复盘

### Day 50：提交 PR

**参考：**
- vLLM Contributing Guide：https://github.com/vllm-project/vllm/blob/main/CONTRIBUTING.md
- GitHub PR 最佳实践

---

### Day 51：Code Review 响应

**实操：**
- 关注维护者反馈
- 修改代码、补充测试
- 保持沟通

---

### Day 52：AI-Infra 仓库回顾 + 系统地图更新

**阅读：**
- 回顾 AI-Infra 仓库全部目录
- 标记已读章节
- 选 1-2 个方向做深度阅读计划

---

### Day 53：vLLM 知识体系梳理

**输出任务：**
- vLLM 架构全景图
- 关键模块总结：量化 / PagedAttention / 异步调度 / TP / FP8

---

### Day 54：8 周总结文档撰写

**输出任务：**
- 完整的 8 周总结文档
- 实验数据汇总
- 代码仓库链接
- PR 链接和贡献记录

---

### Day 55：后续方向规划

**方向参考：**
- 算子优化：Triton / CUDA kernel 编写
  - Triton 教程：https://triton-lang.org/
- 分布式训练：DeepSpeed / Megatron-LM
  - DeepSpeed：https://github.com/microsoft/DeepSpeed
  - Megatron-LM：https://github.com/NVIDIA/Megatron-LM
- MLOps：Kubeflow / 模型部署流水线
  - Kubeflow：https://www.kubeflow.org/
- 更多 vLLM 贡献

---

### Day 56：最终复盘

**输出任务：**
- Top 5 收获 + Top 3 待改进
- 整理所有产出
- （可选）写博客分享学习经历

---

## 附录：核心仓库 & 资源清单

### 体系化地图
- AI-Infra-from-Zero-to-Hero：https://github.com/HuaizhengZhang/AI-Infra-from-Zero-to-Hero
  - `inference.md`、`llm_serving.md`、`training.md`、`infra.md`

### vLLM 生态
- vLLM 主仓库：https://github.com/vllm-project/vllm
- vLLM 文档：https://docs.vllm.ai/

### 量化 / FP8
- AutoAWQ：https://github.com/casper-hansen/AutoAWQ
- AutoGPTQ：https://github.com/AutoGPTQ/AutoGPTQ
- compressed-tensors：https://github.com/vllm-project/compressed-tensors
- llm-compressor：https://github.com/vllm-project/llm-compressor

### 算子 / 性能
- FlashAttention：https://github.com/Dao-AILab/flash-attention
- Triton：https://triton-lang.org/

### 调试 / Profiling
- Nsight Systems：https://docs.nvidia.com/nsight-systems/
- vLLM 架构文档：https://docs.vllm.ai/en/latest/getting_started/architecture.html

### C++ / Pybind11
- Pybind11：https://pybind11.readthedocs.io/
- C++ Reference：https://en.cppreference.com/
