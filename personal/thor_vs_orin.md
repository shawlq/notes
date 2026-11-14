NVIDIA \*\*Thor 和 Orin X（通常指 NVIDIA DRIVE AGX Orin / Jetson AGX Orin）\*\*的差异，不只是 TOPS 算力提升，还包括 GPU 架构、SM 数量、CUDA Core、Tensor Core、显存容量与带宽，以及 CPU 和内存子系统的升级。

对于智驾车端部署，尤其是 BEV、OCC、E2E、多路摄像头感知等场景，我建议重点比较以下几个维度：

1. GPU：SM 数量、CUDA Core、Tensor Core、架构。
2. AI 算力：FP16、BF16、FP8、INT8、FP4 TOPS。
3. 内存：容量、类型、带宽、显存共享机制。
4. CPU：核心数量与架构。
5. 功耗：算力与功耗比。
6. 对 TensorRT 模型部署的实际影响。

注意：Thor 和 Orin X 都有不同产品配置，尤其是 DRIVE Thor 与 Jetson AGX Thor 不能直接混为一谈。下面以车端平台常见的 Orin X 64GB 和 DRIVE Thor 高配版本为主要比较对象。

## 1. 核心硬件规格对比：DRIVE Thor vs DRIVE Orin X

先看 NVIDIA 官方公布的 DRIVE 平台参数。

对比对象：单颗 SoC，非多芯片组合。

| 硬件指标          | DRIVE Orin X      | DRIVE Thor       |
| ------------- | ----------------- | ---------------- |
| GPU 架构        | Ampere            | Blackwell        |
| CPU 架构        | Cortex-A78AE      | Neoverse V3AE    |
| CPU 核心数       | 12 核              | 14 核             |
| GPU CUDA Core | 2048              | 官方 DRIVE 规格未公开   |
| GPU SM 数量     | 16（按 Orin GPU 配置） | 官方 DRIVE 规格未公开   |
| Tensor Core   | 64                | 第五代 Tensor Core  |
| AI 算力 INT8    | 254 TOPS          | 1000 TOPS        |
| AI 算力 FP4     | 不支持               | 最高 2000 TFLOPS   |
| FP32 GPU 算力   | 约 5.2 TFLOPS      | DRIVE 规格未单独公布    |
| 系统内存          | 32GB              | 64GB             |
| 内存类型          | LPDDR5            | LPDDR5X          |
| 内存位宽          | 256-bit           | 256-bit          |
| 内存带宽          | 约 200 GB/s        | 273 GB/s         |
| DLA           | 2 个 NVDLA         | 不适用原 Orin DLA 架构 |
| PVA           | PVA v2            | PVA              |
| ISP 吞吐量       | 1.85 Gpixel/s     | 3.5 Gpixel/s     |

数据来源：NVIDIA DRIVE 官方平台规格及技术博客。

[image](https://www.google.com/s2/favicons?domain=https://developer.nvidia.com\&sz=32)

NVIDIA Developer+2



这里有两个重要说明：

- Orin X 的 254 TOPS 包含 DLA 算力，不能简单认为全部由 GPU 提供。
- Thor 的 1000 INT8 TOPS 和 2000 FP4 TFLOPS 是不同精度下的峰值，不能直接把两者作为同一计算指标比较。

## 2. GPU 架构：SM、CUDA Core、Tensor Core

这是我认为最值得深入比较的部分。

[Exploring ways to optimize compute shaders - Part 1. :: The GutHub — My humble blog](https://images.openai.com/static-rsc-4/s0iB59Uy1W9MG3ISsG4NLPZPC2ARtdmesJPChJtOC9vRRkUjT5A9jRlm7cAVH1BLOUwzsgwtaumzr5-3HZZ4VMN5MK2egQlNFWjo7IxXyMziv7ZNdMAqGTdc9wLUGoFNsy2jHoPbJZaDn9frUz2LaEeio9cvrul4TEG2daUSeo0?purpose=inline)

Orin X：Ampere 架构

- 2048 CUDA Cores
- 16 SM
- 64 Tensor Cores
- 第二代 Tensor Core
- 支持 FP16、INT8 等计算模式

[NVIDIA GeForce RTX 5080: Обзор, тесты, технологии Blackwell](https://images.openai.com/static-rsc-4/SZcrFwx8zhO4FTHztbKY3HyVeAEoQN8wzEfQ_RUJpisdphoIgm6gspltnIFfyENGEZ7xHx1gITzKBE3nHwe17-9jSQyJ5ILmwGv4J_utey7cK7qSceeaDhza6sZ8Q27AXfUBYSXZKknBZSVHJOyHTzAZrgwHU6pN1_CMsH22fU0?purpose=inline)

Thor：Blackwell 架构

- 第五代 Tensor Core
- 支持 FP4、FP8、FP16、BF16、INT8 等
- 支持 MIG
- 支持更先进的计算压缩和稀疏计算机制

[image](https://www.google.com/s2/favicons?domain=https://www.nvidia.com\&sz=32)

NVIDIA+2

### Thor 的 SM 数量到底是多少？

这里需要特别注意：NVIDIA 有 Jetson Thor 和 DRIVE Thor 两套产品规格，不能混用。

对于 Jetson T5000：

| 指标             | 数值                        |
| -------------- | ------------------------- |
| GPC            | 3                         |
| TPC            | 10                        |
| SM             | 20                        |
| CUDA Core      | 2560                      |
| CUDA Core / SM | 128                       |
| Tensor Core    | 官方资料列出 96 个，但内部计数口径存在说明差异 |

Jetson T5000 的 2560 CUDA Core 是官方数据。其 GPU 运行时也有 20 SM 的实际查询记录。

[image](https://www.google.com/s2/favicons?domain=https://developer.download.nvidia.com\&sz=32)

NVIDIA Developer Download+1



对于 DRIVE Thor，不建议直接把 Jetson T5000 的 20 SM、2560 CUDA Core 当成 DRIVE Thor 的已确认规格。 NVIDIA 的 DRIVE 平台公开规格没有给出这组 GPU 详细参数。

## 3. AI 算力对比

INT8 峰值算力

Orin X

254 TOPS

Thor

1000 TOPS

# 约 3.94 倍

按官方 INT8 峰值规格计算，并非相同模型的实际性能提升倍数。

Thor 的重要变化是：

- Ampere → Blackwell，Tensor Core 架构升级。
- 引入 FP4 计算能力，适合更低精度的 AI 推理。
- FP8、BF16 等精度选择更加丰富。
- 稀疏计算可以进一步提高峰值吞吐量，但需要模型和算子支持。

对于智驾模型而言，FP4 并不意味着可以直接把现有 FP16 模型转成 FP4 就获得数倍加速。需要考虑精度损失、量化方案、TensorRT 支持以及实际算子覆盖情况。

## 4. 内存容量与带宽对比

对于 BEV、OCC、E2E 等智驾模型，内存不仅决定模型能否装载，还会影响多路相机输入、特征图读写和多模型并行推理。

| 指标   | Orin X    | DRIVE Thor | 提升    |
| ---- | --------- | ---------- | ----- |
| 内存容量 | 32GB      | 64GB       | 2倍    |
| 内存类型 | LPDDR5    | LPDDR5X    | 升级    |
| 内存位宽 | 256-bit   | 256-bit    | 相同    |
| 内存频率 | 约 3200MHz | 4266MHz    | 提升    |
| 带宽   | 200GB/s   | 273GB/s    | 1.37倍 |

[image](https://www.google.com/s2/favicons?domain=https://developer.nvidia.com\&sz=32)

NVIDIA Developer+2

这里说的内存是统一系统内存（Unified Memory），并不是独立显卡上的 GDDR 显存。

CPU、GPU、DLA 等硬件单元共享 SoC 内存资源，但各自的访问路径、缓存机制和带宽竞争情况不同。

对智驾推理的影响：

- 模型权重：Thor 可以容纳更大的模型。
- Feature Map：更大的内存容量有利于多路模型同时运行。
- 多摄像头：更大的内存带宽有利于图像和特征数据搬运。
- 多模型调度：可以为不同模型、Buffer、Tensor 分配更多资源。

特别是 BEV/OCC 网络，实际性能不一定受 TOPS 限制，可能更容易受到显存带宽、数据搬运和内存访问效率限制。

## 5. CPU 性能对比

| 指标     | Orin X           | DRIVE Thor             |
| ------ | ---------------- | ---------------------- |
| CPU 架构 | ARM Cortex-A78AE | ARM Neoverse V3AE      |
| 核心数    | 12               | 14                     |
| 指令集    | ARMv8.2-A        | ARMv9.2-A              |
| L2/L3  | 3MB L2 + 6MB L3  | 每核 1MB L2 + 16MB 共享 L3 |
| CPU 性能 | 基准               | 约 2.3 倍 SPECrate       |

NVIDIA 给出的 CPU 性能提升约为 2.3 倍，这是 SPEC CPU 2017 整数吞吐率估算值，不代表所有 C++ 业务代码都能提升 2.3 倍。

[image](https://www.google.com/s2/favicons?domain=https://developer.nvidia.com\&sz=32)

NVIDIA Technical Blog+1



对于车端推理系统，这个提升也很有意义：

- CPU 预处理和后处理。
- 多模型调度。
- TensorRT Engine 执行管理。
- 图像数据搬运。
- 多线程任务管理。
- ROS2 通信与消息处理。

## 6. 对你们现有智驾推理业务的影响

结合你们目前 Thor 平台上的多模型部署场景，可以这样理解：

| 业务场景    | Orin X → Thor 的主要收益            |
| ------- | ------------------------------ |
| OD / LD | 更高的 Tensor Core 吞吐能力           |
| OCC     | 更大的内存容量和带宽                     |
| BEV     | 更强的矩阵计算能力和内存子系统                |
| E2E     | FP8、FP4 等低精度计算选择               |
| 多路相机    | ISP 吞吐量从 1.85 提升至 3.5 Gpixel/s |
| 多模型并行   | 更大的内存空间和更强的 CPU                |
| 图像编解码   | 更高的视频编解码吞吐量                    |

[image](https://www.google.com/s2/favicons?domain=https://developer.nvidia.com\&sz=32)

NVIDIA Developer+1

### 对 Eko 推理引擎的几个建议

从 Orin 迁移到 Thor，不建议只关注模型转换和 TensorRT Engine 兼容性。

我认为有四个值得重点投入的方向：

1. Tensor Core 利用率分析：针对 FP16、BF16、FP8、INT8 分别建立性能基线。
2. 内存带宽分析：使用 Nsight Systems / Nsight Compute 分析 DRAM Throughput、L2 Cache、Memory Bound。
3. 多模型并行调度：充分利用 Thor 更大的内存容量，同时避免多个模型争抢 GPU 和内存带宽。
4. 算子优化：针对 Blackwell 架构重新评估 TensorRT Plugin、CUDA Kernel、CUDA Graph 的实现。

## 7. 总结：Thor 相比 Orin X 究竟提升多少？

| 对比维度      | Thor 相对 Orin X     |
| --------- | ------------------ |
| INT8 峰值算力 | 约 3.94 倍           |
| 系统内存      | 2 倍                |
| 内存带宽      | 1.37 倍             |
| CPU 整数吞吐率 | 约 2.3 倍            |
| ISP 吞吐量   | 约 1.89 倍           |
| GPU 架构    | Ampere → Blackwell |
| 精度支持      | 增加 FP8、FP4 等能力     |

我的判断：Thor 并不是简单地把 Orin X 的 GPU 放大几倍，而是一次计算架构升级。

尤其值得关注的是 FP8/FP4、第五代 Tensor Core、更大的内存容量以及 Blackwell 的缓存和数据压缩机制。

对于你们 Eko 的长期建设，应该把 Thor 当成一个新的 GPU 架构平台来做性能优化，而不是仅仅做 Orin Engine 的迁移。

### 8. 推荐参考的 NVIDIA 官方资料

- [DRIVE AGX Thor / Orin 官方规格对比](https://developer.nvidia.com/drive/agx?trk=public_post_comment-text\&utm_source=chatgpt.com)
- [DRIVE AGX Thor 平台技术规格 PDF](https://developer.download.nvidia.com/drive/docs/nvidia-drive-agx-thor-platform-for-developers.pdf?utm_source=chatgpt.com)
- [Jetson AGX Orin 技术规格](https://www.nvidia.com/content/dam/en-zz/Solutions/gtcf21/jetson-orin/nvidia-jetson-agx-orin-technical-brief.pdf?utm_source=chatgpt.com)
- [Thor SoC Technical Reference Manual](https://cdck-file-uploads-global.s3.dualstack.us-west-2.amazonaws.com/nvidia/original/4X/a/0/8/a0808d84bcb2524b5ee54ddea208b294887cd42e.pdf?utm_source=chatgpt.com)

还有一个值得进一步确认的问题：

你现在使用的是 DGX Thor、DRIVE Thor，还是 Jetson AGX Thor？

因为你之前提到的 Thor U 是 DRIVE OS 7.0.4、TensorRT 10.13.1 环境。如果你现在要分析的是这套车端平台，我建议进一步整理一份 DRIVE Thor vs Orin X 的 GPU 微架构对比，重点包括 SM、Warp Scheduler、CUDA Core、Tensor Core、L1/L2 Cache、Shared Memory、显存带宽，以及 FP16/INT8 实际计算能力。这样会比单纯比较 TOPS 更有利于指导 Eko 的 Kernel 优化。