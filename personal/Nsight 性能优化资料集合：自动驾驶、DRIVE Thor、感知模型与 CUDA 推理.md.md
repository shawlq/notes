# Nsight 性能优化资料集合：自动驾驶、DRIVE Thor、感知模型与 CUDA 推理

> 版本：v1.0  
> 整理日期：2026-11-16  
> 目标：收集约 100 项 NVIDIA 官方或一手技术资料，优先覆盖自动驾驶、BEV、多摄像头视觉、感知模型、TensorRT、DRIVE 平台，以及 Nsight Systems / Nsight Compute 性能分析。  
> 文件名：`nsight性能优化集合.md`

---

## 0. 阅读说明与资料质量分级

本集合不是“100 篇全部都是 Nsight 成功案例”的虚假清单。公开资料中，真正同时满足“使用 Nsight 分析 + 给出明确根因 + 给出优化措施 + 有量化结果”的案例数量远少于 100。因此按可复用程度收录三类资料：

- **A 类：直接案例**——文章明确使用 Nsight，分析性能问题，并给出优化措施；有公开量化结果时列出结果。
- **B 类：方法/教程**——NVIDIA 官方讲解 Nsight、CUDA 性能分析或可复现实验，适合建立分析方法。
- **C 类：技术参考**——CUDA、TensorRT、DRIVE、内存/调度/算子官方文档，可与 Nsight 结合使用，但不一定是独立的 Nsight 案例。

### 资料可信度和迁移边界

1. 优先选择 NVIDIA Technical Blog、NVIDIA Developer、NVIDIA 官方文档、NVIDIA GTC/培训资料。
2. 对非 Thor 平台的性能数字，只能作为方法参考，不能直接推算 Thor 上的收益。
3. RTX / H100 / V100 / Orin / DRIVE Thor 的架构、内存层级、SM 数、频率、功耗约束不同；必须在目标平台重新测量。
4. “SM Active”“GPU Utilization”“Kernel Occupancy”“Tensor Core Utilization”是不同指标，不能互相替代。
5. Nsight Compute 的硬件计数器采集可能触发 Kernel Replay 或序列化，改变并发行为；多模型端到端调度应优先以 Nsight Systems 和真实业务时延为准。
6. 文章中没有公开优化前后数字的，明确写“未公开”，不补造数据。

### 优先级定义

- **P0：必须先读**——直接涉及自动驾驶/BEV/多摄像头/多模型/DRIVE，或与当前 Thor 多 Engine 竞争问题高度相似。
- **P1：强相关**——TensorRT、视觉 Pipeline、CUDA Stream、CPU/GPU 协同、Kernel/访存优化。
- **P2：基础方法**——Nsight 工具、CUDA 指标、通用优化技术。
- **P3：扩展参考**——HPC、通用 GPU 算法、较旧但仍有价值的实践。

---

# 第一部分：自动驾驶、BEV、视觉感知、DRIVE 平台（最高优先级）

## 1. [P0][A] Optimizing a Neural Reconstruction Pipeline Using NVIDIA Nsight Developer Tools
- 链接：https://developer.nvidia.com/blog/optimizing-a-neural-reconstruction-pipeline-using-nvidia-nsight-developer-tools/
- 领域：自动驾驶仿真、多传感器重建、Camera / LiDAR、Nsight Systems + Nsight Compute。
- 问题与发现：NuRec 的 `interpolate` 前向流程产生大量小 Kernel；频繁 `cudaStreamSynchronize` 使 CPU/GPU 工作不能充分重叠；反向 `renderBackward` Kernel Occupancy 约 15%，Camera 与 LiDAR 使用相同资源配置但访问模式不同。
- 优化：融合大量小 Kernel；移除不必要同步；按 Camera / LiDAR 拆分 Kernel；调整 `launch_bounds`、寄存器和 Shared Memory 配置。
- 公开结果：插值融合接近 50 倍加速；Occupancy 从约 15% 提升到 30%～50%；最长 LiDAR Kernel 从 31ms 降到 18ms。
- Thor/Eko 参考：**极高**。可直接借鉴“先 nsys 看整条多传感器 Pipeline，再 ncu 看热点 Kernel”的流程。特别适合 OD/LD/ADAS/TRL 多模型执行链路和共享 Backbone 特征处理。

## 2. [P0][A] Accelerating BEV Pooling on NVIDIA GPUs for Physical AI Applications
- 链接：https://developer.nvidia.com/blog/accelerating-bev-pooling-on-nvidia-gpus-for-physical-ai-applications/
- 领域：自动驾驶 BEV、多摄像头特征投影、TensorRT Plugin、Nsight Compute。
- 问题与发现：BEV Pooling 是 Gather/Scatter 密集型算子，包含重复索引读取、Scatter 流量、随机访存、运行时整数除法和 Atomic 写入。工作集是否放得进 L2 会改变瓶颈。
- 优化：五数组 INT32 Scatter Map；预计算索引；减少重复 Depth Load；按 Interval 归属输出写入，减少/避免 Atomic；按 DRAM-bound 与 L2-resident 两种内存状态选择 Kernel；FP16/FP8 路径分别调优。
- 公开结果：文章在 RTX A6000 与 RTX PRO 6000 Blackwell Max-Q 上展示了最高数十倍相对 V2 的性能改善；典型配置中 Blackwell TensorRT Plugin V2 FP16 为 274µs，V3 FP16 为 17.3µs，V3 FP8 为 16.4µs。文章说明早期 edge 实验中 FP16 路径可迁移至 DRIVE AGX Thor，但 FP8 不保证有收益。
- Thor/Eko 参考：**最高**。这是本集合中最贴近感知部署和 Thor 的算子优化案例。重点研究 Scatter Map、索引预计算、数据复用、L2 命中、寄存器压力和精度路径。

## 3. [P0][A] Accelerating Vision AI Pipelines with Batch Mode VC-6 and NVIDIA Nsight
- 链接：https://developer.nvidia.com/blog/accelerating-vision-ai-pipelines-with-batch-mode-vc-6-and-nvidia-nsight/
- 领域：多路视觉、图像解码、Batch、Pipeline、Nsight Systems + Nsight Compute。
- 问题与发现：N 个图像分别创建 N 个 Decoder，产生大量小 Kernel 和 CUDA API Launch；CPU 处理、PCIe 上传、GPU 解码之间未形成高效 Pipeline。
- 优化：从 N 个 Decoder 改为单个 Batch Decoder；Minibatch Pipelining；把部分窄层级处理迁移到 GPU；减少 CPU 侧动态尺寸处理与同步；用 NCU 优化 Range Decoder Kernel。
- 公开结果：Range Decoder Kernel 约 20% 加速；Batch 模式在大 Batch 下每图解码时间最高降低约 85%。
- Thor/Eko 参考：**极高**。适用于 11 路摄像头预处理、图像解码、Batch 组织和 CPU/GPU 数据管线。注意车载实时系统的 Batch 等待时间必须纳入端到端时延。

## 4. [P0][A] Build High-Performance Vision AI Pipelines with NVIDIA CUDA-Accelerated VC-6
- 链接：https://developer.nvidia.com/blog/build-high-performance-vision-ai-pipelines-with-nvidia-cuda-accelerated-vc-6/
- 领域：视觉数据解码、图像 Pipeline、CUDA、Nsight。
- 问题与发现：原有 OpenCL/CPU 路径存在上下文切换、Upsampling Chain Kernel Launch 开销、分支发散和 Kernel 并行度限制。
- 优化：迁移到 CUDA；降低跨计算框架切换成本；利用图像分辨率层级和 ROI 选择性解码；通过 Nsight Systems / Compute 定位后续优化点。
- 公开结果：CUDA 单图解码相对 CPU 最多约 13 倍，相对 OpenCL GPU 约 1.2～1.6 倍（文章测试平台为 RTX PRO 6000 Blackwell Server Edition）。
- Thor/Eko 参考：高。可借鉴摄像头数据入口、解码与 Tensor 输入之间的 Pipeline 设计。

## 5. [P0][A] Optimizing the CV Pipeline in Automotive Vehicle Development Using the PVA Engine
- 链接：https://developer.nvidia.com/blog/optimizing-the-cv-pipeline-in-automotive-vehicle-development-using-the-pva-engine/
- 领域：DRIVE Orin、汽车 CV、PVA、VIC、DLA、GPU 资源分配。
- 问题与发现：图像 Layout 转换、YUV-RGB、Mosaic/Mask 等预处理和后处理占用 GPU/VIC；CPU 逐个提交任务和同步造成 Pipeline 开销。
- 优化：将适合的 CV 操作迁移到 PVA；NvStreams / NvSciSync 跨硬件引擎同步；Zero-copy Buffer；PVA 任务批量提交和双 VPU 调度。
- 公开结果：NIO 案例 GPU 资源使用降低约 10%，VIC 资源释放；PVA 总体利用率约 25%。
- Thor/Eko 参考：高，但硬件代际有差异。Orin PVA 案例不能直接等同 Thor PVA 能力；应核对 DRIVE OS / PVA Runtime / SDK 版本及许可证。

## 6. [P0][A] How to Eliminate Pipeline Friction in AI Model Serving
- 链接：https://developer.nvidia.com/blog/how-to-eliminate-pipeline-friction-in-ai-model-serving/
- 领域：AI 推理服务、TensorRT、Nsight Deep Learning Designer、Nsight Systems。
- 核心方法：先用 `trtexec` 测模型独立基线；再用 Nsight Deep Learning Designer 看 Layer 级耗时；最后用 Nsight Systems 看 CPU/GPU Timeline、Preprocess、同步和 GPU 空闲。
- 重要判断：模型单独运行已经慢，优先查模型/Engine；单独运行快、集成后慢，优先查系统 Pipeline、CPU 调度和同步。
- Thor/Eko 参考：极高。建议把该工具分工作为 Eko 性能定位标准流程。

## 7. [P1][A] Measuring the GPU Occupancy of Multi-stream Workloads
- 链接：https://developer.nvidia.com/blog/measuring-the-gpu-occupancy-of-multi-stream-workloads/
- 领域：多 Stream、SM Active、Nsight Systems GPU Metrics。
- 内容：用 Nsight Systems GPU Metrics 采集 SM Active，再通过 DataService / SQLite 数据计算 Gross、Net、Effective GPU Utilization。
- 重要边界：SM Active 是设备层面活动采样，不直接说明每个模型占用了多少 SM，也不等于 Kernel Occupancy。
- Thor/Eko 参考：极高。适用于 OD/LD/ADAS/TRL 并发时分析 GPU 是否真正被持续使用。

## 8. [P1][A] Optimizing CUDA Memory Transfers with NVIDIA Nsight Systems
- 链接：https://developer.nvidia.com/blog/optimizing-cuda-memory-transfers-with-nsight-systems/
- 领域：多 Stream、CPU→GPU 迁移、Memcpy、GROMACS。
- 问题与发现：Memory Transfer 占据大量 Stream 时间；不同 Stream 传输串行；Kernel 间出现约 3ms 空隙。
- 优化：将部分 CPU 计算迁移到 GPU；调整 Stream 和传输组织；让 Kernel 与传输重叠。
- 公开结果：迭代时间由 19ms 降至 10ms；最长 Kernel 仍约 10ms，收益主要来自 Pipeline 空隙和传输重叠。
- Thor/Eko 参考：极高。可映射到图像预处理、特征传递、多个 Engine 之间的 Buffer 传递。

## 9. [P1][A] Nsight Systems Exposes New GPU Optimization Opportunities
- 链接：https://developer.nvidia.com/blog/nsight-systems-exposes-gpu-optimization/
- 领域：GPU Starvation、CPU Thread、CUDA API、同步。
- 问题与发现：GPU 时间线存在空洞；通过 Kernel 与 CPU Thread / CUDA API 关联定位未及时提交工作的线程；发现 `cudaMemcpyAsync` 后紧跟不必要的 `cudaStreamSynchronize`。
- 优化：移除冗余同步，合理使用 CUDA Event，改善 CPU 提交任务的节奏。
- 公开结果：VMD 团队报告超过 3 倍性能提升（具体收益依赖工作负载）。
- Thor/Eko 参考：极高。尤其适合排查 TRT Enqueue、Stream Wait、同步封装和 CPU 调度。

## 10. [P1][A] Streamlining GPU Porting for EDF's Fluid Dynamics Simulations with NVIDIA Nsight Profilers
- 链接：https://developer.nvidia.com/blog/streamlining-gpu-porting-for-edfs-fluid-dynamics-simulations-with-nvidia-nsight-profilers/
- 领域：CPU→GPU 迁移、Unified Memory、Nsight Systems。
- 问题与发现：CPU-only Gradient 阶段及 Unified Memory Page Migration 影响整体流程。
- 优化：把 Gradient 计算迁移到 GPU，并降低后续数据迁移。
- 公开结果：Gradient 计算 12.3ms 降至 0.69ms，约 18 倍；下游 Kernel 约 4 倍加速。
- Thor/Eko 参考：高。可借鉴把 CPU 侧预处理/后处理迁移至 GPU 的判断方式，但需评估 Thor CPU/GPU 共享内存架构。

## 11. [P1][A] Tracxpoint Deep Learning GPU-Starvation Case (Nsight Systems Customer Story)
- 链接：https://developer.nvidia.com/nsight-systems
- 领域：深度学习训练、GPU Starvation、GPU 利用率。
- 问题与发现：NVIDIA Nsight Systems 官方页面收录 Tracxpoint 客户案例：训练期间 GPU 出现 Starvation，需要通过 CPU/GPU 全链路分析定位冷点。
- 公开结果：客户反馈 GPU 利用率提升到 90% 以上；一个训练任务由约 600 分钟降至约 90 分钟。该数字为客户案例陈述，具体工作负载和测试条件需以客户原始资料为准。
- Thor/Eko 参考：中高。可借鉴“GPU 空闲时向 CPU Thread / 提交路径回溯”的诊断思路；训练工作负载不能直接代表车载推理。

## 12. [P1][A] Improving GPU Performance by Reducing Instruction Cache Misses
- 链接：https://developer.nvidia.com/blog/improving-gpu-performance-by-reducing-instruction-cache-misses-2/
- 领域：Nsight Compute、Instruction Cache、Warp Stall。
- 问题与发现：Smith-Waterman 工作负载在规模增大时出现 Instruction Cache Miss；过度 Loop Unrolling 扩大热点指令 Footprint，`No Instruction` Stall 上升。
- 优化：减少外层循环展开，保留适度内层展开。
- 公开结果：热点指令规模由 39,360 降至 16,912，显著减少 Instruction Cache Miss。
- Thor/Eko 参考：中高。适用于复杂 Plugin、过度模板展开、长 CUDA Kernel 的指令规模分析。

## 13. [P1][A] How to Improve CUDA Kernel Performance with Shared Memory Register Spilling
- 链接：https://developer.nvidia.com/blog/how-to-improve-cuda-kernel-performance-with-shared-memory-register-spilling/
- 领域：CUDA 13、Register Pressure、Local Memory、Shared Memory、Nsight Compute。
- 问题与发现：高寄存器使用导致 Register Spill，访问 Local Memory 并增加 L2 压力。
- 优化：CUDA 13.0 的 Shared Memory Register Spilling，让部分 Spill 优先落入 Shared Memory。
- 公开结果：示例 Kernel 8.35µs 降至 7.71µs，约 7.76%；QUDA Kernel 常见收益约 5%～10%。
- Thor/Eko 参考：中高。需核对 DRIVE OS CUDA 版本、编译器和目标架构是否支持该特性。

## 14. [P1][A] Accelerating HPC Applications with NVIDIA Nsight Compute Roofline Analysis
- 链接：https://developer.nvidia.com/blog/accelerating-hpc-applications-with-nsight-compute-roofline-analysis/
- 领域：Roofline、Arithmetic Intensity、Compute Bound / Memory Bound。
- 内容：用 Roofline 判断 Kernel 是计算受限还是访存受限；通过循环重构提高 Arithmetic Intensity。
- 公开结果：案例把 Arithmetic Intensity 从 7.39 提升到 20 FLOP/Byte；优化高延迟除法和绝对值操作后，运行时间约减半。
- Thor/Eko 参考：高。适合区分 TRT Plugin 是算力瓶颈、DRAM 带宽瓶颈还是缓存/延迟瓶颈。

## 15. [P1][A] Analysis-Driven Optimization: Preparing for Analysis with Nsight Compute, Part 1
- 链接：https://developer.nvidia.com/blog/analysis-driven-optimization-preparing-for-analysis-with-nvidia-nsight-compute-part-1/
- 领域：Nsight Compute 入门、Grid/Block、并行度。
- 问题与发现：Kernel 初始 Grid 只有一个 Block，无法利用 GPU 的 SM 并行能力。
- 优化：重新组织 Grid，增加足够 Block 并行度。
- Thor/Eko 参考：高。适合自定义 CUDA Plugin 的基本分析训练。

## 16. [P1][A] Analysis-Driven Optimization: Analyzing and Improving Performance with Nsight Compute, Part 2
- 链接：https://developer.nvidia.com/blog/analysis-driven-optimization-analyzing-and-improving-performance-with-nvidia-nsight-compute-part-2/
- 领域：访存合并、LSU、Warp Stall、Shared Memory。
- 问题与发现：Global Memory Transaction 效率低，示例出现约 8:1 的 Transaction 效率问题；`Stall LG Throttle` 高。
- 优化：调整 Warp-stride / Block-stride；重新分配数据集到多个 Block；继续观察瓶颈转移。
- 公开结果：示例 Kernel 2.92s → 78.9ms → 21.6ms。
- Thor/Eko 参考：高。对 NHWC/NCHW、Tensor Layout、Gather/Scatter、非合并访存尤其有价值。

## 17. [P1][A] Analysis-Driven Optimization: Finishing the Analysis with Nsight Compute, Part 3
- 链接：https://developer.nvidia.com/blog/analysis-driven-optimization-finishing-the-analysis-with-nvidia-nsight-compute-part-3/
- 领域：Long Scoreboard、Memory Latency、Warp Shuffle、cuBLAS。
- 问题与发现：主要 Stall 转为 `Long Scoreboard`，说明等待 L1TEX 数据依赖；Shared Memory Reduction 也产生开销。
- 优化：两阶段 Warp Shuffle Reduction；把计算重构为 Vector Averaging + cuBLAS SGEMM。
- 公开结果：21.6ms → 12.7ms → 5.53ms。
- Thor/Eko 参考：高。展示了瓶颈会随着优化而变化，不能只针对最初的 Stall 指标持续优化。

## 18. [P1][A] The Modern CUDA Toolbox in Practice: A Step-by-Step Optimization Walkthrough
- 链接：https://developer.nvidia.com/blog/the-modern-cuda-toolbox-in-practice-a-step-by-step-optimization-walkthrough/
- 领域：图像处理、NVTX、Nsight Systems、Streams、Pinned Memory、CCCL/CUB。
- 问题与发现：Median Kernel、内存分配、Host-to-Device Copy 和串行执行限制图像处理 Pipeline。
- 优化：用 CUB 替换手写 Kernel；内存池；Pinned Host Memory；每个工作线程独立 Stream；异步 Copy。
- 公开结果：示例总时间由 6.8s 降至约 23ms（多项优化叠加，不应将收益归因于单一 Nsight 操作）。
- Thor/Eko 参考：极高。可借鉴图像预处理 Pipeline 的逐层优化顺序。

## 19. [P1][A] Profiling and Optimizing Deep Neural Networks with DLProf and PyProf
- 链接：https://developer.nvidia.com/blog/profiling-and-optimizing-deep-neural-networks-with-dlprof-and-pyprof/
- 领域：深度学习、ResNet50、Nsight Systems、TensorFlow/PyTorch。
- 内容：通过 DLProf / PyProf / Nsight Systems 识别训练算子和数据路径；对比 TF32、XLA、AMP 的性能变化。
- Thor/Eko 参考：中高。可借鉴把模型图节点与 CUDA Kernel 关联的方式。

## 20. [P1][A] Unleashing the Power of NVIDIA Ampere Architecture with NVIDIA Nsight Developer Tools
- 链接：https://developer.nvidia.com/blog/unleashing-power-of-nvidia-ampere-architecture-with-nsight-developer-tools/
- 领域：Ampere、Nsight Systems、Nsight Compute、Roofline、GPU 架构。
- 内容：介绍系统级 GPU Starvation、Roofline、架构指标和新硬件特性的分析方式。
- Thor/Eko 参考：中高。架构代际不同，指标含义和分析方法可迁移，具体阈值不可照搬。

---

# 第二部分：Nsight Systems——系统时间线、调度、CPU/GPU 协同、Stream 与 Pipeline

## 21. [P1][B] Three Things You Need to Know About Nsight Systems
- 链接：https://developer.nvidia.com/blog/three-things-you-need-to-know-about-nsight-systems/
- 重点：Nsight Systems 是性能分析第一站；查 GPU Starvation、不必要同步、CPU 并行度不足、昂贵算法；使用 NVTX 关联业务阶段。
- Thor 用法：先看整条 OD/LD/ADAS/TRL 时间线，再决定是否进入 NCU。

## 22. [P1][B] CUDA Pro Tip: Generate Custom Application Profile Timelines with NVTX
- 链接：https://developer.nvidia.com/blog/cuda-pro-tip-generate-custom-application-profile-timelines-nvtx/
- 重点：用 NVTX Range 标记业务阶段、函数、线程和资源；NVTX3 可用于现代 Nsight Systems。
- Thor 用法：建议在 Eko 中标记 `Preprocess`、`Backbone`、`OD`、`LD`、`ADAS`、`TRL`、`DLP`、`Postprocess`，并将 Engine 名称和帧 ID 关联到 Timeline。

## 23. [P1][B] Getting Started with CUDA Graphs
- 链接：https://developer.nvidia.com/blog/cuda-graphs/
- 重点：通过 CUDA Graph 捕获多个 Kernel、Memcpy 和同步操作，减少 Host Launch Overhead。
- Thor 用法：当单次推理由大量短 Kernel 组成时值得评估；动态 Shape、动态地址、资源生命周期和多 Stream 依赖需要验证。

## 24. [P1][B] Advanced API Performance: Async Compute and Overlap
- 链接：https://developer.nvidia.com/blog/advanced-api-performance-async-compute-and-overlap/
- 重点：异步计算需要结合硬件单元 Throughput、寄存器、Shared Memory、数据路径判断，不能只看 Occupancy。
- Thor 用法：用于评估多个 Engine 是否存在互补资源，而不是简单把所有任务放到不同 Stream。

## 25. [P1][B] CUDA Streams Best Practices（CUDA Programming Guide）
- 链接：https://docs.nvidia.com/cuda/cuda-c-programming-guide/
- 重点：Stream 顺序语义、异步执行、Event 依赖、默认 Stream 行为。
- Thor 用法：排查 TRT Engine 是否因为默认 Stream 或跨 Stream Event 依赖而串行。

## 26. [P1][B] CUDA Runtime API — Stream Management
- 链接：https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__STREAM.html
- 重点：Stream 创建、同步、优先级、查询和销毁 API。
- Thor 用法：逐一检查 Eko Stream 管理是否存在过度同步、重复创建销毁或生命周期问题。

## 27. [P1][B] CUDA Runtime API — Event Management
- 链接：https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__EVENT.html
- 重点：Event 记录、跨 Stream Wait、计时和依赖。
- Thor 用法：用 Event 建立模型间依赖，不要用 CPU 全局同步替代局部依赖。

## 28. [P1][B] CUDA C++ Best Practices — Overlapping Transfers and Computation
- 链接：https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/
- 重点：Pinned Memory、异步 Copy、Copy/Compute Overlap、多 Stream。
- Thor 用法：分析图像输入、特征输出和模型输入是否可重叠。

## 29. [P1][B] Nsight Systems User Guide
- 链接：https://docs.nvidia.com/nsight-systems/UserGuide/
- 重点：CPU Sampling、CUDA API、GPU Timeline、GPU Metrics、OS Runtime、NVTX、SQLite 导出。
- Thor 用法：作为 nsys 报告解释的权威手册；注意 GPU Metrics 对设备采样与 Context Trace 的粒度不同。

## 30. [P1][B] Nsight Systems Get Started
- 链接：https://developer.nvidia.com/nsight-systems/get-started
- 重点：Nsight Systems 视频教程、GPU 应用 Profiling、内存分配、NCCL 和多节点分析。
- Thor 用法：建议团队新人先完成 Intro to Systems 和 Timeline 分析课程。

## 31. [P1][B] CUDA Developer Tools Tutorials
- 链接：https://developer.nvidia.com/blog/new-video-series-cuda-developer-tools-tutorials/
- 重点：Nsight Tools Ecosystem、Nsight Systems、Nsight Compute 的入门视频。
- Thor 用法：用于统一团队 Profiling 基础概念。

## 32. [P1][B] Nsight Systems — GPU Metrics
- 链接：https://docs.nvidia.com/nsight-systems/UserGuide/
- 重点：SM Active、Tensor Core Activity、Warp Occupancy、PCIe/NVLink/GPU Memory Throughput 等设备级采样。
- Thor 用法：评估多模型并发期间 SM 活动和内存带宽随时间的变化。先确认 Thor/DRIVE OS 版本支持哪些 Metrics。

## 33. [P1][B] Nsight Systems — CUDA API Trace
- 链接：https://docs.nvidia.com/nsight-systems/UserGuide/
- 重点：CUDA Runtime / Driver API 与 GPU Kernel 关联。
- Thor 用法：把 `cudaLaunchKernel`、Memcpy、Event、Stream Synchronize 与 TRT enqueue 调用链对齐。

## 34. [P1][B] Nsight Systems — NVTX Trace
- 链接：https://docs.nvidia.com/nsight-systems/UserGuide/
- 重点：自定义业务 Range、Domain、Category、Payload。
- Thor 用法：将帧周期、模型阶段、线程和 Engine 实例映射到时间线。

## 35. [P1][B] Nsight Systems — SQLite Export and Data Mining
- 链接：https://docs.nvidia.com/nsight-systems/UserGuide/
- 重点：将报告导出 SQLite，按 Kernel 名称、Stream、Context、时间戳、CPU Thread 聚合。
- Thor 用法：自动生成 OD/LD/ADAS/TRL 每帧耗时、Kernel 数量、排队间隙和 GPU 活动统计。

## 36. [P1][B] Nsight Systems — CPU Sampling and Backtrace
- 链接：https://docs.nvidia.com/nsight-systems/UserGuide/
- 重点：CPU 热点函数、线程阻塞、锁竞争、调用栈。
- Thor 用法：当 GPU Timeline 有空洞时，向上追踪提交 Kernel 的 CPU 线程。

## 37. [P1][B] Nsight Systems — CUDA Memory Allocation Trace
- 链接：https://developer.nvidia.com/nsight-systems/get-started
- 重点：识别频繁 `cudaMalloc/cudaFree`、内存分配同步和池化机会。
- Thor 用法：检查 Engine 重建、动态 Buffer、临时 Tensor 是否在热路径分配内存。

## 38. [P1][B] Nsight Systems — Multi-node Profiling
- 链接：https://developer.nvidia.com/nsight-systems
- 重点：多节点 CPU/GPU/NIC 网络活动关联。
- Thor 用法：车端单 SoC 直接相关性低；可迁移到云端回放、训练和数据处理系统。

## 39. [P1][B] Nsight Systems — Analyzing NCCL Usage
- 链接：https://developer.nvidia.com/nsight-systems/get-started
- 重点：分析 NCCL 通信和多 GPU 同步。
- Thor 用法：车端单 GPU 通常不是重点；对多 GPU 离线训练/仿真有价值。

## 40. [P1][B] Nsight Systems — Optimizing CUDA Memory Allocations
- 链接：https://developer.nvidia.com/nsight-systems/get-started
- 重点：内存分配 API、分配热点和内存池化。
- Thor 用法：适用于 TensorRT Context、Plugin Workspace 和预处理 Buffer 生命周期设计。

---

# 第三部分：Nsight Compute——SM、Warp、Occupancy、指令、算子与 Roofline

## 41. [P1][B] Nsight Compute Profiling Guide
- 链接：https://docs.nvidia.com/nsight-compute/ProfilingGuide/
- 重点：指标定义、采集模式、Replay、Source Correlation、Memory Workload、Warp State、Roofline。
- Thor 用法：所有 NCU 指标解释以当前目标版本文档为准。

## 42. [P1][B] Nsight Compute Compute Triage Guide
- 链接：https://docs.nvidia.com/nsight-compute/ComputeTriage/
- 重点：从最重要瓶颈开始；先 nsys 定位 Kernel，再 ncu；每次只改一个因素并重新 Profiling。
- Thor 用法：建立 Eko 算子优化闭环。

## 43. [P1][B] Nsight Compute Documentation
- 链接：https://docs.nvidia.com/nsight-compute/
- 重点：UI、CLI、Profiling Guide、Triage、Python Report Interface、Occupancy Calculator。
- Thor 用法：团队的 NCU 工具总入口。

## 44. [P1][B] Nsight Compute — Memory Workload Analysis
- 链接：https://developer.nvidia.com/nsight-compute/
- 重点：L1/L2/DRAM 流量、Transaction、Cache、访问指令与源代码关联。
- Thor 用法：分析 QDQ、Transpose、Gather、Scatter、Feature Copy 和 Plugin 访存。

## 45. [P1][B] Nsight Compute — Warp State Statistics
- 链接：https://docs.nvidia.com/nsight-compute/ProfilingGuide/
- 重点：`Long Scoreboard`、`LG Throttle`、`Short Scoreboard`、`Barrier`、`No Instruction` 等 Warp 状态。
- Thor 用法：只在 Scheduler 确实无法发射指令时追查 Stall；不能把所有 Stall 百分比简单相加当作可优化时间。

## 46. [P1][B] Nsight Compute — Speed of Light
- 链接：https://docs.nvidia.com/nsight-compute/ProfilingGuide/
- 重点：SM、DRAM、Tensor Core 等吞吐率相对硬件峰值。
- Thor 用法：判断 Kernel 的主要瓶颈属于计算还是内存资源。

## 47. [P1][B] Nsight Compute — Roofline Charts
- 链接：https://docs.nvidia.com/nsight-compute/ProfilingGuide/
- 重点：Arithmetic Intensity、FLOP/s、带宽上限、计算上限。
- Thor 用法：区分卷积/GEMM 与 Scatter/Gather 类型算子的优化方向。

## 48. [P1][B] Nsight Compute — Source Page / SASS Correlation
- 链接：https://developer.nvidia.com/nsight-compute/
- 重点：把指标热点映射到 CUDA C++、PTX、SASS 指令。
- Thor 用法：用于 Eko Plugin 的源代码级优化；需使用带 `-lineinfo` 的构建。

## 49. [P1][B] Nsight Compute CLI
- 链接：https://docs.nvidia.com/nsight-compute/NsightComputeCli/
- 重点：Kernel 过滤、NVTX Range 过滤、指标选择、报告导出、基线比较。
- Thor 用法：将常见 Plugin Profiling 命令封装成脚本。

## 50. [P1][B] Nsight Compute Profile Series
- 链接：https://developer.nvidia.com/blog/nsight-compute-2020-3-simplifies-cuda-kernel-profiling-and-optimization/
- 重点：批量扫描 Registers、Shared Memory 和配置组合。
- Thor 用法：用于自定义 Kernel 参数 Sweep；需控制变量并在真实 Engine 上复测。

## 51. [P1][B] Nsight Compute Application Replay
- 链接：https://docs.nvidia.com/nsight-compute/ProfilingGuide/
- 重点：通过重放应用或 Kernel 收集多个硬件计数器。
- Thor 用法：注意重放与并发调度可能不同，不能拿 Replay 结果代替真实多模型时延。

## 52. [P1][B] Nsight Compute Range Replay
- 链接：https://docs.nvidia.com/nsight-compute/ProfilingGuide/
- 重点：对 NVTX Range 内的一组 Kernel 进行 Profiling。
- Thor 用法：适合把一次模型阶段作为分析范围；必须检查 Range 的依赖和重放兼容性。

## 53. [P1][B] Nsight Compute Occupancy Calculator
- 链接：https://docs.nvidia.com/nsight-compute/
- 重点：线程数、寄存器、Shared Memory 对 Block Residency 的影响。
- Thor 用法：评估 Plugin 的 Block Size、寄存器占用和 Shared Memory 配置。

## 54. [P1][B] Nsight Compute Python Report Interface
- 链接：https://docs.nvidia.com/nsight-compute/
- 重点：自动读取 `.ncu-rep` 并抽取指标。
- Thor 用法：把 NCU 报告转成 Eko 算子性能数据库。

## 55. [P1][B] Nsight Compute — Tensor Core Metrics
- 链接：https://docs.nvidia.com/nsight-compute/ProfilingGuide/
- 重点：Tensor Core Pipe、MMA 指令、Tensor Throughput。
- Thor 用法：判断 GEMM/卷积是否真正利用 Tensor Core；不要期望整个 GPU 任务 100% Tensor Core 活跃。

## 56. [P1][B] Nsight Compute — Register Usage
- 链接：https://docs.nvidia.com/nsight-compute/ProfilingGuide/
- 重点：Registers per Thread、每 SM 活跃 Warp、Register Spill。
- Thor 用法：检查过多寄存器是否压低并发度。

## 57. [P1][B] Nsight Compute — Shared Memory
- 链接：https://docs.nvidia.com/nsight-compute/ProfilingGuide/
- 重点：Shared Memory 使用量、Bank Conflict、容量约束。
- Thor 用法：评估 Tile、Reduction、数据复用和 Occupancy 的平衡。

## 58. [P1][B] Nsight Compute — Global Memory Coalescing
- 链接：https://developer.nvidia.com/blog/unlock-gpu-performance-global-memory-access-in-cuda/
- 重点：相邻线程访问相邻地址、减少 Stride、提高 Transaction Efficiency。
- Thor 用法：Tensor Layout、图像 HWC/NCHW 转换、Plugin Load/Store。

## 59. [P1][B] Nsight Compute — Instruction Cache Metrics
- 链接：https://developer.nvidia.com/blog/improving-gpu-performance-by-reducing-instruction-cache-misses-2/
- 重点：`No Instruction` Stall、Instruction Cache Miss、Loop Unrolling。
- Thor 用法：复杂 Plugin、模板 Kernel 和大规模展开代码。

## 60. [P1][B] Nsight Compute — Local Memory and Register Spill
- 链接：https://docs.nvidia.com/nsight-compute/ProfilingGuide/
- 重点：Local Memory 实际位于设备内存层级；寄存器 Spill 可能增加访存。
- Thor 用法：配合编译器 `ptxas` 输出和 NCU 检查寄存器压力。

---

# 第四部分：访存优化、Cache、数据布局、CPU/GPU 迁移与数据传输

## 61. [P1][B] Unlock GPU Performance: Global Memory Access in CUDA
- 链接：https://developer.nvidia.com/blog/unlock-gpu-performance-global-memory-access-in-cuda/
- 重点：Coalesced Access、减少 Strided Access、通过 NCU 验证。
- Thor 用法：重点检查 Tensor Layout、Gather/Scatter、图像数据通道访问。

## 62. [P1][B] Improving GPU Memory Oversubscription Performance
- 链接：https://developer.nvidia.com/blog/improving-gpu-memory-oversubscription-performance/
- 重点：Unified Memory Page Migration、Zero-copy、Pinned Memory、Prefetch、访问模式。
- Thor 用法：车载 SoC 与 PCIe 独显平台不同，不能直接照搬带宽数字；方法可用于分析内存迁移和页错误。

## 63. [P1][B] Maximizing Unified Memory Performance in CUDA
- 链接：https://developer.nvidia.com/blog/maximizing-unified-memory-performance-cuda/
- 重点：理解 On-demand Page Migration、Prefetch、Zero-copy 与显式 Copy 的区别。
- Thor 用法：用于理解共享内存架构上的 CPU/GPU 数据可见性和迁移成本。

## 64. [P1][B] How to Optimize Data Transfers in CUDA C/C++
- 链接：https://developer.nvidia.com/blog/how-optimize-data-transfers-cuda-cc/
- 重点：Pinned Memory、批量传输、减少传输量、二维/三维 Copy。
- Thor 用法：多摄像头数据传输和多个模型输入 Buffer 的组织。

## 65. [P1][B] CUDA C++ Best Practices Guide
- 链接：https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/
- 重点：内存合并、Global/Shared/Constant Memory、Pinned Memory、异步 Copy、Occupancy。
- Thor 用法：作为 Eko CUDA 算子代码 Review 清单。

## 66. [P1][B] CUDA C++ Programming Guide — Memory Hierarchy
- 链接：https://docs.nvidia.com/cuda/cuda-c-programming-guide/
- 重点：Register、Local、Shared、Global、Constant、Texture、L1/L2。
- Thor 用法：建立 Thor GPU 的内存层级分析模型。

## 67. [P1][B] CUDA C++ Programming Guide — Asynchronous Execution
- 链接：https://docs.nvidia.com/cuda/cuda-c-programming-guide/
- 重点：Host/Device 异步执行、Stream、Event、依赖关系。
- Thor 用法：分析多 Engine 调度、异步推理与同步点。

## 68. [P1][B] CUDA C++ Programming Guide — Concurrent Execution
- 链接：https://docs.nvidia.com/cuda/cuda-c-programming-guide/
- 重点：并发 Kernel、Copy/Compute Overlap、硬件约束。
- Thor 用法：理解不同 Stream 的 Kernel 为什么不一定同时执行。

## 69. [P1][B] CUDA C++ Programming Guide — Unified Memory
- 链接：https://docs.nvidia.com/cuda/cuda-c-programming-guide/
- 重点：Unified Memory 编程模型、Prefetch、Advice、访问行为。
- Thor 用法：理解 SoC 内存一致性与数据迁移成本。

## 70. [P1][B] CUDA C++ Programming Guide — Cooperative Groups
- 链接：https://docs.nvidia.com/cuda/cuda-c-programming-guide/
- 重点：线程组、Warp/Block 协作、同步和 Reduction。
- Thor 用法：自定义 Plugin 的线程协作设计。

## 71. [P1][B] CUDA C++ Best Practices — Coalesced Global Memory Access
- 链接：https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/
- 重点：减少每个 Warp 访问的 Memory Transaction 数。
- Thor 用法：HWC/NCHW、HWC8/HWC16、Vectorized Load/Store。

## 72. [P1][B] CUDA C++ Best Practices — Shared Memory
- 链接：https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/
- 重点：利用 Shared Memory 复用数据，减少重复 Global Memory 访问。
- Thor 用法：卷积、Tile、局部数据复用；注意 Shared Memory 过大可能降低 Occupancy。

## 73. [P1][B] CUDA C++ Best Practices — Occupancy
- 链接：https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/
- 重点：寄存器、Shared Memory、Block Size 对并发驻留的影响。
- Thor 用法：不要盲目追求 100% Occupancy，应结合吞吐率和实际耗时。

## 74. [P1][B] CUDA C++ Best Practices — Pinned Memory
- 链接：https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/
- 重点：Page-locked Host Memory 与异步传输。
- Thor 用法：需要先确认 DRIVE SoC Buffer / NvSciBuf 互操作路径，不能机械套用 PCIe 离散 GPU 的分配方式。

## 75. [P1][B] CUDA C++ Best Practices — Data Transfer Batching
- 链接：https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/
- 重点：合并大量小 Copy，减少 API 和传输固定开销。
- Thor 用法：多相机小 Buffer、多个模型输入输出的传输组织。

---

# 第五部分：Kernel Fusion、Launch Overhead、CUDA Graph、算法并行度

## 76. [P1][A] Kernel Fusion in NVIDIA CUDA: Optimizing Memory Traffic and Launch Overhead
- 链接：https://developer.nvidia.com/blog/kernel-fusion-in-nvidia-cuda-optimizing-memory-traffic-and-launch-overhead/
- 领域：Kernel Fusion、全局内存流量、Launch Overhead。
- 问题与发现：两个 Kernel 中间需要写入/读取临时 Buffer，产生额外 Global Memory Traffic 和 Launch。
- 优化：把多个操作融合到一个 Kernel，让中间结果留在寄存器；CUDA Graph 则从 Host Launch 层减少提交开销。
- 公开结果：示例 2 Kernel → 1 Kernel，数据流量 3GB → 1GB，3.51ms → 1.18ms，约 3 倍。
- Thor/Eko 参考：极高。对 QDQ、Cast、Transpose、Elementwise、激活函数融合非常有价值；需防止融合后寄存器压力上升。

## 77. [P1][B] CUDA Graphs — Programming Guide
- 链接：https://docs.nvidia.com/cuda/cuda-c-programming-guide/
- 重点：Graph Capture、Instantiate、Launch、节点依赖。
- Thor 用法：用于减少短 Kernel 序列的 Host Launch Overhead。

## 78. [P1][B] CUDA Graphs — Best Practices
- 链接：https://developer.nvidia.com/blog/cuda-graphs/
- 重点：捕获重复工作流、减少 CPU Launch 成本。
- Thor 用法：针对静态 Shape 的重复推理图进行试验；动态 Shape 和 Buffer 地址变化要单独验证。

## 79. [P1][B] CUDA Graphs and TensorRT
- 链接：https://docs.nvidia.com/deeplearning/tensorrt/
- 重点：TensorRT 执行与 CUDA Graph 捕获、重复执行和动态输入约束。
- Thor 用法：在目标 DRIVE TensorRT 版本上验证支持范围，不要只依据桌面版文档。

## 80. [P1][B] CUDA Kernel Launch Overhead
- 链接：https://developer.nvidia.com/blog/cuda-graphs/
- 重点：短 Kernel 的 Launch 开销可能与 Kernel 本身执行时间同量级。
- Thor 用法：当模型由大量微小 Kernel 组成时，评估 Fusion / CUDA Graph / Batch。

## 81. [P1][B] CUDA Streams and Asynchronous Copies in the Modern CUDA Toolbox
- 链接：https://developer.nvidia.com/blog/the-modern-cuda-toolbox-in-practice-a-step-by-step-optimization-walkthrough/
- 重点：每个并行工作线程独立 Stream，使用异步 Copy，让 H2D、Kernel、D2H 重叠。
- Thor 用法：多摄像头预处理与多个模型输入管线。

## 82. [P1][B] CUB Device Algorithms
- 链接：https://nvidia.github.io/cccl/
- 重点：Scan、Reduce、Sort、Select 等经过优化的通用 GPU 原语。
- Thor 用法：Plugin 中优先评估成熟 Primitive，避免重复造轮子。

## 83. [P1][B] Cooperative Groups
- 链接：https://docs.nvidia.com/cuda/cuda-c-programming-guide/
- 重点：跨 Warp / Block 的并行组织和同步。
- Thor 用法：Reduction、Prefix Sum、分块数据处理。

## 84. [P1][B] CUDA Tile Programming
- 链接：https://developer.nvidia.com/blog/
- 重点：Tile 级编程、数据布局和并行计算抽象。
- Thor 用法：关注目标 SM 架构支持与编译器成熟度；不能默认 Thor 上所有新 CUDA 特性都可用。

## 85. [P1][B] CUTLASS
- 链接：https://github.com/NVIDIA/cutlass
- 重点：GEMM、Convolution、Tensor Core Tile、数据搬运和 Pipeline。
- Thor 用法：研究 Tensor Core 算子、Tile Shape、Warp Specialization；必须核对 SM101 / DRIVE 编译支持。

---

# 第六部分：TensorRT、模型部署、Plugin 与推理性能

## 86. [P0][B] TensorRT Documentation
- 链接：https://docs.nvidia.com/deeplearning/tensorrt/
- 重点：Engine、Builder、Tactic、Precision、Plugin、Runtime、动态 Shape。
- Thor 用法：对照 DRIVE OS 对应 TensorRT 版本，不直接用最新桌面版功能推断车端支持。

## 87. [P0][B] TensorRT Best Practices
- 链接：https://docs.nvidia.com/deeplearning/tensorrt/developer-guide/
- 重点：Benchmark、Profiling、Builder、Runtime、并发、CUDA Graph、内存管理。
- Thor 用法：Engine 构建和运行时性能优化的主要依据。

## 88. [P0][B] TensorRT trtexec
- 链接：https://docs.nvidia.com/deeplearning/tensorrt/command-line-programs.html
- 重点：模型独立基线、Latency、Throughput、Layer Profiling、Engine 构建参数。
- Thor 用法：先单模型建立稳定基线，再与集成系统的 nsys 结果比较。

## 89. [P0][B] TensorRT IPluginV3
- 链接：https://docs.nvidia.com/deeplearning/tensorrt/
- 重点：自定义算子集成、Format/Datatype、Workspace、序列化、运行时。
- Thor 用法：Eko Plugin 的实现与性能边界。

## 90. [P0][B] TensorRT Plugin Samples
- 链接：https://github.com/NVIDIA/TensorRT
- 重点：官方 Plugin 示例、ONNX Parser、Plugin 注册和构建。
- Thor 用法：对照官方实现检查 Plugin 的格式支持、Workspace、Kernel Launch 和精度路径。

## 91. [P0][B] TensorRT Performance Best Practices — Benchmarking
- 链接：https://docs.nvidia.com/deeplearning/tensorrt/
- 重点：预热、重复迭代、GPU 时钟、功耗、统计口径。
- Thor 用法：固定温度、功耗模式、时钟和输入 Shape，避免把频率波动当成优化收益。

## 92. [P0][B] TensorRT Layer Profiling
- 链接：https://docs.nvidia.com/deeplearning/tensorrt/
- 重点：逐 Layer 耗时和 Engine 内部瓶颈。
- Thor 用法：结合 Nsight Systems / Nsight Compute 定位具体 Layer 对应 Kernel。

## 93. [P0][B] TensorRT Quantization and Precision
- 链接：https://docs.nvidia.com/deeplearning/tensorrt/
- 重点：INT8、FP16、FP8、Q/DQ、精度约束与融合。
- Thor 用法：分析 Backbone INT8 QAT 后的 QDQ/Cast 开销，以及 Head 量化是否能减少资源竞争。

## 94. [P0][B] TensorRT Dynamic Shapes
- 链接：https://docs.nvidia.com/deeplearning/tensorrt/
- 重点：Optimization Profile、动态输入尺寸、Shape Tensor。
- Thor 用法：动态 Shape 可能影响 Tactic、Graph Capture 和内存复用。

## 95. [P0][B] TensorRT CUDA Graph Integration
- 链接：https://docs.nvidia.com/deeplearning/tensorrt/
- 重点：重复推理的 Graph Capture 与执行约束。
- Thor 用法：针对固定输入形状和固定 Buffer 的 Engine 进行对比试验。

---

# 第七部分：DRIVE、车载异构计算与视觉前后处理

## 96. [P0][B] NVIDIA DRIVE Developer Documentation
- 链接：https://developer.nvidia.com/drive/documentation
- 重点：DRIVE OS、TensorRT、PVA、NvMedia、NvSci、CUDA 等车端 SDK。
- Thor 用法：必须以实际 DRIVE OS 版本（如 DRIVE OS 7.0.x / 7.2）对应文档为准。

## 97. [P0][B] NVIDIA DRIVE AGX Thor Platform
- 链接：https://developer.nvidia.com/drive
- 重点：车载计算平台、异构加速器、AI 推理能力和软件栈入口。
- Thor 用法：用于确认平台能力和官方软件资料入口；具体硬件资源和版本以对应产品/开发文档为准。

## 98. [P0][B] NvSciBuf / NvSciSync / NvStreams
- 链接：https://developer.nvidia.com/drive/documentation
- 重点：跨硬件模块 Buffer 共享、同步和异步任务调度。
- Thor 用法：分析 CPU、GPU、PVA、VIC、DLA 之间的数据共享与同步，避免不必要的 CPU Round Trip。

## 99. [P1][B] DeepStream Performance Documentation
- 链接：https://docs.nvidia.com/metropolis/deepstream/dev-guide/
- 重点：视频解码、Batch、Pipeline、零拷贝、推理和后处理。
- Thor 用法：部分概念可用于多摄像头视觉 Pipeline；但 DeepStream 服务器/Jetson 部署与 DRIVE Thor 的集成方式不完全相同。

## 100. [P1][B] NVIDIA Data Loading Library (DALI)
- 链接：https://docs.nvidia.com/deeplearning/dali/user-guide/docs/
- 重点：图像解码、预处理、CPU/GPU Pipeline、Batch 和数据搬运。
- Thor 用法：适合作为图像预处理 Pipeline 的参考实现；是否可直接部署到 DRIVE OS 需验证平台兼容性。

---

# 第八部分：扩展阅读——建议继续补充到团队知识库的官方入口

以下入口本身可能包含多个教程、章节、示例或版本文档。它们不是额外的独立“优化案例”，但适合持续跟踪：

1. Nsight Systems 官方入口：https://developer.nvidia.com/nsight-systems
2. Nsight Compute 官方入口：https://developer.nvidia.com/nsight-compute
3. Nsight Compute Blogs：https://developer.nvidia.com/nsight-compute-blogs
4. Nsight Systems Get Started：https://developer.nvidia.com/nsight-systems/get-started
5. Nsight Compute Profiling Guide：https://docs.nvidia.com/nsight-compute/ProfilingGuide/
6. Nsight Compute Triage Guide：https://docs.nvidia.com/nsight-compute/ComputeTriage/
7. Nsight Systems User Guide：https://docs.nvidia.com/nsight-systems/UserGuide/
8. CUDA C++ Programming Guide：https://docs.nvidia.com/cuda/cuda-c-programming-guide/
9. CUDA C++ Best Practices Guide：https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/
10. CUDA Samples：https://github.com/NVIDIA/cuda-samples
11. TensorRT：https://docs.nvidia.com/deeplearning/tensorrt/
12. CUTLASS：https://github.com/NVIDIA/cutlass
13. CCCL：https://github.com/NVIDIA/cccl
14. DRIVE Developer：https://developer.nvidia.com/drive
15. NVIDIA Technical Blog：https://developer.nvidia.com/blog/

---

# 第九部分：按性能根因分类的优化方法论

本节不是简单按工具分类，而是按“性能问题 → Nsight 证据 → 根因 → 优化手段”组织。建议用它作为 Thor 多模型性能分析的主索引。

## A. GPU 调度、Kernel 排队与多模型资源竞争

**典型现象**
- OD / LD / ADAS / TRL 的模型总耗时接近单模型耗时之和。
- 多个 Stream 存在，但 GPU 时间线上 Kernel 仍然串行。
- 某个模型的 CUDA API 已返回或 Enqueue 已完成，但 GPU Kernel 尚未开始。
- GPU 长时间 Busy，但每个模型的端到端时延都变长。

**优先工具**
- Nsight Systems：CUDA HW Timeline、Streams、CUDA API、GPU Metrics、NVTX。
- Nsight Compute：只对代表性 Kernel 做进一步资源分析；不要把 NCU Replay 当作并发时延基线。

**重点指标/证据**
- Kernel Start / End 时间。
- Stream ID、Context、CUDA API 时间。
- SM Active、Tensor Active、DRAM Throughput。
- 各模型 NVTX Range 的重叠关系。
- Kernel 是否实际重叠，还是仅仅 Host 侧并发提交。

**优化候选**
1. 检查 Stream 是否真正独立，是否被默认 Stream 或 Event 依赖串行化。
2. 检查不同 Engine 是否共享同一个执行 Stream / Context / Buffer。
3. 调整模型调度顺序，减少大 Kernel 对关键实时任务的阻塞。
4. 对低负载、小 Kernel 阶段尝试并发；对已饱和的 SM-bound Kernel，不要盲目增加并发。
5. 评估模型分时、优先级、分批执行或异构硬件卸载。
6. 对重复固定工作流评估 CUDA Graph，减少 Host 提交开销。

**重要判断**
- 多 Stream ≠ 多 Kernel 一定并行。
- GPU Busy ≠ SM 计算效率高。
- 多模型时延上升可能来自 SM、寄存器、Shared Memory、L2、DRAM、调度和同步的任一资源竞争。
- Thor 是否支持某种硬件分区能力，必须以 DRIVE OS / GPU 架构支持为准，不能直接把 MIG 经验套用到 Thor。

## B. CPU 迁移 GPU / CPU-GPU 协同优化

**典型现象**
- GPU Timeline 有空洞，CPU 线程在执行图像处理或后处理。
- CPU API 提交不及时，GPU 等待工作。
- CPU 预处理和 GPU 推理无法重叠。
- Unified Memory Page Fault / Migration 活动明显。

**优先工具**
- Nsight Systems：CPU Thread、CPU Sampling、CUDA API、Memory Transfer、NVTX。
- 必要时结合 Nsight Compute 评估迁移后的 Kernel。

**优化候选**
1. 将计算量大、并行度高、数据已在 GPU 附近的阶段迁移到 GPU。
2. 对低计算量但高频调用的操作，评估是否应与相邻 Kernel 融合，而不是单独迁移。
3. 避免 CPU↔GPU 来回搬运中间结果。
4. 对单次使用的小数据评估 Zero-copy；对高复用数据评估设备侧驻留。
5. 在 DRIVE 上同时评估 PVA、VIC、DLA 等异构引擎，不应默认所有 CV 工作都迁到 GPU。
6. 对跨硬件引擎数据交换研究 NvSciBuf / NvSciSync / NvStreams。

## C. 访存优化：Global Memory / L1 / L2 / DRAM

**典型现象**
- SM Compute Throughput 不高，但 Kernel 很慢。
- Long Scoreboard Stall 明显。
- Memory Workload 显示 DRAM/L2 流量高。
- Global Load/Store Transaction 数异常。
- Gather/Scatter、Transpose、Layout Conversion 的时间突出。

**优先工具**
- Nsight Compute：Memory Workload Analysis、Speed of Light、Warp State、Source Page。
- Nsight Systems：确认访存热点在整体业务中占比。

**优化候选**
1. 保证 Warp 内访问尽可能 Coalesced。
2. 减少 Strided / Random Access。
3. 通过 Tile、数据布局和循环顺序提高数据复用。
4. 对重复索引做预计算或压缩索引表示。
5. 减少重复读取和不必要的中间 Tensor。
6. 让高复用数据留在 Register / Shared Memory / Cache。
7. 根据工作集与 L2 容量选择不同实现。
8. 对输出流式写入评估 Cache Policy，避免大输出驱逐热点索引。
9. 用实际指标判断 FP16/FP8 是否真的降低访存成本，不只看数据类型位宽。

**与感知模型的对应**
- BEV Pooling：Scatter / Gather / Atomic / Index。
- NMS：排序、筛选、Reduction。
- Lane / Occupancy：不规则采样和后处理。
- Backbone：卷积数据复用、Tensor Core。
- 多相机拼接：Layout、Memcpy、数据转换。

## D. Kernel Fusion 与减少 Launch Overhead

**典型现象**
- 很多短 Kernel 连续执行。
- 每个 Kernel 本身很短，但累计时间很高。
- 中间 Tensor 被反复写入和读取。
- CPU CUDA API 调用密集。

**优先工具**
- Nsight Systems：统计 Kernel 数量、持续时间、空隙和 API 调用。
- Nsight Compute：融合后检查寄存器、Occupancy 和访存。

**优化候选**
1. 融合相邻 Elementwise / Activation / Cast / QDQ。
2. 将 Layout 转换与相邻计算融合。
3. 减少中间 Tensor 写回 Global Memory。
4. 对重复执行的短 Kernel 序列评估 CUDA Graph。
5. 对多个独立小输入评估 Batch 化，减少大量小 Kernel Launch。
6. 融合后必须检查寄存器压力、指令缓存和 Occupancy，防止过度融合。

## E. Warp Stall / 指令吞吐 / 指令缓存

**典型现象**
- Kernel Duration 高，但 DRAM Throughput 不高。
- `No Instruction`、`Long Scoreboard`、`LG Throttle`、`Short Scoreboard` 等 Stall 较突出。
- 大量 Loop Unrolling 或复杂分支。
- Kernel 的指令规模异常大。

**优先工具**
- Nsight Compute：Warp State Statistics、Scheduler Statistics、Instruction Stats、Source/SASS。
- Nsight Systems：判断该 Kernel 是否为真实业务热点。

**优化候选**
1. `Long Scoreboard`：检查 Global/Local Memory 依赖、访问模式和数据复用。
2. `LG Throttle`：检查 LSU 指令队列压力，考虑合并低宽度访存、交错计算与访存。
3. `No Instruction`：检查 Instruction Cache、代码体积、循环展开和分支路径。
4. `Barrier`：检查 Block 内同步是否过多。
5. `Short Scoreboard`：检查短延迟依赖和指令调度。
6. 不要仅凭 Stall 占比决定优化方向，先确认 Scheduler 是否真的无法发射指令。

## F. Occupancy、寄存器与 Shared Memory

**典型现象**
- Occupancy 低。
- Registers per Thread 很高。
- Register Spill 增加。
- Shared Memory 使用过大。
- Block Size 变化导致性能大幅变化。

**优先工具**
- Nsight Compute：Launch Statistics、Occupancy、Register、Shared Memory、Local Memory。
- 编译器 `ptxas` 输出和 Occupancy Calculator。

**优化候选**
1. 按输入类型/数据路径拆分 Kernel，避免通用 Kernel 使用过多资源。
2. 选择合适 Block Size / Warp 数。
3. 控制寄存器生命周期和临时变量。
4. 避免过度 Loop Unrolling。
5. 利用 Shared Memory 复用数据，但不能为了提高复用而牺牲过多 Occupancy。
6. 评估目标 CUDA 版本是否支持 Shared Memory Register Spilling。
7. 以 Kernel 时延和整体吞吐为最终指标，不以 Occupancy 百分比为目标本身。

## G. Tensor Core / GEMM / 精度优化

**典型现象**
- GEMM/Conv 占比高。
- Tensor Core 活跃程度低。
- FP16/INT8/FP8 切换没有获得预期收益。
- QDQ/Cast/Format Conversion Kernel 占用明显。

**优先工具**
- Nsight Systems：整体 Layer/Kernel 时间占比。
- Nsight Compute：Tensor Core Pipe、SM Throughput、Memory、Instruction。
- TensorRT Layer Profiling：逐层确认精度和算子执行路径。

**优化候选**
1. 检查输入 Shape、Channel、对齐和数据格式是否匹配高效 Tactic。
2. 检查 Tensor Core 是否实际执行，而不是仅仅设置 FP16/INT8。
3. 分析 QDQ/Cast 是否引入额外 Kernel 或格式转换。
4. 评估 Head QAT INT8 是否减少整体 GPU 资源占用，而不仅是单模型 latency。
5. 比较 FP16/INT8/FP8 的端到端精度、吞吐和时延。
6. 关注小矩阵和小 Batch 下的 Launch/转换开销，低精度不一定更快。

## H. 视觉 Pipeline：解码、预处理、后处理和异构引擎

**典型现象**
- 推理 Engine 很快，但帧周期仍然超时。
- Decode、Resize、Color Conversion、Crop、Normalize、NMS 占据大量 CPU/GPU 时间。
- 多相机 Pipeline 存在同步和数据复制。

**优先工具**
- Nsight Systems：NVTX 阶段标记、CPU/GPU 时间线、Copy、同步。
- Nsight Compute：自定义图像 Kernel。
- DRIVE 专用工具：PVA / VIC / DLA 的 Runtime Profiling（以对应 SDK 为准）。

**优化候选**
1. 预处理算子融合。
2. 选择性 ROI / 分辨率处理。
3. Batch 或 Mini-batch，前提是不违反实时延迟预算。
4. 将适合的传统 CV 操作卸载至 PVA/VIC。
5. 减少跨模块 Copy，使用可共享 Buffer。
6. 用硬件同步机制减少 CPU 参与。
7. 分别统计每路摄像头的输入等待、预处理、推理和后处理时间。

---

# 第十部分：面向 DRIVE AGX Thor / Eko 的落地 Profiling 方案

## 1. 建议建立统一的 NVTX 命名规范

推荐把一次完整帧处理拆成如下 Range：

```text
Frame[frame_id]
 ├─ CameraInput
 │   ├─ FrontLong
 │   ├─ FrontWide
 │   └─ ...
 ├─ Preprocess
 ├─ Backbone
 ├─ Head_OD
 ├─ Head_LD
 ├─ Head_ADAS
 ├─ Model_TRL
 ├─ Decision_DLP
 └─ Postprocess
```

每个 Range 应能关联：
- Frame ID / Camera ID
- Engine 名称
- CUDA Stream ID
- Tensor Shape / Data Type
- 模型开始和结束时间
- CPU Enqueue 时间和 GPU Kernel 时间

## 2. 建议分三轮 Profiling

### Round A：系统级基线（Nsight Systems）

目标：回答“整体慢在哪里”。

采集：
- CUDA API
- CUDA GPU Kernel
- Memory Copy
- NVTX
- CPU Sampling
- GPU Metrics（目标设备支持时）

输出：
- 每个模型的端到端时间
- 每个模型 GPU Kernel 累计时间
- Kernel 排队等待和空洞
- Stream 间真实重叠情况
- CPU 线程阻塞/同步情况
- GPU SM / DRAM / Tensor 活动变化

### Round B：算子级分析（Nsight Compute）

目标：回答“热点 Kernel 为什么慢”。

每次只挑少量代表性 Kernel：
- Backbone Conv/GEMM
- QDQ/Cast
- Transpose/Layout
- BEV Pooling / Gather / Scatter
- NMS / Reduction
- Eko 自定义 Plugin

优先分析：
- Speed of Light
- Memory Workload
- Warp State
- Occupancy
- Register / Shared Memory
- Source / SASS

注意：NCU 计数器采集可能改变并发调度。对多模型真实时延的结论仍以 nsys 和端到端测试为准。

### Round C：优化回归

每次只改变一个主要因素：
1. 建立原始基线。
2. 记录 Engine / Plugin / CUDA / TRT 版本。
3. 修改代码或配置。
4. 重跑相同输入、相同帧数和相同系统负载。
5. 对比 P50/P90/P99、最大时延、帧周期、GPU 利用率和精度。
6. 复查是否把瓶颈转移到其他模型或 Pipeline 阶段。

## 3. 建议优先验证的 12 个 Thor 优化假设

| 序号 | 假设 | 主要证据 | 可能措施 |
|---|---|---|---|
| 1 | 多个 Head 的 Kernel 其实没有重叠 | nsys Stream Timeline | 调整 Stream / 依赖 / 调度 |
| 2 | GPU 已经 SM 饱和 | GPU Metrics + NCU | 降低并发、错峰或异构卸载 |
| 3 | TRL 排队等待 OD/LD/ADAS | NVTX + Kernel Start | 关键任务优先级/调度 |
| 4 | QDQ/Cast 产生额外 Kernel | Layer Profile + nsys | 量化图优化、格式融合 |
| 5 | Transpose/Layout 转换过多 | Kernel 名称、Memory Workload | 统一 Tensor Layout |
| 6 | BEV/Gather/Scatter 访存低效 | NCU Transaction / L2 / DRAM | 索引预计算、数据重排 |
| 7 | 自定义 Plugin Register Spill | NCU Local Memory / Registers | 缩短生命周期、调 Tile |
| 8 | 小 Kernel Launch 太多 | nsys Kernel Timeline | Fusion / CUDA Graph |
| 9 | CPU 预处理造成 GPU 空洞 | CPU Thread + NVTX | CPU 并行化或迁移到加速器 |
| 10 | Host/Device Copy 阻塞 | nsys Memcpy / Stream | 异步、零拷贝、Buffer 复用 |
| 11 | Kernel 长尾来自输入形状变化 | 按 Shape 分组 Profiling | 多 Shape Tactic / 专用 Kernel |
| 12 | 单模型优化损害多模型总时延 | 多模型联合压测 | 以全系统 P99 和帧周期优化 |

## 4. 结果记录模板

建议每次优化至少保留：

| 字段 | 记录内容 |
|---|---|
| 平台 | DRIVE AGX Thor / 其他 |
| DRIVE OS | 完整版本 |
| CUDA / TensorRT | 完整版本 |
| 模型/Engine | OD / LD / ADAS / TRL / DLP |
| 输入 Shape / Batch | 完整记录 |
| 精度 | FP32 / FP16 / INT8 / FP8 |
| Stream | Stream ID 和依赖 |
| 优化前 | P50 / P90 / P99 / Max |
| 优化后 | P50 / P90 / P99 / Max |
| Nsys 证据 | Timeline / API / GPU Metrics |
| NCU 证据 | SOL / Memory / Warp / Occupancy |
| 精度回归 | 指标与阈值 |
| 结论 | 保留 / 回退 / 继续验证 |

---

# 第十一部分：推荐阅读顺序

如果只安排 2～3 天进行第一轮学习，建议按以下顺序：

1. **BEV Pooling**：理解真实自动驾驶算子如何通过 NCU 优化 Scatter/Gather。
2. **NuRec 多传感器案例**：学习 nsys + ncu 的系统级到 Kernel 级闭环。
3. **VC-6 Batch Pipeline**：学习多路视觉、Batch、CPU/GPU Pipeline。
4. **GROMACS Memory Transfer**：理解 CPU 迁移 GPU、Stream Overlap 和同步优化。
5. **Multi-stream Occupancy**：建立多 Stream 的 GPU 活跃度分析方法。
6. **Nsight Compute Part 1～3**：系统学习 Occupancy、Memory Transaction、Warp Stall 和 Roofline。
7. **Modern CUDA Toolbox**：学习 NVTX、Pinned Memory、内存池和异步 Stream 的组合优化。
8. **TensorRT Best Practices + Nsight Systems**：把模型独立基线和系统集成时延分开分析。
9. **DRIVE PVA CV Pipeline**：评估 Thor 异构计算卸载的可行方向。
10. **CUDA Graph + Kernel Fusion**：在确认 Launch Overhead 是根因后再实施。

---

# 第十二部分：最终结论

对于 DRIVE AGX Thor 上的多模型感知部署，Nsight 的核心价值不是“把 GPU 利用率打出来”，而是建立完整证据链：

**业务阶段 → CPU Thread → CUDA API → Stream → Kernel → SM / Warp / Memory → 优化措施 → 端到端回归。**

优先级建议：

1. 先用 Nsight Systems 判断多模型是否真正并发、是否有排队、同步和 CPU 空洞。
2. 再用 Nsight Compute 判断热点 Kernel 的瓶颈是计算、访存、指令、寄存器还是 Occupancy。
3. 对 TensorRT Engine 使用 `trtexec` 和 Layer Profiling 建立独立基线。
4. 对 BEV / Gather / Scatter / Layout / QDQ 等算子建立专项优化。
5. 对车载 Pipeline 同时评估 GPU、PVA、VIC、DLA 的异构资源分配。
6. 最终以 10Hz 帧周期、P99 时延、精度回归和系统稳定性作为验收标准，而不是单个 Kernel 的峰值指标。

> 本集合约 100 项资料，其中前 20 项优先级最高；前半部分以直接案例为主，后半部分包含官方工具指南、CUDA/TensorRT/DRIVE 技术参考。对于没有公开 Nsight 证据的辅助资料，已明确标记为方法或技术参考，不应当作为“Nsight 已验证的优化案例”引用。
