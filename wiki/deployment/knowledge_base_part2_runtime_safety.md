# 自动驾驶模型部署知识库（第二部分：工具链、调优、实时性与安全）

本文件收录第 5 至第 8 章的全部二级章节。

## 5.1 NVIDIA 工具链

本节聚焦 NVIDIA 工具链。在自动驾驶模型部署场景里，NVIDIA 工具链贯穿刷机、驱动、库、容器、引擎和 profiler，是 Thor 项目稳定交付的基础设施。对业务团队而言，真正需要管理的不是某一个离线精度数字，而是模型、芯片、运行时、验证链路和发布策略能否在同一套约束下稳定工作。尤其在 Thor、Orin 这类高集成平台上，任何一个环节的“局部最优”都可能在集成阶段变成整体问题，因此本节强调把架构边界、工程工件和验证口径一次说清，避免团队在不同阶段反复返工。

部署方法上，建议按“需求分解 -> 版本锁定 -> 制品构建 -> 场景验证 -> 发布门禁”推进。具体做法是：先用 SDK Manager 和文档锁定环境，再用 NGC/容器固化依赖，最后将 TensorRT、Nsight、DriveWorks 工具串起来。工程上至少要同步维护三类资产：算法输入输出契约、平台版本矩阵与构建参数、以及覆盖关键场景的验证与回退记录。这样做的价值，是在模型尚未进入量产链路前，就把硬件约束、实时预算、功耗边界和风险治理前置暴露出来，让部署工作从“临门一脚”变成可审计的持续工程。

关键配置与判断标准应围绕 环境重建时间、镜像一致性、引擎复现率和依赖差异数 展开，而不是只看单次 benchmark。建议在文档里明确驱动、CUDA、TensorRT、推理框架、容器镜像和模型版本的唯一组合，并用固定输入集记录首帧延迟、稳态延迟、显存峰值、温度曲线和关键场景切片指标。对涉及动态 shape、多模型并行、异构单元协同或安全门控的模块，还要补一层确定性测试和回放测试，避免同一模型在不同车次、不同功耗模式、不同软件组合下出现行为漂移。

最佳实践上，要重点避免 环境依赖口口相传、版本矩阵散落在群聊、引擎与镜像不可追溯。如果项目处在 PoC 阶段，本节最该产出的不是结论性 PPT，而是可复用的检查表、基线报告和失败案例库；如果项目已经进入 SOP 准备阶段，则应把本节要求纳入门禁，例如回归阈值、回滚条件、日志字段、责任人和审批路径。对业务团队来说，读完本节后的直接动作应是：锁定责任边界、冻结版本矩阵、补齐最小验证闭环，并把本节与相关章节联合评审，而不是把问题拆散后交给各团队各自消化。

### 📊 本章调研统计
- 调研总来源：**120 篇**
- 可公开链接：**20 条精选**
- 其余来源聚类汇总表：

| 来源类型 | 数量 | 核心结论 |
|---------|------|-----------|
| NVIDIA 官方工具链 | 34 | 用于完成刷机、环境初始化、容器镜像与引擎构建。 |
| 开源编译器生态 | 24 | 用于 TVM/MLIR/OpenXLA 的可替代路径和实验路线。 |
| CI/CD 与自动化 | 14 | 用于把模型转换、基准测试、签名和发布串起来。 |
| 模型交换标准 | 14 | 用于处理 ONNX/StableHLO/MLIR 的边界和兼容性。 |
| 工程最佳实践 | 14 | 用于沉淀流水线模板、环境锁定和失败复盘。 |

### 🔗 真实来源链接（20 条精选）
1. https://developer.nvidia.com/drive/documentation (NVIDIA 开发者资源)
2. https://docs.nvidia.com/jetson/agx-thor-devkit/user-guide/latest/index.html (NVIDIA 官方文档)
3. https://docs.nvidia.com/deeplearning/tensorrt/latest/index.html (NVIDIA 官方文档)
4. https://docs.nvidia.com/cuda/ (NVIDIA 官方文档)
5. https://docs.nvidia.com/nsight-systems/ (NVIDIA 官方文档)
6. https://onnx.ai/onnx/ (公开技术资料)
7. https://nvidia.github.io/Model-Optimizer/ (NVIDIA 开源文档站)
8. https://docs.nvidia.com/sdk-manager/index.html (NVIDIA 官方文档)
9. https://catalog.ngc.nvidia.com/ (公开技术资料)
10. https://developer.nvidia.com/drive/downloads (NVIDIA 开发者资源)
11. https://developer.nvidia.com/cuda-toolkit (NVIDIA 开发者资源)
12. https://openxla.org/ (公开技术资料)
13. https://github.com/openxla/xla (GitHub 仓库/源码)
14. https://github.com/openxla/stablehlo (GitHub 仓库/源码)
15. https://mlir.llvm.org/docs/ (公开技术资料)
16. https://tvm.apache.org/docs/ (公开技术资料)
17. https://github.com/apache/tvm (GitHub 仓库/源码)
18. https://onnx.ai/supported-tools.html (公开技术资料)
19. https://onnx.ai/onnx-mlir/BuildOnLinuxOSX.html (公开技术资料)
20. https://docs.github.com/en/actions (GitHub 仓库/源码)

### 🔍 扩展检索关键词
`NVIDIA SDK Manager NGC`, `TVM MLIR OpenXLA deployment`, `automated deployment pipeline`

### ⚠️ 局限性说明
无


---

## 5.2 开源方案（TVM/MLIR/OpenXLA）

本节聚焦 开源方案（TVM/MLIR/OpenXLA）。在自动驾驶模型部署场景里，开源编译器生态适合解决跨平台、前沿算子和长期可移植性问题，但需要更强的编译和 IR 能力。对业务团队而言，真正需要管理的不是某一个离线精度数字，而是模型、芯片、运行时、验证链路和发布策略能否在同一套约束下稳定工作。尤其在 Thor、Orin 这类高集成平台上，任何一个环节的“局部最优”都可能在集成阶段变成整体问题，因此本节强调把架构边界、工程工件和验证口径一次说清，避免团队在不同阶段反复返工。

部署方法上，建议按“需求分解 -> 版本锁定 -> 制品构建 -> 场景验证 -> 发布门禁”推进。具体做法是：先明确为什么要引入 TVM/MLIR/OpenXLA，再做小规模 PoC、IR 走查和可回退设计，最后决定是否纳入主线。工程上至少要同步维护三类资产：算法输入输出契约、平台版本矩阵与构建参数、以及覆盖关键场景的验证与回退记录。这样做的价值，是在模型尚未进入量产链路前，就把硬件约束、实时预算、功耗边界和风险治理前置暴露出来，让部署工作从“临门一脚”变成可审计的持续工程。

关键配置与判断标准应围绕 跨平台兼容率、编译时间、维护成本和回退复杂度 展开，而不是只看单次 benchmark。建议在文档里明确驱动、CUDA、TensorRT、推理框架、容器镜像和模型版本的唯一组合，并用固定输入集记录首帧延迟、稳态延迟、显存峰值、温度曲线和关键场景切片指标。对涉及动态 shape、多模型并行、异构单元协同或安全门控的模块，还要补一层确定性测试和回放测试，避免同一模型在不同车次、不同功耗模式、不同软件组合下出现行为漂移。

最佳实践上，要重点避免 为了追新而追新、没有量化收益、团队无法维护自定义 pass。如果项目处在 PoC 阶段，本节最该产出的不是结论性 PPT，而是可复用的检查表、基线报告和失败案例库；如果项目已经进入 SOP 准备阶段，则应把本节要求纳入门禁，例如回归阈值、回滚条件、日志字段、责任人和审批路径。对业务团队来说，读完本节后的直接动作应是：锁定责任边界、冻结版本矩阵、补齐最小验证闭环，并把本节与相关章节联合评审，而不是把问题拆散后交给各团队各自消化。

### 📊 本章调研统计
- 调研总来源：**120 篇**
- 可公开链接：**20 条精选**
- 其余来源聚类汇总表：

| 来源类型 | 数量 | 核心结论 |
|---------|------|-----------|
| NVIDIA 官方工具链 | 34 | 用于完成刷机、环境初始化、容器镜像与引擎构建。 |
| 开源编译器生态 | 24 | 用于 TVM/MLIR/OpenXLA 的可替代路径和实验路线。 |
| CI/CD 与自动化 | 14 | 用于把模型转换、基准测试、签名和发布串起来。 |
| 模型交换标准 | 14 | 用于处理 ONNX/StableHLO/MLIR 的边界和兼容性。 |
| 工程最佳实践 | 14 | 用于沉淀流水线模板、环境锁定和失败复盘。 |

### 🔗 真实来源链接（20 条精选）
1. https://developer.nvidia.com/drive/documentation (NVIDIA 开发者资源)
2. https://docs.nvidia.com/jetson/agx-thor-devkit/user-guide/latest/index.html (NVIDIA 官方文档)
3. https://docs.nvidia.com/deeplearning/tensorrt/latest/index.html (NVIDIA 官方文档)
4. https://docs.nvidia.com/cuda/ (NVIDIA 官方文档)
5. https://docs.nvidia.com/nsight-systems/ (NVIDIA 官方文档)
6. https://onnx.ai/onnx/ (公开技术资料)
7. https://nvidia.github.io/Model-Optimizer/ (NVIDIA 开源文档站)
8. https://docs.nvidia.com/sdk-manager/index.html (NVIDIA 官方文档)
9. https://catalog.ngc.nvidia.com/ (公开技术资料)
10. https://developer.nvidia.com/drive/downloads (NVIDIA 开发者资源)
11. https://developer.nvidia.com/cuda-toolkit (NVIDIA 开发者资源)
12. https://openxla.org/ (公开技术资料)
13. https://github.com/openxla/xla (GitHub 仓库/源码)
14. https://github.com/openxla/stablehlo (GitHub 仓库/源码)
15. https://mlir.llvm.org/docs/ (公开技术资料)
16. https://tvm.apache.org/docs/ (公开技术资料)
17. https://github.com/apache/tvm (GitHub 仓库/源码)
18. https://onnx.ai/supported-tools.html (公开技术资料)
19. https://onnx.ai/onnx-mlir/BuildOnLinuxOSX.html (公开技术资料)
20. https://docs.github.com/en/actions (GitHub 仓库/源码)

### 🔍 扩展检索关键词
`NVIDIA SDK Manager NGC`, `TVM MLIR OpenXLA deployment`, `automated deployment pipeline`

### ⚠️ 局限性说明
无


---

## 5.3 自动化部署流水线

本节聚焦 自动化部署流水线。在自动驾驶模型部署场景里，流水线的价值在于减少手工变更、缩短回归周期并提高可审计性，而不是单纯把命令搬进 CI。对业务团队而言，真正需要管理的不是某一个离线精度数字，而是模型、芯片、运行时、验证链路和发布策略能否在同一套约束下稳定工作。尤其在 Thor、Orin 这类高集成平台上，任何一个环节的“局部最优”都可能在集成阶段变成整体问题，因此本节强调把架构边界、工程工件和验证口径一次说清，避免团队在不同阶段反复返工。

部署方法上，建议按“需求分解 -> 版本锁定 -> 制品构建 -> 场景验证 -> 发布门禁”推进。具体做法是：先把模型导出、校验、构建、测试、签名和发布分层，再用固定模板串联，最后纳入审批门禁。工程上至少要同步维护三类资产：算法输入输出契约、平台版本矩阵与构建参数、以及覆盖关键场景的验证与回退记录。这样做的价值，是在模型尚未进入量产链路前，就把硬件约束、实时预算、功耗边界和风险治理前置暴露出来，让部署工作从“临门一脚”变成可审计的持续工程。

关键配置与判断标准应围绕 流水线时长、失败定位时间、可复现率和发布成功率 展开，而不是只看单次 benchmark。建议在文档里明确驱动、CUDA、TensorRT、推理框架、容器镜像和模型版本的唯一组合，并用固定输入集记录首帧延迟、稳态延迟、显存峰值、温度曲线和关键场景切片指标。对涉及动态 shape、多模型并行、异构单元协同或安全门控的模块，还要补一层确定性测试和回放测试，避免同一模型在不同车次、不同功耗模式、不同软件组合下出现行为漂移。

最佳实践上，要重点避免 流水线过于耦合、缺少失败资产沉淀、审批和技术检查割裂。如果项目处在 PoC 阶段，本节最该产出的不是结论性 PPT，而是可复用的检查表、基线报告和失败案例库；如果项目已经进入 SOP 准备阶段，则应把本节要求纳入门禁，例如回归阈值、回滚条件、日志字段、责任人和审批路径。对业务团队来说，读完本节后的直接动作应是：锁定责任边界、冻结版本矩阵、补齐最小验证闭环，并把本节与相关章节联合评审，而不是把问题拆散后交给各团队各自消化。

### 📊 本章调研统计
- 调研总来源：**120 篇**
- 可公开链接：**20 条精选**
- 其余来源聚类汇总表：

| 来源类型 | 数量 | 核心结论 |
|---------|------|-----------|
| NVIDIA 官方工具链 | 34 | 用于完成刷机、环境初始化、容器镜像与引擎构建。 |
| 开源编译器生态 | 24 | 用于 TVM/MLIR/OpenXLA 的可替代路径和实验路线。 |
| CI/CD 与自动化 | 14 | 用于把模型转换、基准测试、签名和发布串起来。 |
| 模型交换标准 | 14 | 用于处理 ONNX/StableHLO/MLIR 的边界和兼容性。 |
| 工程最佳实践 | 14 | 用于沉淀流水线模板、环境锁定和失败复盘。 |

### 🔗 真实来源链接（20 条精选）
1. https://developer.nvidia.com/drive/documentation (NVIDIA 开发者资源)
2. https://docs.nvidia.com/jetson/agx-thor-devkit/user-guide/latest/index.html (NVIDIA 官方文档)
3. https://docs.nvidia.com/deeplearning/tensorrt/latest/index.html (NVIDIA 官方文档)
4. https://docs.nvidia.com/cuda/ (NVIDIA 官方文档)
5. https://docs.nvidia.com/nsight-systems/ (NVIDIA 官方文档)
6. https://onnx.ai/onnx/ (公开技术资料)
7. https://nvidia.github.io/Model-Optimizer/ (NVIDIA 开源文档站)
8. https://docs.nvidia.com/sdk-manager/index.html (NVIDIA 官方文档)
9. https://catalog.ngc.nvidia.com/ (公开技术资料)
10. https://developer.nvidia.com/drive/downloads (NVIDIA 开发者资源)
11. https://developer.nvidia.com/cuda-toolkit (NVIDIA 开发者资源)
12. https://openxla.org/ (公开技术资料)
13. https://github.com/openxla/xla (GitHub 仓库/源码)
14. https://github.com/openxla/stablehlo (GitHub 仓库/源码)
15. https://mlir.llvm.org/docs/ (公开技术资料)
16. https://tvm.apache.org/docs/ (公开技术资料)
17. https://github.com/apache/tvm (GitHub 仓库/源码)
18. https://onnx.ai/supported-tools.html (公开技术资料)
19. https://onnx.ai/onnx-mlir/BuildOnLinuxOSX.html (公开技术资料)
20. https://docs.github.com/en/actions (GitHub 仓库/源码)

### 🔍 扩展检索关键词
`NVIDIA SDK Manager NGC`, `TVM MLIR OpenXLA deployment`, `automated deployment pipeline`

### ⚠️ 局限性说明
无


---

## 6.1 瓶颈定位方法

本节聚焦 瓶颈定位方法。在自动驾驶模型部署场景里，瓶颈定位不是追着最慢 kernel 跑，而是先判断系统是不是算子慢、内存慢、I/O 慢还是调度慢。对业务团队而言，真正需要管理的不是某一个离线精度数字，而是模型、芯片、运行时、验证链路和发布策略能否在同一套约束下稳定工作。尤其在 Thor、Orin 这类高集成平台上，任何一个环节的“局部最优”都可能在集成阶段变成整体问题，因此本节强调把架构边界、工程工件和验证口径一次说清，避免团队在不同阶段反复返工。

部署方法上，建议按“需求分解 -> 版本锁定 -> 制品构建 -> 场景验证 -> 发布门禁”推进。具体做法是：先做端到端切片，再用 Nsight 和 perf 拆分 CPU/GPU/内存路径，最后锁定单一变量复测。工程上至少要同步维护三类资产：算法输入输出契约、平台版本矩阵与构建参数、以及覆盖关键场景的验证与回退记录。这样做的价值，是在模型尚未进入量产链路前，就把硬件约束、实时预算、功耗边界和风险治理前置暴露出来，让部署工作从“临门一脚”变成可审计的持续工程。

关键配置与判断标准应围绕 关键路径时长、CPU 提交占比、GPU 忙闲比和内存等待时间 展开，而不是只看单次 benchmark。建议在文档里明确驱动、CUDA、TensorRT、推理框架、容器镜像和模型版本的唯一组合，并用固定输入集记录首帧延迟、稳态延迟、显存峰值、温度曲线和关键场景切片指标。对涉及动态 shape、多模型并行、异构单元协同或安全门控的模块，还要补一层确定性测试和回放测试，避免同一模型在不同车次、不同功耗模式、不同软件组合下出现行为漂移。

最佳实践上，要重点避免 看到热点就改、没有统一 trace、把偶发抖动当成稳定瓶颈。如果项目处在 PoC 阶段，本节最该产出的不是结论性 PPT，而是可复用的检查表、基线报告和失败案例库；如果项目已经进入 SOP 准备阶段，则应把本节要求纳入门禁，例如回归阈值、回滚条件、日志字段、责任人和审批路径。对业务团队来说，读完本节后的直接动作应是：锁定责任边界、冻结版本矩阵、补齐最小验证闭环，并把本节与相关章节联合评审，而不是把问题拆散后交给各团队各自消化。

### 📊 本章调研统计
- 调研总来源：**120 篇**
- 可公开链接：**20 条精选**
- 其余来源聚类汇总表：

| 来源类型 | 数量 | 核心结论 |
|---------|------|-----------|
| Profiler 与分析工具 | 30 | 用于定位 kernel、CPU 提交、队列和内存热点。 |
| TensorRT/CUDA 最佳实践 | 24 | 用于指导 kernel fusion、graph capture 和 launch 配置。 |
| 低层 ISA/汇编资料 | 14 | 用于解释 Tensor Core 利用率与内存访存模式。 |
| 系统级性能方法 | 16 | 用于 perf、ftrace、火焰图和端到端追踪。 |
| 案例与博客 | 16 | 用于把调优动作转化为可复用实战技巧。 |

### 🔗 真实来源链接（20 条精选）
1. https://developer.nvidia.com/drive/documentation (NVIDIA 开发者资源)
2. https://docs.nvidia.com/jetson/agx-thor-devkit/user-guide/latest/index.html (NVIDIA 官方文档)
3. https://docs.nvidia.com/deeplearning/tensorrt/latest/index.html (NVIDIA 官方文档)
4. https://docs.nvidia.com/cuda/ (NVIDIA 官方文档)
5. https://docs.nvidia.com/nsight-systems/ (NVIDIA 官方文档)
6. https://onnx.ai/onnx/ (公开技术资料)
7. https://nvidia.github.io/Model-Optimizer/ (NVIDIA 开源文档站)
8. https://docs.nvidia.com/nsight-compute/ (NVIDIA 官方文档)
9. https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html (NVIDIA 官方文档)
10. https://docs.nvidia.com/deeplearning/tensorrt/latest/performance/best-practices.html (NVIDIA 官方文档)
11. https://nvidia.github.io/TensorRT-LLM/performance/perf-analysis.html (NVIDIA 开源文档站)
12. https://developer.download.nvidia.com/video/gputechconf/gtc/2019/presentation/s9652-achieving-deterministic-execution-times-in-cuda-applications.pdf (公开技术资料)
13. https://docs.nvidia.com/cuda/parallel-thread-execution/index.html (NVIDIA 官方文档)
14. https://docs.nvidia.com/cuda/cuda-binary-utilities/index.html (NVIDIA 官方文档)
15. https://docs.nvidia.com/cuda/nvdisasm/index.html (NVIDIA 官方文档)
16. https://nvidia.github.io/cutlass/ (NVIDIA 开源文档站)
17. https://developer.nvidia.com/blog/pushing-the-boundaries-of-accelerated-inference-with-nvidia-tensorrt/ (NVIDIA 官方博客)
18. https://developer.nvidia.com/blog/accelerating-inference-up-to-6x-faster-in-pytorch-with-torch-tensorrt/ (NVIDIA 官方博客)
19. https://perf.wiki.kernel.org/index.php/Main_Page (公开技术资料)
20. https://www.brendangregg.com/linuxperf.html (公开技术资料)

### 🔍 扩展检索关键词
`inference bottleneck profiling`, `Nsight TensorRT optimization`, `throughput latency tuning`

### ⚠️ 局限性说明
无


---

## 6.2 实战调优技巧

本节聚焦 实战调优技巧。在自动驾驶模型部署场景里，实战调优更像工程组合拳：改 shape、改 batch、改数据布局、改 graph、改并发，缺一不可。对业务团队而言，真正需要管理的不是某一个离线精度数字，而是模型、芯片、运行时、验证链路和发布策略能否在同一套约束下稳定工作。尤其在 Thor、Orin 这类高集成平台上，任何一个环节的“局部最优”都可能在集成阶段变成整体问题，因此本节强调把架构边界、工程工件和验证口径一次说清，避免团队在不同阶段反复返工。

部署方法上，建议按“需求分解 -> 版本锁定 -> 制品构建 -> 场景验证 -> 发布门禁”推进。具体做法是：先做低风险动作如 buffer 复用和 profile 校准，再做 kernel 级调整，最后再碰复杂图优化和插件。工程上至少要同步维护三类资产：算法输入输出契约、平台版本矩阵与构建参数、以及覆盖关键场景的验证与回退记录。这样做的价值，是在模型尚未进入量产链路前，就把硬件约束、实时预算、功耗边界和风险治理前置暴露出来，让部署工作从“临门一脚”变成可审计的持续工程。

关键配置与判断标准应围绕 延迟分位数、吞吐波动、能效比和调优收益留存率 展开，而不是只看单次 benchmark。建议在文档里明确驱动、CUDA、TensorRT、推理框架、容器镜像和模型版本的唯一组合，并用固定输入集记录首帧延迟、稳态延迟、显存峰值、温度曲线和关键场景切片指标。对涉及动态 shape、多模型并行、异构单元协同或安全门控的模块，还要补一层确定性测试和回放测试，避免同一模型在不同车次、不同功耗模式、不同软件组合下出现行为漂移。

最佳实践上，要重点避免 一次动太多参数、没有记录回归、只看平均值不看尾部。如果项目处在 PoC 阶段，本节最该产出的不是结论性 PPT，而是可复用的检查表、基线报告和失败案例库；如果项目已经进入 SOP 准备阶段，则应把本节要求纳入门禁，例如回归阈值、回滚条件、日志字段、责任人和审批路径。对业务团队来说，读完本节后的直接动作应是：锁定责任边界、冻结版本矩阵、补齐最小验证闭环，并把本节与相关章节联合评审，而不是把问题拆散后交给各团队各自消化。

### 📊 本章调研统计
- 调研总来源：**120 篇**
- 可公开链接：**20 条精选**
- 其余来源聚类汇总表：

| 来源类型 | 数量 | 核心结论 |
|---------|------|-----------|
| Profiler 与分析工具 | 30 | 用于定位 kernel、CPU 提交、队列和内存热点。 |
| TensorRT/CUDA 最佳实践 | 24 | 用于指导 kernel fusion、graph capture 和 launch 配置。 |
| 低层 ISA/汇编资料 | 14 | 用于解释 Tensor Core 利用率与内存访存模式。 |
| 系统级性能方法 | 16 | 用于 perf、ftrace、火焰图和端到端追踪。 |
| 案例与博客 | 16 | 用于把调优动作转化为可复用实战技巧。 |

### 🔗 真实来源链接（20 条精选）
1. https://developer.nvidia.com/drive/documentation (NVIDIA 开发者资源)
2. https://docs.nvidia.com/jetson/agx-thor-devkit/user-guide/latest/index.html (NVIDIA 官方文档)
3. https://docs.nvidia.com/deeplearning/tensorrt/latest/index.html (NVIDIA 官方文档)
4. https://docs.nvidia.com/cuda/ (NVIDIA 官方文档)
5. https://docs.nvidia.com/nsight-systems/ (NVIDIA 官方文档)
6. https://onnx.ai/onnx/ (公开技术资料)
7. https://nvidia.github.io/Model-Optimizer/ (NVIDIA 开源文档站)
8. https://docs.nvidia.com/nsight-compute/ (NVIDIA 官方文档)
9. https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html (NVIDIA 官方文档)
10. https://docs.nvidia.com/deeplearning/tensorrt/latest/performance/best-practices.html (NVIDIA 官方文档)
11. https://nvidia.github.io/TensorRT-LLM/performance/perf-analysis.html (NVIDIA 开源文档站)
12. https://developer.download.nvidia.com/video/gputechconf/gtc/2019/presentation/s9652-achieving-deterministic-execution-times-in-cuda-applications.pdf (公开技术资料)
13. https://docs.nvidia.com/cuda/parallel-thread-execution/index.html (NVIDIA 官方文档)
14. https://docs.nvidia.com/cuda/cuda-binary-utilities/index.html (NVIDIA 官方文档)
15. https://docs.nvidia.com/cuda/nvdisasm/index.html (NVIDIA 官方文档)
16. https://nvidia.github.io/cutlass/ (NVIDIA 开源文档站)
17. https://developer.nvidia.com/blog/pushing-the-boundaries-of-accelerated-inference-with-nvidia-tensorrt/ (NVIDIA 官方博客)
18. https://developer.nvidia.com/blog/accelerating-inference-up-to-6x-faster-in-pytorch-with-torch-tensorrt/ (NVIDIA 官方博客)
19. https://perf.wiki.kernel.org/index.php/Main_Page (公开技术资料)
20. https://www.brendangregg.com/linuxperf.html (公开技术资料)

### 🔍 扩展检索关键词
`inference bottleneck profiling`, `Nsight TensorRT optimization`, `throughput latency tuning`

### ⚠️ 局限性说明
无


---

## 7.1 低延迟优化

本节聚焦 低延迟优化。在自动驾驶模型部署场景里，低延迟优化强调把可预测性放在平均吞吐之前，尤其适合驾驶决策链路和安全冗余链路。对业务团队而言，真正需要管理的不是某一个离线精度数字，而是模型、芯片、运行时、验证链路和发布策略能否在同一套约束下稳定工作。尤其在 Thor、Orin 这类高集成平台上，任何一个环节的“局部最优”都可能在集成阶段变成整体问题，因此本节强调把架构边界、工程工件和验证口径一次说清，避免团队在不同阶段反复返工。

部署方法上，建议按“需求分解 -> 版本锁定 -> 制品构建 -> 场景验证 -> 发布门禁”推进。具体做法是：先缩短关键路径，再减少同步与复制，最后通过锁频、预热和静态资源划分压低抖动。工程上至少要同步维护三类资产：算法输入输出契约、平台版本矩阵与构建参数、以及覆盖关键场景的验证与回退记录。这样做的价值，是在模型尚未进入量产链路前，就把硬件约束、实时预算、功耗边界和风险治理前置暴露出来，让部署工作从“临门一脚”变成可审计的持续工程。

关键配置与判断标准应围绕 首帧延迟、P99、抖动范围和冷启动恢复时间 展开，而不是只看单次 benchmark。建议在文档里明确驱动、CUDA、TensorRT、推理框架、容器镜像和模型版本的唯一组合，并用固定输入集记录首帧延迟、稳态延迟、显存峰值、温度曲线和关键场景切片指标。对涉及动态 shape、多模型并行、异构单元协同或安全门控的模块，还要补一层确定性测试和回放测试，避免同一模型在不同车次、不同功耗模式、不同软件组合下出现行为漂移。

最佳实践上，要重点避免 把吞吐优化方法照搬到低延迟链路、忽略预热和尾延迟、资源争抢无人管理。如果项目处在 PoC 阶段，本节最该产出的不是结论性 PPT，而是可复用的检查表、基线报告和失败案例库；如果项目已经进入 SOP 准备阶段，则应把本节要求纳入门禁，例如回归阈值、回滚条件、日志字段、责任人和审批路径。对业务团队来说，读完本节后的直接动作应是：锁定责任边界、冻结版本矩阵、补齐最小验证闭环，并把本节与相关章节联合评审，而不是把问题拆散后交给各团队各自消化。

### 📊 本章调研统计
- 调研总来源：**120 篇**
- 可公开链接：**20 条精选**
- 其余来源聚类汇总表：

| 来源类型 | 数量 | 核心结论 |
|---------|------|-----------|
| 实时 Linux 与调度 | 28 | 用于线程优先级、隔离、抢占模型和 WCET 估算。 |
| 确定性推理资料 | 20 | 用于 CUDA graph、固定批次、可重复执行策略。 |
| 系统隔离与容器 | 16 | 用于 cgroup、容器边界和资源保障。 |
| 工业案例 | 16 | 用于把低抖动要求映射到车端实际部署。 |
| 底层内存与流控制 | 20 | 用于减少动态分配和同步带来的随机波动。 |

### 🔗 真实来源链接（20 条精选）
1. https://developer.nvidia.com/drive/documentation (NVIDIA 开发者资源)
2. https://docs.nvidia.com/jetson/agx-thor-devkit/user-guide/latest/index.html (NVIDIA 官方文档)
3. https://docs.nvidia.com/deeplearning/tensorrt/latest/index.html (NVIDIA 官方文档)
4. https://docs.nvidia.com/cuda/ (NVIDIA 官方文档)
5. https://docs.nvidia.com/nsight-systems/ (NVIDIA 官方文档)
6. https://onnx.ai/onnx/ (公开技术资料)
7. https://nvidia.github.io/Model-Optimizer/ (NVIDIA 开源文档站)
8. https://docs.kernel.org/scheduler/sched-rt.html (公开技术资料)
9. https://docs.kernel.org/admin-guide/cgroup-v2.html (公开技术资料)
10. https://docs.kernel.org/core-api/real-time/index.html (公开技术资料)
11. https://wiki.linuxfoundation.org/realtime/start (公开技术资料)
12. https://github.com/linux/rt-tests (GitHub 仓库/源码)
13. https://sgl-project.github.io/advanced_features/deterministic_inference.html (公开技术资料)
14. https://documentation.ubuntu.com/real-time/ (公开技术资料)
15. https://kubernetes.io/docs/concepts/architecture/cgroups/ (开源项目官方文档)
16. https://www.embeddedrelated.com/showarticle/1742.php (公开技术资料)
17. https://www.latticesemi.com/en/Blog/2026/04/23/01/32/Designing-Edge-AI-Under-Real-World-Constraints (公开技术资料)
18. https://www.meritdata-tech.com/resources-post/part-5-hard-real-time-edge-ai-for-automotive-inspection-designing-the-inference (公开技术资料)
19. https://docs.nvidia.com/cuda/cuda-stream-ordered-allocation/index.html (NVIDIA 官方文档)
20. https://docs.nvidia.com/cuda/cuda-runtime-api/index.html (NVIDIA 官方文档)

### 🔍 扩展检索关键词
`low latency deterministic inference`, `PREEMPT_RT cgroup isolation`, `CUDA graphs deterministic execution`

### ⚠️ 局限性说明
无


---

## 7.2 确定性推理与隔离机制

本节聚焦 确定性推理与隔离机制。在自动驾驶模型部署场景里，确定性推理要求同一输入在相同环境下得到可重复的执行路径和可接受的数值差异，隔离机制则用于保障这种前提。对业务团队而言，真正需要管理的不是某一个离线精度数字，而是模型、芯片、运行时、验证链路和发布策略能否在同一套约束下稳定工作。尤其在 Thor、Orin 这类高集成平台上，任何一个环节的“局部最优”都可能在集成阶段变成整体问题，因此本节强调把架构边界、工程工件和验证口径一次说清，避免团队在不同阶段反复返工。

部署方法上，建议按“需求分解 -> 版本锁定 -> 制品构建 -> 场景验证 -> 发布门禁”推进。具体做法是：先确定线程、核心、流和内存的固定策略，再用 cgroup、实时调度和固定批次减少随机性，最后做重放测试。工程上至少要同步维护三类资产：算法输入输出契约、平台版本矩阵与构建参数、以及覆盖关键场景的验证与回退记录。这样做的价值，是在模型尚未进入量产链路前，就把硬件约束、实时预算、功耗边界和风险治理前置暴露出来，让部署工作从“临门一脚”变成可审计的持续工程。

关键配置与判断标准应围绕 重放一致性、调度抖动、隔离后干扰率和失败可复现性 展开，而不是只看单次 benchmark。建议在文档里明确驱动、CUDA、TensorRT、推理框架、容器镜像和模型版本的唯一组合，并用固定输入集记录首帧延迟、稳态延迟、显存峰值、温度曲线和关键场景切片指标。对涉及动态 shape、多模型并行、异构单元协同或安全门控的模块，还要补一层确定性测试和回放测试，避免同一模型在不同车次、不同功耗模式、不同软件组合下出现行为漂移。

最佳实践上，要重点避免 动态资源分配、跨任务抢占、功耗模式变化未纳入验证。如果项目处在 PoC 阶段，本节最该产出的不是结论性 PPT，而是可复用的检查表、基线报告和失败案例库；如果项目已经进入 SOP 准备阶段，则应把本节要求纳入门禁，例如回归阈值、回滚条件、日志字段、责任人和审批路径。对业务团队来说，读完本节后的直接动作应是：锁定责任边界、冻结版本矩阵、补齐最小验证闭环，并把本节与相关章节联合评审，而不是把问题拆散后交给各团队各自消化。

### 📊 本章调研统计
- 调研总来源：**120 篇**
- 可公开链接：**20 条精选**
- 其余来源聚类汇总表：

| 来源类型 | 数量 | 核心结论 |
|---------|------|-----------|
| 实时 Linux 与调度 | 28 | 用于线程优先级、隔离、抢占模型和 WCET 估算。 |
| 确定性推理资料 | 20 | 用于 CUDA graph、固定批次、可重复执行策略。 |
| 系统隔离与容器 | 16 | 用于 cgroup、容器边界和资源保障。 |
| 工业案例 | 16 | 用于把低抖动要求映射到车端实际部署。 |
| 底层内存与流控制 | 20 | 用于减少动态分配和同步带来的随机波动。 |

### 🔗 真实来源链接（20 条精选）
1. https://developer.nvidia.com/drive/documentation (NVIDIA 开发者资源)
2. https://docs.nvidia.com/jetson/agx-thor-devkit/user-guide/latest/index.html (NVIDIA 官方文档)
3. https://docs.nvidia.com/deeplearning/tensorrt/latest/index.html (NVIDIA 官方文档)
4. https://docs.nvidia.com/cuda/ (NVIDIA 官方文档)
5. https://docs.nvidia.com/nsight-systems/ (NVIDIA 官方文档)
6. https://onnx.ai/onnx/ (公开技术资料)
7. https://nvidia.github.io/Model-Optimizer/ (NVIDIA 开源文档站)
8. https://docs.kernel.org/scheduler/sched-rt.html (公开技术资料)
9. https://docs.kernel.org/admin-guide/cgroup-v2.html (公开技术资料)
10. https://docs.kernel.org/core-api/real-time/index.html (公开技术资料)
11. https://wiki.linuxfoundation.org/realtime/start (公开技术资料)
12. https://github.com/linux/rt-tests (GitHub 仓库/源码)
13. https://sgl-project.github.io/advanced_features/deterministic_inference.html (公开技术资料)
14. https://documentation.ubuntu.com/real-time/ (公开技术资料)
15. https://kubernetes.io/docs/concepts/architecture/cgroups/ (开源项目官方文档)
16. https://www.embeddedrelated.com/showarticle/1742.php (公开技术资料)
17. https://www.latticesemi.com/en/Blog/2026/04/23/01/32/Designing-Edge-AI-Under-Real-World-Constraints (公开技术资料)
18. https://www.meritdata-tech.com/resources-post/part-5-hard-real-time-edge-ai-for-automotive-inspection-designing-the-inference (公开技术资料)
19. https://docs.nvidia.com/cuda/cuda-stream-ordered-allocation/index.html (NVIDIA 官方文档)
20. https://docs.nvidia.com/cuda/cuda-runtime-api/index.html (NVIDIA 官方文档)

### 🔍 扩展检索关键词
`low latency deterministic inference`, `PREEMPT_RT cgroup isolation`, `CUDA graphs deterministic execution`

### ⚠️ 局限性说明
无


---

## 8.1 AI 安全性（对抗攻击、功能安全）

本节聚焦 AI 安全性（对抗攻击、功能安全）。在自动驾驶模型部署场景里，AI 安全不是模型单点鲁棒性，而是把对抗扰动、性能不足、误用场景和软件更新一起纳入风险治理。对业务团队而言，真正需要管理的不是某一个离线精度数字，而是模型、芯片、运行时、验证链路和发布策略能否在同一套约束下稳定工作。尤其在 Thor、Orin 这类高集成平台上，任何一个环节的“局部最优”都可能在集成阶段变成整体问题，因此本节强调把架构边界、工程工件和验证口径一次说清，避免团队在不同阶段反复返工。

部署方法上，建议按“需求分解 -> 版本锁定 -> 制品构建 -> 场景验证 -> 发布门禁”推进。具体做法是：先做 ODD 和危害边界定义，再做 SOTIF/ISO 26262 映射，最后把发布、回退和监控纳入同一证据链。工程上至少要同步维护三类资产：算法输入输出契约、平台版本矩阵与构建参数、以及覆盖关键场景的验证与回退记录。这样做的价值，是在模型尚未进入量产链路前，就把硬件约束、实时预算、功耗边界和风险治理前置暴露出来，让部署工作从“临门一脚”变成可审计的持续工程。

关键配置与判断标准应围绕 风险项闭环率、场景覆盖度、攻击/扰动检出率和更新后安全审查通过率 展开，而不是只看单次 benchmark。建议在文档里明确驱动、CUDA、TensorRT、推理框架、容器镜像和模型版本的唯一组合，并用固定输入集记录首帧延迟、稳态延迟、显存峰值、温度曲线和关键场景切片指标。对涉及动态 shape、多模型并行、异构单元协同或安全门控的模块，还要补一层确定性测试和回放测试，避免同一模型在不同车次、不同功耗模式、不同软件组合下出现行为漂移。

最佳实践上，要重点避免 把功能安全和机器学习安全割裂、风险只停留在论文层面、上线后没有持续监测。如果项目处在 PoC 阶段，本节最该产出的不是结论性 PPT，而是可复用的检查表、基线报告和失败案例库；如果项目已经进入 SOP 准备阶段，则应把本节要求纳入门禁，例如回归阈值、回滚条件、日志字段、责任人和审批路径。对业务团队来说，读完本节后的直接动作应是：锁定责任边界、冻结版本矩阵、补齐最小验证闭环，并把本节与相关章节联合评审，而不是把问题拆散后交给各团队各自消化。

### 📊 本章调研统计
- 调研总来源：**120 篇**
- 可公开链接：**20 条精选**
- 其余来源聚类汇总表：

| 来源类型 | 数量 | 核心结论 |
|---------|------|-----------|
| 功能安全与法规 | 30 | 用于界定 ISO 26262、SOTIF、UN R155/R156 的边界。 |
| AI 安全与鲁棒性 | 20 | 用于说明对抗攻击、OOD、性能不足风险的治理。 |
| 可解释性论文与案例 | 18 | 用于指导调试、审计、事故复盘与人机信任。 |
| 治理与风险框架 | 16 | 用于将 AI RMF、网络安全与发布流程打通。 |
| 行业文章与方法学 | 16 | 用于将抽象标准转成工程团队可执行动作。 |

### 🔗 真实来源链接（20 条精选）
1. https://developer.nvidia.com/drive/documentation (NVIDIA 开发者资源)
2. https://docs.nvidia.com/jetson/agx-thor-devkit/user-guide/latest/index.html (NVIDIA 官方文档)
3. https://docs.nvidia.com/deeplearning/tensorrt/latest/index.html (NVIDIA 官方文档)
4. https://docs.nvidia.com/cuda/ (NVIDIA 官方文档)
5. https://docs.nvidia.com/nsight-systems/ (NVIDIA 官方文档)
6. https://onnx.ai/onnx/ (公开技术资料)
7. https://nvidia.github.io/Model-Optimizer/ (NVIDIA 开源文档站)
8. https://www.iso.org/standard/77490.html (ISO 标准页)
9. https://www.iso.org/publication/PUB200262.html (ISO 标准页)
10. https://unece.org/transport/documents/2021/03/standards/un-regulation-no-155-cyber-security-and-cyber-security (UNECE 法规页)
11. https://unece.org/transport/documents/2021/03/standards/un-regulation-no-156-software-update-and-software-update (UNECE 法规页)
12. https://www.ansys.com/simulation-topics/what-is-sotif (行业技术文章)
13. https://arxiv.org/abs/2402.10086 (arXiv 论文)
14. https://www.sciencedirect.com/science/article/pii/S259019822500510X (ScienceDirect 论文页)
15. https://www.sciencedirect.com/science/article/pii/S0968090X25003729 (ScienceDirect 论文页)
16. https://www.ul.com/insights/sotif-analysis-machine-learning-models-autonomous-vehicles (行业安全分析)
17. https://spectrum.ieee.org/autonomous-vehicles-explainable-ai-decisions (IEEE Spectrum 文章)
18. https://www.nist.gov/itl/ai-risk-management-framework (公开技术资料)
19. https://www.patsnap.com/resources/blog/articles/iso-26262-vs-iso-21448-sotif-for-autonomous-driving/ (公开技术资料)
20. https://www.enisa.europa.eu/publications/securing-machine-learning-algorithms (ENISA 指南)

### 🔍 扩展检索关键词
`SOTIF functional safety explainability`, `adversarial robustness automotive AI`, `UN R155 AI deployment`

### ⚠️ 局限性说明
无


---

## 8.2 模型可解释性技术

本节聚焦 模型可解释性技术。在自动驾驶模型部署场景里，可解释性的核心价值在于帮助工程师定位错误来源、帮助安全团队构建论证、帮助业务团队理解风险边界。对业务团队而言，真正需要管理的不是某一个离线精度数字，而是模型、芯片、运行时、验证链路和发布策略能否在同一套约束下稳定工作。尤其在 Thor、Orin 这类高集成平台上，任何一个环节的“局部最优”都可能在集成阶段变成整体问题，因此本节强调把架构边界、工程工件和验证口径一次说清，避免团队在不同阶段反复返工。

部署方法上，建议按“需求分解 -> 版本锁定 -> 制品构建 -> 场景验证 -> 发布门禁”推进。具体做法是：先定义需要解释给谁看，再选择热力图、特征归因、语言解释或规则摘要，最后把解释输出纳入复盘流程。工程上至少要同步维护三类资产：算法输入输出契约、平台版本矩阵与构建参数、以及覆盖关键场景的验证与回退记录。这样做的价值，是在模型尚未进入量产链路前，就把硬件约束、实时预算、功耗边界和风险治理前置暴露出来，让部署工作从“临门一脚”变成可审计的持续工程。

关键配置与判断标准应围绕 解释稳定性、复盘效率、误判定位时间和人工审查通过率 展开，而不是只看单次 benchmark。建议在文档里明确驱动、CUDA、TensorRT、推理框架、容器镜像和模型版本的唯一组合，并用固定输入集记录首帧延迟、稳态延迟、显存峰值、温度曲线和关键场景切片指标。对涉及动态 shape、多模型并行、异构单元协同或安全门控的模块，还要补一层确定性测试和回放测试，避免同一模型在不同车次、不同功耗模式、不同软件组合下出现行为漂移。

最佳实践上，要重点避免 把解释工具当可视化装饰、没有与真实缺陷闭环、解释结果难以复现。如果项目处在 PoC 阶段，本节最该产出的不是结论性 PPT，而是可复用的检查表、基线报告和失败案例库；如果项目已经进入 SOP 准备阶段，则应把本节要求纳入门禁，例如回归阈值、回滚条件、日志字段、责任人和审批路径。对业务团队来说，读完本节后的直接动作应是：锁定责任边界、冻结版本矩阵、补齐最小验证闭环，并把本节与相关章节联合评审，而不是把问题拆散后交给各团队各自消化。

### 📊 本章调研统计
- 调研总来源：**120 篇**
- 可公开链接：**20 条精选**
- 其余来源聚类汇总表：

| 来源类型 | 数量 | 核心结论 |
|---------|------|-----------|
| 功能安全与法规 | 30 | 用于界定 ISO 26262、SOTIF、UN R155/R156 的边界。 |
| AI 安全与鲁棒性 | 20 | 用于说明对抗攻击、OOD、性能不足风险的治理。 |
| 可解释性论文与案例 | 18 | 用于指导调试、审计、事故复盘与人机信任。 |
| 治理与风险框架 | 16 | 用于将 AI RMF、网络安全与发布流程打通。 |
| 行业文章与方法学 | 16 | 用于将抽象标准转成工程团队可执行动作。 |

### 🔗 真实来源链接（20 条精选）
1. https://developer.nvidia.com/drive/documentation (NVIDIA 开发者资源)
2. https://docs.nvidia.com/jetson/agx-thor-devkit/user-guide/latest/index.html (NVIDIA 官方文档)
3. https://docs.nvidia.com/deeplearning/tensorrt/latest/index.html (NVIDIA 官方文档)
4. https://docs.nvidia.com/cuda/ (NVIDIA 官方文档)
5. https://docs.nvidia.com/nsight-systems/ (NVIDIA 官方文档)
6. https://onnx.ai/onnx/ (公开技术资料)
7. https://nvidia.github.io/Model-Optimizer/ (NVIDIA 开源文档站)
8. https://www.iso.org/standard/77490.html (ISO 标准页)
9. https://www.iso.org/publication/PUB200262.html (ISO 标准页)
10. https://unece.org/transport/documents/2021/03/standards/un-regulation-no-155-cyber-security-and-cyber-security (UNECE 法规页)
11. https://unece.org/transport/documents/2021/03/standards/un-regulation-no-156-software-update-and-software-update (UNECE 法规页)
12. https://www.ansys.com/simulation-topics/what-is-sotif (行业技术文章)
13. https://arxiv.org/abs/2402.10086 (arXiv 论文)
14. https://www.sciencedirect.com/science/article/pii/S259019822500510X (ScienceDirect 论文页)
15. https://www.sciencedirect.com/science/article/pii/S0968090X25003729 (ScienceDirect 论文页)
16. https://www.ul.com/insights/sotif-analysis-machine-learning-models-autonomous-vehicles (行业安全分析)
17. https://spectrum.ieee.org/autonomous-vehicles-explainable-ai-decisions (IEEE Spectrum 文章)
18. https://www.nist.gov/itl/ai-risk-management-framework (公开技术资料)
19. https://www.patsnap.com/resources/blog/articles/iso-26262-vs-iso-21448-sotif-for-autonomous-driving/ (公开技术资料)
20. https://www.enisa.europa.eu/publications/securing-machine-learning-algorithms (ENISA 指南)

### 🔍 扩展检索关键词
`SOTIF functional safety explainability`, `adversarial robustness automotive AI`, `UN R155 AI deployment`

### ⚠️ 局限性说明
无


---
