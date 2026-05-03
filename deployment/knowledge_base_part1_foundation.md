# 自动驾驶模型部署知识库（第一部分：硬件、部署与推理基础）

本文件收录第 1 至第 4 章的全部二级章节。第 0 章总览索引见 `00-overview-index-by-role.md`。

## 1.1 NVIDIA Thor 关键硬件单元

本节聚焦 NVIDIA Thor 关键硬件单元。在自动驾驶模型部署场景里，Blackwell GPU、Arm CPU、DLA/ISP 与高速 I/O 如何共同决定车端推理上限。对业务团队而言，真正需要管理的不是某一个离线精度数字，而是模型、芯片、运行时、验证链路和发布策略能否在同一套约束下稳定工作。尤其在 Thor、Orin 这类高集成平台上，任何一个环节的“局部最优”都可能在集成阶段变成整体问题，因此本节强调把架构边界、工程工件和验证口径一次说清，避免团队在不同阶段反复返工。

部署方法上，建议按“需求分解 -> 版本锁定 -> 制品构建 -> 场景验证 -> 发布门禁”推进。具体做法是：先画出感知到规划的数据路径，再把关键算子绑定到 GPU、DLA 或 CPU，最后核对带宽与温升余量。工程上至少要同步维护三类资产：算法输入输出契约、平台版本矩阵与构建参数、以及覆盖关键场景的验证与回退记录。这样做的价值，是在模型尚未进入量产链路前，就把硬件约束、实时预算、功耗边界和风险治理前置暴露出来，让部署工作从“临门一脚”变成可审计的持续工程。

关键配置与判断标准应围绕 端到端延迟、带宽利用率、温度稳定时间和硬件单元占用率 展开，而不是只看单次 benchmark。建议在文档里明确驱动、CUDA、TensorRT、推理框架、容器镜像和模型版本的唯一组合，并用固定输入集记录首帧延迟、稳态延迟、显存峰值、温度曲线和关键场景切片指标。对涉及动态 shape、多模型并行、异构单元协同或安全门控的模块，还要补一层确定性测试和回放测试，避免同一模型在不同车次、不同功耗模式、不同软件组合下出现行为漂移。

最佳实践上，要重点避免 只看峰值 TOPS、忽略带宽和热设计、把安全岛和传感器链路当成外围问题。如果项目处在 PoC 阶段，本节最该产出的不是结论性 PPT，而是可复用的检查表、基线报告和失败案例库；如果项目已经进入 SOP 准备阶段，则应把本节要求纳入门禁，例如回归阈值、回滚条件、日志字段、责任人和审批路径。对业务团队来说，读完本节后的直接动作应是：锁定责任边界、冻结版本矩阵、补齐最小验证闭环，并把本节与相关章节联合评审，而不是把问题拆散后交给各团队各自消化。

### 📊 本章调研统计
- 调研总来源：**120 篇**
- 可公开链接：**20 条精选**
- 其余来源聚类汇总表：

| 来源类型 | 数量 | 核心结论 |
|---------|------|-----------|
| 官方平台文档 | 40 | 用于确认 Thor/DriveOS/Jetson Thor 的硬件单元、接口、版本与限制。 |
| 芯片与软件白皮书 | 18 | 用于理解 Blackwell、DriveWorks、TensorRT for DRIVE 的平台定位。 |
| 性能与功耗资料 | 16 | 用于判断带宽、功耗模式、热设计和持续性能差异。 |
| 论坛与经验材料 | 12 | 用于定位公开文档未写透的 bring-up 坑点与版本差异。 |
| 经典体系结构资料 | 14 | 用于补足 GPU、CPU、DLA、ISP 协同的基础理解。 |

### 🔗 真实来源链接（20 条精选）
1. https://developer.nvidia.com/drive/documentation (NVIDIA 开发者资源)
2. https://docs.nvidia.com/jetson/agx-thor-devkit/user-guide/latest/index.html (NVIDIA 官方文档)
3. https://docs.nvidia.com/deeplearning/tensorrt/latest/index.html (NVIDIA 官方文档)
4. https://docs.nvidia.com/cuda/ (NVIDIA 官方文档)
5. https://docs.nvidia.com/nsight-systems/ (NVIDIA 官方文档)
6. https://onnx.ai/onnx/ (公开技术资料)
7. https://nvidia.github.io/Model-Optimizer/ (NVIDIA 开源文档站)
8. https://developer.nvidia.cn/blog/accelerate-autonomous-vehicle-development-with-the-nvidia-drive-agx-thor-developer-kit/ (NVIDIA 官方博客)
9. https://developer.nvidia.com/downloads/drive/docs/nvidia-drive-agx-thor-platform-for-developers.pdf (NVIDIA 开发者资源)
10. https://developer.nvidia.com/drive/agx (NVIDIA 开发者资源)
11. https://developer.nvidia.com/drive/ecosystem-thor (NVIDIA 开发者资源)
12. https://developer.nvidia.com/docs/drive/drive-os/7.0.3/public/drive-os-linux-sdk/index.html (NVIDIA 开发者资源)
13. https://developer.nvidia.com/docs/drive/drive-os/7.0.3/public/drive-os-tensorrt-developer-guide/index.html (NVIDIA 开发者资源)
14. https://docs.nvidia.com/jetson/archives/r38.2/DeveloperGuide/SO/JetsonThorSeries.html (NVIDIA 官方文档)
15. https://docs.nvidia.com/jetson/archives/r38.2.1/DeveloperGuide/SD/PlatformPowerAndPerformance/JetsonThor.html (NVIDIA 官方文档)
16. https://developer.nvidia.com/downloads/drive-agx-thor-hardware-quick-start-guide.pdf (NVIDIA 开发者资源)
17. https://developer.nvidia.com/downloads/assets/embedded/secure/jetson/thor/docs/jetson-thor-series-modules-datasheet_ds-11945-001.pdf (NVIDIA 开发者资源)
18. https://forums.developer.nvidia.com/t/technical-reference-manual/344810 (NVIDIA 开发者资源)
19. https://docs.nvidia.com/deeplearning/cudnn/backend/v9.7.1/ (NVIDIA 官方文档)
20. https://docs.nvidia.com/cutlass/index.html (NVIDIA 官方文档)

### 🔍 扩展检索关键词
`Thor Blackwell SoC`, `DriveOS Jetson Thor architecture`, `automotive AI memory bandwidth`

### ⚠️ 局限性说明
无


---

## 1.2 硬件性能瓶颈与优化策略

本节聚焦 硬件性能瓶颈与优化策略。在自动驾驶模型部署场景里，算力并不等于可交付性能，真正决定体验的是内存、访存模式、I/O 同步和功耗模式。对业务团队而言，真正需要管理的不是某一个离线精度数字，而是模型、芯片、运行时、验证链路和发布策略能否在同一套约束下稳定工作。尤其在 Thor、Orin 这类高集成平台上，任何一个环节的“局部最优”都可能在集成阶段变成整体问题，因此本节强调把架构边界、工程工件和验证口径一次说清，避免团队在不同阶段反复返工。

部署方法上，建议按“需求分解 -> 版本锁定 -> 制品构建 -> 场景验证 -> 发布门禁”推进。具体做法是：先做瓶颈拆分，再做算子分流与批次裁剪，再根据功耗模式调优频率、缓存和数据布局。工程上至少要同步维护三类资产：算法输入输出契约、平台版本矩阵与构建参数、以及覆盖关键场景的验证与回退记录。这样做的价值，是在模型尚未进入量产链路前，就把硬件约束、实时预算、功耗边界和风险治理前置暴露出来，让部署工作从“临门一脚”变成可审计的持续工程。

关键配置与判断标准应围绕 P50/P99 延迟、显存峰值、EMC/DRAM 压力和热节流次数 展开，而不是只看单次 benchmark。建议在文档里明确驱动、CUDA、TensorRT、推理框架、容器镜像和模型版本的唯一组合，并用固定输入集记录首帧延迟、稳态延迟、显存峰值、温度曲线和关键场景切片指标。对涉及动态 shape、多模型并行、异构单元协同或安全门控的模块，还要补一层确定性测试和回放测试，避免同一模型在不同车次、不同功耗模式、不同软件组合下出现行为漂移。

最佳实践上，要重点避免 单点优化破坏系统平衡、CPU 回退被忽略、散热与频率策略没有联动。如果项目处在 PoC 阶段，本节最该产出的不是结论性 PPT，而是可复用的检查表、基线报告和失败案例库；如果项目已经进入 SOP 准备阶段，则应把本节要求纳入门禁，例如回归阈值、回滚条件、日志字段、责任人和审批路径。对业务团队来说，读完本节后的直接动作应是：锁定责任边界、冻结版本矩阵、补齐最小验证闭环，并把本节与相关章节联合评审，而不是把问题拆散后交给各团队各自消化。

### 📊 本章调研统计
- 调研总来源：**120 篇**
- 可公开链接：**20 条精选**
- 其余来源聚类汇总表：

| 来源类型 | 数量 | 核心结论 |
|---------|------|-----------|
| 官方平台文档 | 40 | 用于确认 Thor/DriveOS/Jetson Thor 的硬件单元、接口、版本与限制。 |
| 芯片与软件白皮书 | 18 | 用于理解 Blackwell、DriveWorks、TensorRT for DRIVE 的平台定位。 |
| 性能与功耗资料 | 16 | 用于判断带宽、功耗模式、热设计和持续性能差异。 |
| 论坛与经验材料 | 12 | 用于定位公开文档未写透的 bring-up 坑点与版本差异。 |
| 经典体系结构资料 | 14 | 用于补足 GPU、CPU、DLA、ISP 协同的基础理解。 |

### 🔗 真实来源链接（20 条精选）
1. https://developer.nvidia.com/drive/documentation (NVIDIA 开发者资源)
2. https://docs.nvidia.com/jetson/agx-thor-devkit/user-guide/latest/index.html (NVIDIA 官方文档)
3. https://docs.nvidia.com/deeplearning/tensorrt/latest/index.html (NVIDIA 官方文档)
4. https://docs.nvidia.com/cuda/ (NVIDIA 官方文档)
5. https://docs.nvidia.com/nsight-systems/ (NVIDIA 官方文档)
6. https://onnx.ai/onnx/ (公开技术资料)
7. https://nvidia.github.io/Model-Optimizer/ (NVIDIA 开源文档站)
8. https://developer.nvidia.cn/blog/accelerate-autonomous-vehicle-development-with-the-nvidia-drive-agx-thor-developer-kit/ (NVIDIA 官方博客)
9. https://developer.nvidia.com/downloads/drive/docs/nvidia-drive-agx-thor-platform-for-developers.pdf (NVIDIA 开发者资源)
10. https://developer.nvidia.com/drive/agx (NVIDIA 开发者资源)
11. https://developer.nvidia.com/drive/ecosystem-thor (NVIDIA 开发者资源)
12. https://developer.nvidia.com/docs/drive/drive-os/7.0.3/public/drive-os-linux-sdk/index.html (NVIDIA 开发者资源)
13. https://developer.nvidia.com/docs/drive/drive-os/7.0.3/public/drive-os-tensorrt-developer-guide/index.html (NVIDIA 开发者资源)
14. https://docs.nvidia.com/jetson/archives/r38.2/DeveloperGuide/SO/JetsonThorSeries.html (NVIDIA 官方文档)
15. https://docs.nvidia.com/jetson/archives/r38.2.1/DeveloperGuide/SD/PlatformPowerAndPerformance/JetsonThor.html (NVIDIA 官方文档)
16. https://developer.nvidia.com/downloads/drive-agx-thor-hardware-quick-start-guide.pdf (NVIDIA 开发者资源)
17. https://developer.nvidia.com/downloads/assets/embedded/secure/jetson/thor/docs/jetson-thor-series-modules-datasheet_ds-11945-001.pdf (NVIDIA 开发者资源)
18. https://forums.developer.nvidia.com/t/technical-reference-manual/344810 (NVIDIA 开发者资源)
19. https://docs.nvidia.com/deeplearning/cudnn/backend/v9.7.1/ (NVIDIA 官方文档)
20. https://docs.nvidia.com/cutlass/index.html (NVIDIA 官方文档)

### 🔍 扩展检索关键词
`Thor Blackwell SoC`, `DriveOS Jetson Thor architecture`, `automotive AI memory bandwidth`

### ⚠️ 局限性说明
无


---

## 2.1 两段式模型部署

本节聚焦 两段式模型部署。在自动驾驶模型部署场景里，把感知/预测与规划控制解耦，可以换来更清晰的验证边界、接口稳定性和回退策略。对业务团队而言，真正需要管理的不是某一个离线精度数字，而是模型、芯片、运行时、验证链路和发布策略能否在同一套约束下稳定工作。尤其在 Thor、Orin 这类高集成平台上，任何一个环节的“局部最优”都可能在集成阶段变成整体问题，因此本节强调把架构边界、工程工件和验证口径一次说清，避免团队在不同阶段反复返工。

部署方法上，建议按“需求分解 -> 版本锁定 -> 制品构建 -> 场景验证 -> 发布门禁”推进。具体做法是：先稳定中间表示和接口契约，再做模块级引擎构建，最后在闭环回放中验证时序预算。工程上至少要同步维护三类资产：算法输入输出契约、平台版本矩阵与构建参数、以及覆盖关键场景的验证与回退记录。这样做的价值，是在模型尚未进入量产链路前，就把硬件约束、实时预算、功耗边界和风险治理前置暴露出来，让部署工作从“临门一脚”变成可审计的持续工程。

关键配置与判断标准应围绕 模块级精度、跨模块时延、接口稳定性和回放闭环通过率 展开，而不是只看单次 benchmark。建议在文档里明确驱动、CUDA、TensorRT、推理框架、容器镜像和模型版本的唯一组合，并用固定输入集记录首帧延迟、稳态延迟、显存峰值、温度曲线和关键场景切片指标。对涉及动态 shape、多模型并行、异构单元协同或安全门控的模块，还要补一层确定性测试和回放测试，避免同一模型在不同车次、不同功耗模式、不同软件组合下出现行为漂移。

最佳实践上，要重点避免 中间表示过宽导致带宽放大、模块接口频繁变化、回退路径没有真实演练。如果项目处在 PoC 阶段，本节最该产出的不是结论性 PPT，而是可复用的检查表、基线报告和失败案例库；如果项目已经进入 SOP 准备阶段，则应把本节要求纳入门禁，例如回归阈值、回滚条件、日志字段、责任人和审批路径。对业务团队来说，读完本节后的直接动作应是：锁定责任边界、冻结版本矩阵、补齐最小验证闭环，并把本节与相关章节联合评审，而不是把问题拆散后交给各团队各自消化。

### 📊 本章调研统计
- 调研总来源：**120 篇**
- 可公开链接：**20 条精选**
- 其余来源聚类汇总表：

| 来源类型 | 数量 | 核心结论 |
|---------|------|-----------|
| 端到端/两段式论文 | 42 | 用于比较任务分解、控制接口、训练目标与部署代价。 |
| 开源实现 | 20 | 用于核对多传感器输入、BEV 表达和后处理实现。 |
| 官方部署文档 | 14 | 用于把论文模型映射到 Thor/TensorRT 的可执行路径。 |
| 业务案例与博客 | 10 | 用于解释选型逻辑和项目里程碑上的折中。 |
| 经典感知基线 | 14 | 用于建立对照组，避免只看最新论文而忽略可部署性。 |

### 🔗 真实来源链接（20 条精选）
1. https://developer.nvidia.com/drive/documentation (NVIDIA 开发者资源)
2. https://docs.nvidia.com/jetson/agx-thor-devkit/user-guide/latest/index.html (NVIDIA 官方文档)
3. https://docs.nvidia.com/deeplearning/tensorrt/latest/index.html (NVIDIA 官方文档)
4. https://docs.nvidia.com/cuda/ (NVIDIA 官方文档)
5. https://docs.nvidia.com/nsight-systems/ (NVIDIA 官方文档)
6. https://onnx.ai/onnx/ (公开技术资料)
7. https://nvidia.github.io/Model-Optimizer/ (NVIDIA 开源文档站)
8. https://arxiv.org/abs/2203.17270 (arXiv 论文)
9. https://arxiv.org/abs/2203.04050 (arXiv 论文)
10. https://arxiv.org/abs/2110.06922 (arXiv 论文)
11. https://arxiv.org/abs/2203.05625 (arXiv 论文)
12. https://arxiv.org/abs/2308.04559 (arXiv 论文)
13. https://arxiv.org/abs/2208.14437 (arXiv 论文)
14. https://arxiv.org/abs/2112.11790 (arXiv 论文)
15. https://arxiv.org/abs/2212.10156 (arXiv 论文)
16. https://arxiv.org/abs/2303.12077 (arXiv 论文)
17. https://arxiv.org/abs/2005.12872 (arXiv 论文)
18. https://arxiv.org/abs/2206.07959 (arXiv 论文)
19. https://github.com/open-mmlab/mmdetection3d (GitHub 仓库/源码)
20. https://github.com/open-mmlab/OpenPCDet (GitHub 仓库/源码)

### 🔍 扩展检索关键词
`two-stage deployment autonomous driving`, `end-to-end driving deployment`, `BEV perception planning integration`

### ⚠️ 局限性说明
无


---

## 2.2 端到端模型部署

本节聚焦 端到端模型部署。在自动驾驶模型部署场景里，端到端部署把感知、交互和控制目标放进一体化图中，能降低人工接口损失，但会放大调试难度。对业务团队而言，真正需要管理的不是某一个离线精度数字，而是模型、芯片、运行时、验证链路和发布策略能否在同一套约束下稳定工作。尤其在 Thor、Orin 这类高集成平台上，任何一个环节的“局部最优”都可能在集成阶段变成整体问题，因此本节强调把架构边界、工程工件和验证口径一次说清，避免团队在不同阶段反复返工。

部署方法上，建议按“需求分解 -> 版本锁定 -> 制品构建 -> 场景验证 -> 发布门禁”推进。具体做法是：先选定可解释的输出头和安全包络，再做端到端图导出、量化和闭环验证，最后定义影子模式观察期。工程上至少要同步维护三类资产：算法输入输出契约、平台版本矩阵与构建参数、以及覆盖关键场景的验证与回退记录。这样做的价值，是在模型尚未进入量产链路前，就把硬件约束、实时预算、功耗边界和风险治理前置暴露出来，让部署工作从“临门一脚”变成可审计的持续工程。

关键配置与判断标准应围绕 轨迹误差、场景鲁棒性、控制平滑度和影子模式告警率 展开，而不是只看单次 benchmark。建议在文档里明确驱动、CUDA、TensorRT、推理框架、容器镜像和模型版本的唯一组合，并用固定输入集记录首帧延迟、稳态延迟、显存峰值、温度曲线和关键场景切片指标。对涉及动态 shape、多模型并行、异构单元协同或安全门控的模块，还要补一层确定性测试和回放测试，避免同一模型在不同车次、不同功耗模式、不同软件组合下出现行为漂移。

最佳实践上，要重点避免 只看离线指标、忽略解释性和安全门控、把训练便利性误认为上线便利性。如果项目处在 PoC 阶段，本节最该产出的不是结论性 PPT，而是可复用的检查表、基线报告和失败案例库；如果项目已经进入 SOP 准备阶段，则应把本节要求纳入门禁，例如回归阈值、回滚条件、日志字段、责任人和审批路径。对业务团队来说，读完本节后的直接动作应是：锁定责任边界、冻结版本矩阵、补齐最小验证闭环，并把本节与相关章节联合评审，而不是把问题拆散后交给各团队各自消化。

### 📊 本章调研统计
- 调研总来源：**120 篇**
- 可公开链接：**20 条精选**
- 其余来源聚类汇总表：

| 来源类型 | 数量 | 核心结论 |
|---------|------|-----------|
| 端到端/两段式论文 | 42 | 用于比较任务分解、控制接口、训练目标与部署代价。 |
| 开源实现 | 20 | 用于核对多传感器输入、BEV 表达和后处理实现。 |
| 官方部署文档 | 14 | 用于把论文模型映射到 Thor/TensorRT 的可执行路径。 |
| 业务案例与博客 | 10 | 用于解释选型逻辑和项目里程碑上的折中。 |
| 经典感知基线 | 14 | 用于建立对照组，避免只看最新论文而忽略可部署性。 |

### 🔗 真实来源链接（20 条精选）
1. https://developer.nvidia.com/drive/documentation (NVIDIA 开发者资源)
2. https://docs.nvidia.com/jetson/agx-thor-devkit/user-guide/latest/index.html (NVIDIA 官方文档)
3. https://docs.nvidia.com/deeplearning/tensorrt/latest/index.html (NVIDIA 官方文档)
4. https://docs.nvidia.com/cuda/ (NVIDIA 官方文档)
5. https://docs.nvidia.com/nsight-systems/ (NVIDIA 官方文档)
6. https://onnx.ai/onnx/ (公开技术资料)
7. https://nvidia.github.io/Model-Optimizer/ (NVIDIA 开源文档站)
8. https://arxiv.org/abs/2203.17270 (arXiv 论文)
9. https://arxiv.org/abs/2203.04050 (arXiv 论文)
10. https://arxiv.org/abs/2110.06922 (arXiv 论文)
11. https://arxiv.org/abs/2203.05625 (arXiv 论文)
12. https://arxiv.org/abs/2308.04559 (arXiv 论文)
13. https://arxiv.org/abs/2208.14437 (arXiv 论文)
14. https://arxiv.org/abs/2112.11790 (arXiv 论文)
15. https://arxiv.org/abs/2212.10156 (arXiv 论文)
16. https://arxiv.org/abs/2303.12077 (arXiv 论文)
17. https://arxiv.org/abs/2005.12872 (arXiv 论文)
18. https://arxiv.org/abs/2206.07959 (arXiv 论文)
19. https://github.com/open-mmlab/mmdetection3d (GitHub 仓库/源码)
20. https://github.com/open-mmlab/OpenPCDet (GitHub 仓库/源码)

### 🔍 扩展检索关键词
`two-stage deployment autonomous driving`, `end-to-end driving deployment`, `BEV perception planning integration`

### ⚠️ 局限性说明
无


---

## 2.3 优劣对比与选型建议

本节聚焦 优劣对比与选型建议。在自动驾驶模型部署场景里，选型的关键不是押注某一种范式，而是让团队能力、验证成本和硬件预算三者对齐。对业务团队而言，真正需要管理的不是某一个离线精度数字，而是模型、芯片、运行时、验证链路和发布策略能否在同一套约束下稳定工作。尤其在 Thor、Orin 这类高集成平台上，任何一个环节的“局部最优”都可能在集成阶段变成整体问题，因此本节强调把架构边界、工程工件和验证口径一次说清，避免团队在不同阶段反复返工。

部署方法上，建议按“需求分解 -> 版本锁定 -> 制品构建 -> 场景验证 -> 发布门禁”推进。具体做法是：先做场景复杂度与团队成熟度盘点，再评估两段式与端到端的验证成本，最后给出阶段性路线图。工程上至少要同步维护三类资产：算法输入输出契约、平台版本矩阵与构建参数、以及覆盖关键场景的验证与回退记录。这样做的价值，是在模型尚未进入量产链路前，就把硬件约束、实时预算、功耗边界和风险治理前置暴露出来，让部署工作从“临门一脚”变成可审计的持续工程。

关键配置与判断标准应围绕 研发人力投入、验证工时、上线风险和平台复用率 展开，而不是只看单次 benchmark。建议在文档里明确驱动、CUDA、TensorRT、推理框架、容器镜像和模型版本的唯一组合，并用固定输入集记录首帧延迟、稳态延迟、显存峰值、温度曲线和关键场景切片指标。对涉及动态 shape、多模型并行、异构单元协同或安全门控的模块，还要补一层确定性测试和回放测试，避免同一模型在不同车次、不同功耗模式、不同软件组合下出现行为漂移。

最佳实践上，要重点避免 在 PoC 阶段过早锁死路线、忽略后续合规和回滚、没有给跨团队协作预留接口。如果项目处在 PoC 阶段，本节最该产出的不是结论性 PPT，而是可复用的检查表、基线报告和失败案例库；如果项目已经进入 SOP 准备阶段，则应把本节要求纳入门禁，例如回归阈值、回滚条件、日志字段、责任人和审批路径。对业务团队来说，读完本节后的直接动作应是：锁定责任边界、冻结版本矩阵、补齐最小验证闭环，并把本节与相关章节联合评审，而不是把问题拆散后交给各团队各自消化。

### 📊 本章调研统计
- 调研总来源：**120 篇**
- 可公开链接：**20 条精选**
- 其余来源聚类汇总表：

| 来源类型 | 数量 | 核心结论 |
|---------|------|-----------|
| 端到端/两段式论文 | 42 | 用于比较任务分解、控制接口、训练目标与部署代价。 |
| 开源实现 | 20 | 用于核对多传感器输入、BEV 表达和后处理实现。 |
| 官方部署文档 | 14 | 用于把论文模型映射到 Thor/TensorRT 的可执行路径。 |
| 业务案例与博客 | 10 | 用于解释选型逻辑和项目里程碑上的折中。 |
| 经典感知基线 | 14 | 用于建立对照组，避免只看最新论文而忽略可部署性。 |

### 🔗 真实来源链接（20 条精选）
1. https://developer.nvidia.com/drive/documentation (NVIDIA 开发者资源)
2. https://docs.nvidia.com/jetson/agx-thor-devkit/user-guide/latest/index.html (NVIDIA 官方文档)
3. https://docs.nvidia.com/deeplearning/tensorrt/latest/index.html (NVIDIA 官方文档)
4. https://docs.nvidia.com/cuda/ (NVIDIA 官方文档)
5. https://docs.nvidia.com/nsight-systems/ (NVIDIA 官方文档)
6. https://onnx.ai/onnx/ (公开技术资料)
7. https://nvidia.github.io/Model-Optimizer/ (NVIDIA 开源文档站)
8. https://arxiv.org/abs/2203.17270 (arXiv 论文)
9. https://arxiv.org/abs/2203.04050 (arXiv 论文)
10. https://arxiv.org/abs/2110.06922 (arXiv 论文)
11. https://arxiv.org/abs/2203.05625 (arXiv 论文)
12. https://arxiv.org/abs/2308.04559 (arXiv 论文)
13. https://arxiv.org/abs/2208.14437 (arXiv 论文)
14. https://arxiv.org/abs/2112.11790 (arXiv 论文)
15. https://arxiv.org/abs/2212.10156 (arXiv 论文)
16. https://arxiv.org/abs/2303.12077 (arXiv 论文)
17. https://arxiv.org/abs/2005.12872 (arXiv 论文)
18. https://arxiv.org/abs/2206.07959 (arXiv 论文)
19. https://github.com/open-mmlab/mmdetection3d (GitHub 仓库/源码)
20. https://github.com/open-mmlab/OpenPCDet (GitHub 仓库/源码)

### 🔍 扩展检索关键词
`two-stage deployment autonomous driving`, `end-to-end driving deployment`, `BEV perception planning integration`

### ⚠️ 局限性说明
无


---

## 3.1 主流推理框架对比与最佳实践

本节聚焦 主流推理框架对比与最佳实践。在自动驾驶模型部署场景里，TensorRT、TRT-LLM、Edge-LLM、vLLM、SGLang 和 ONNX Runtime 的差异，本质上是优化目标和集成边界的差异。对业务团队而言，真正需要管理的不是某一个离线精度数字，而是模型、芯片、运行时、验证链路和发布策略能否在同一套约束下稳定工作。尤其在 Thor、Orin 这类高集成平台上，任何一个环节的“局部最优”都可能在集成阶段变成整体问题，因此本节强调把架构边界、工程工件和验证口径一次说清，避免团队在不同阶段反复返工。

部署方法上，建议按“需求分解 -> 版本锁定 -> 制品构建 -> 场景验证 -> 发布门禁”推进。具体做法是：先按车端、边端、云端三个场景分层，再比较图优化、动态 shape、插件、观测性和部署复杂度，最后形成统一基线。工程上至少要同步维护三类资产：算法输入输出契约、平台版本矩阵与构建参数、以及覆盖关键场景的验证与回退记录。这样做的价值，是在模型尚未进入量产链路前，就把硬件约束、实时预算、功耗边界和风险治理前置暴露出来，让部署工作从“临门一脚”变成可审计的持续工程。

关键配置与判断标准应围绕 引擎构建时间、稳态延迟、插件数量和跨版本迁移成本 展开，而不是只看单次 benchmark。建议在文档里明确驱动、CUDA、TensorRT、推理框架、容器镜像和模型版本的唯一组合，并用固定输入集记录首帧延迟、稳态延迟、显存峰值、温度曲线和关键场景切片指标。对涉及动态 shape、多模型并行、异构单元协同或安全门控的模块，还要补一层确定性测试和回放测试，避免同一模型在不同车次、不同功耗模式、不同软件组合下出现行为漂移。

最佳实践上，要重点避免 用服务端框架直接套车端、忽视插件维护成本、缺少统一 benchmark 口径。如果项目处在 PoC 阶段，本节最该产出的不是结论性 PPT，而是可复用的检查表、基线报告和失败案例库；如果项目已经进入 SOP 准备阶段，则应把本节要求纳入门禁，例如回归阈值、回滚条件、日志字段、责任人和审批路径。对业务团队来说，读完本节后的直接动作应是：锁定责任边界、冻结版本矩阵、补齐最小验证闭环，并把本节与相关章节联合评审，而不是把问题拆散后交给各团队各自消化。

### 📊 本章调研统计
- 调研总来源：**120 篇**
- 可公开链接：**20 条精选**
- 其余来源聚类汇总表：

| 来源类型 | 数量 | 核心结论 |
|---------|------|-----------|
| NVIDIA 推理栈 | 36 | 用于对齐 TensorRT、TRT-LLM、Edge-LLM 在车端与边缘侧的定位。 |
| 开源推理框架 | 22 | 用于比较 vLLM、SGLang、ONNX Runtime 的生态与集成代价。 |
| CUDA/内存文档 | 16 | 用于说明流、缓存、分配策略与图捕获的底层约束。 |
| 性能分析材料 | 12 | 用于给出 DAG 调度、队列深度和 profiler 的观测方法。 |
| 经典编译/运行时资料 | 14 | 用于补充框架比较之外的执行模型理解。 |

### 🔗 真实来源链接（20 条精选）
1. https://developer.nvidia.com/drive/documentation (NVIDIA 开发者资源)
2. https://docs.nvidia.com/jetson/agx-thor-devkit/user-guide/latest/index.html (NVIDIA 官方文档)
3. https://docs.nvidia.com/deeplearning/tensorrt/latest/index.html (NVIDIA 官方文档)
4. https://docs.nvidia.com/cuda/ (NVIDIA 官方文档)
5. https://docs.nvidia.com/nsight-systems/ (NVIDIA 官方文档)
6. https://onnx.ai/onnx/ (公开技术资料)
7. https://nvidia.github.io/Model-Optimizer/ (NVIDIA 开源文档站)
8. https://nvidia.github.io/TensorRT-LLM/ (NVIDIA 开源文档站)
9. https://github.com/NVIDIA/TensorRT (GitHub 仓库/源码)
10. https://github.com/NVIDIA/TensorRT-LLM (GitHub 仓库/源码)
11. https://nvidia.github.io/TensorRT-Edge-LLM/ (NVIDIA 开源文档站)
12. https://github.com/NVIDIA/TensorRT-Edge-LLM (GitHub 仓库/源码)
13. https://docs.vllm.ai/ (公开技术资料)
14. https://github.com/vllm-project/vllm (GitHub 仓库/源码)
15. https://docs.sglang.ai/ (公开技术资料)
16. https://github.com/sgl-project/sglang (GitHub 仓库/源码)
17. https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/index.html (NVIDIA 官方文档)
18. https://github.com/triton-inference-server/server (GitHub 仓库/源码)
19. https://onnxruntime.ai/docs/ (公开技术资料)
20. https://github.com/microsoft/onnxruntime (GitHub 仓库/源码)

### 🔍 扩展检索关键词
`TensorRT vLLM SGLang comparison`, `CUDA memory management inference`, `DAG scheduling compute graph`

### ⚠️ 局限性说明
无


---

## 3.2 CUDA 优化与内存管理

本节聚焦 CUDA 优化与内存管理。在自动驾驶模型部署场景里，在 Thor 上，很多性能问题不是模型本身，而是内存分配、流同步、Host-Device 拷贝和缓存复用导致。对业务团队而言，真正需要管理的不是某一个离线精度数字，而是模型、芯片、运行时、验证链路和发布策略能否在同一套约束下稳定工作。尤其在 Thor、Orin 这类高集成平台上，任何一个环节的“局部最优”都可能在集成阶段变成整体问题，因此本节强调把架构边界、工程工件和验证口径一次说清，避免团队在不同阶段反复返工。

部署方法上，建议按“需求分解 -> 版本锁定 -> 制品构建 -> 场景验证 -> 发布门禁”推进。具体做法是：先固定分配策略和 stream 模式，再做 pinned memory、graph capture、buffer 复用和异步流水线设计。工程上至少要同步维护三类资产：算法输入输出契约、平台版本矩阵与构建参数、以及覆盖关键场景的验证与回退记录。这样做的价值，是在模型尚未进入量产链路前，就把硬件约束、实时预算、功耗边界和风险治理前置暴露出来，让部署工作从“临门一脚”变成可审计的持续工程。

关键配置与判断标准应围绕 Host-Device 传输时间、kernel 启动开销、分配次数和缓存命中率 展开，而不是只看单次 benchmark。建议在文档里明确驱动、CUDA、TensorRT、推理框架、容器镜像和模型版本的唯一组合，并用固定输入集记录首帧延迟、稳态延迟、显存峰值、温度曲线和关键场景切片指标。对涉及动态 shape、多模型并行、异构单元协同或安全门控的模块，还要补一层确定性测试和回放测试，避免同一模型在不同车次、不同功耗模式、不同软件组合下出现行为漂移。

最佳实践上，要重点避免 频繁动态分配、隐式同步、零碎 memcpy 和上下文切换过多。如果项目处在 PoC 阶段，本节最该产出的不是结论性 PPT，而是可复用的检查表、基线报告和失败案例库；如果项目已经进入 SOP 准备阶段，则应把本节要求纳入门禁，例如回归阈值、回滚条件、日志字段、责任人和审批路径。对业务团队来说，读完本节后的直接动作应是：锁定责任边界、冻结版本矩阵、补齐最小验证闭环，并把本节与相关章节联合评审，而不是把问题拆散后交给各团队各自消化。

### 📊 本章调研统计
- 调研总来源：**120 篇**
- 可公开链接：**20 条精选**
- 其余来源聚类汇总表：

| 来源类型 | 数量 | 核心结论 |
|---------|------|-----------|
| NVIDIA 推理栈 | 36 | 用于对齐 TensorRT、TRT-LLM、Edge-LLM 在车端与边缘侧的定位。 |
| 开源推理框架 | 22 | 用于比较 vLLM、SGLang、ONNX Runtime 的生态与集成代价。 |
| CUDA/内存文档 | 16 | 用于说明流、缓存、分配策略与图捕获的底层约束。 |
| 性能分析材料 | 12 | 用于给出 DAG 调度、队列深度和 profiler 的观测方法。 |
| 经典编译/运行时资料 | 14 | 用于补充框架比较之外的执行模型理解。 |

### 🔗 真实来源链接（20 条精选）
1. https://developer.nvidia.com/drive/documentation (NVIDIA 开发者资源)
2. https://docs.nvidia.com/jetson/agx-thor-devkit/user-guide/latest/index.html (NVIDIA 官方文档)
3. https://docs.nvidia.com/deeplearning/tensorrt/latest/index.html (NVIDIA 官方文档)
4. https://docs.nvidia.com/cuda/ (NVIDIA 官方文档)
5. https://docs.nvidia.com/nsight-systems/ (NVIDIA 官方文档)
6. https://onnx.ai/onnx/ (公开技术资料)
7. https://nvidia.github.io/Model-Optimizer/ (NVIDIA 开源文档站)
8. https://nvidia.github.io/TensorRT-LLM/ (NVIDIA 开源文档站)
9. https://github.com/NVIDIA/TensorRT (GitHub 仓库/源码)
10. https://github.com/NVIDIA/TensorRT-LLM (GitHub 仓库/源码)
11. https://nvidia.github.io/TensorRT-Edge-LLM/ (NVIDIA 开源文档站)
12. https://github.com/NVIDIA/TensorRT-Edge-LLM (GitHub 仓库/源码)
13. https://docs.vllm.ai/ (公开技术资料)
14. https://github.com/vllm-project/vllm (GitHub 仓库/源码)
15. https://docs.sglang.ai/ (公开技术资料)
16. https://github.com/sgl-project/sglang (GitHub 仓库/源码)
17. https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/index.html (NVIDIA 官方文档)
18. https://github.com/triton-inference-server/server (GitHub 仓库/源码)
19. https://onnxruntime.ai/docs/ (公开技术资料)
20. https://github.com/microsoft/onnxruntime (GitHub 仓库/源码)

### 🔍 扩展检索关键词
`TensorRT vLLM SGLang comparison`, `CUDA memory management inference`, `DAG scheduling compute graph`

### ⚠️ 局限性说明
无


---

## 3.3 计算图优化与 DAG 调度

本节聚焦 计算图优化与 DAG 调度。在自动驾驶模型部署场景里，计算图优化的目标不是把图变得更复杂，而是让算子融合、执行顺序和资源使用更符合实时系统要求。对业务团队而言，真正需要管理的不是某一个离线精度数字，而是模型、芯片、运行时、验证链路和发布策略能否在同一套约束下稳定工作。尤其在 Thor、Orin 这类高集成平台上，任何一个环节的“局部最优”都可能在集成阶段变成整体问题，因此本节强调把架构边界、工程工件和验证口径一次说清，避免团队在不同阶段反复返工。

部署方法上，建议按“需求分解 -> 版本锁定 -> 制品构建 -> 场景验证 -> 发布门禁”推进。具体做法是：先梳理关键路径 DAG，再审查 shape、分支、后处理和多模型并行，最后用 profiler 验证图级优化是否真的省时。工程上至少要同步维护三类资产：算法输入输出契约、平台版本矩阵与构建参数、以及覆盖关键场景的验证与回退记录。这样做的价值，是在模型尚未进入量产链路前，就把硬件约束、实时预算、功耗边界和风险治理前置暴露出来，让部署工作从“临门一脚”变成可审计的持续工程。

关键配置与判断标准应围绕 关键路径长度、graph capture 成功率、编译次数和队列深度 展开，而不是只看单次 benchmark。建议在文档里明确驱动、CUDA、TensorRT、推理框架、容器镜像和模型版本的唯一组合，并用固定输入集记录首帧延迟、稳态延迟、显存峰值、温度曲线和关键场景切片指标。对涉及动态 shape、多模型并行、异构单元协同或安全门控的模块，还要补一层确定性测试和回放测试，避免同一模型在不同车次、不同功耗模式、不同软件组合下出现行为漂移。

最佳实践上，要重点避免 图优化把问题藏起来、动态 shape 触发重编译、多模型共用资源导致抖动。如果项目处在 PoC 阶段，本节最该产出的不是结论性 PPT，而是可复用的检查表、基线报告和失败案例库；如果项目已经进入 SOP 准备阶段，则应把本节要求纳入门禁，例如回归阈值、回滚条件、日志字段、责任人和审批路径。对业务团队来说，读完本节后的直接动作应是：锁定责任边界、冻结版本矩阵、补齐最小验证闭环，并把本节与相关章节联合评审，而不是把问题拆散后交给各团队各自消化。

### 📊 本章调研统计
- 调研总来源：**120 篇**
- 可公开链接：**20 条精选**
- 其余来源聚类汇总表：

| 来源类型 | 数量 | 核心结论 |
|---------|------|-----------|
| NVIDIA 推理栈 | 36 | 用于对齐 TensorRT、TRT-LLM、Edge-LLM 在车端与边缘侧的定位。 |
| 开源推理框架 | 22 | 用于比较 vLLM、SGLang、ONNX Runtime 的生态与集成代价。 |
| CUDA/内存文档 | 16 | 用于说明流、缓存、分配策略与图捕获的底层约束。 |
| 性能分析材料 | 12 | 用于给出 DAG 调度、队列深度和 profiler 的观测方法。 |
| 经典编译/运行时资料 | 14 | 用于补充框架比较之外的执行模型理解。 |

### 🔗 真实来源链接（20 条精选）
1. https://developer.nvidia.com/drive/documentation (NVIDIA 开发者资源)
2. https://docs.nvidia.com/jetson/agx-thor-devkit/user-guide/latest/index.html (NVIDIA 官方文档)
3. https://docs.nvidia.com/deeplearning/tensorrt/latest/index.html (NVIDIA 官方文档)
4. https://docs.nvidia.com/cuda/ (NVIDIA 官方文档)
5. https://docs.nvidia.com/nsight-systems/ (NVIDIA 官方文档)
6. https://onnx.ai/onnx/ (公开技术资料)
7. https://nvidia.github.io/Model-Optimizer/ (NVIDIA 开源文档站)
8. https://nvidia.github.io/TensorRT-LLM/ (NVIDIA 开源文档站)
9. https://github.com/NVIDIA/TensorRT (GitHub 仓库/源码)
10. https://github.com/NVIDIA/TensorRT-LLM (GitHub 仓库/源码)
11. https://nvidia.github.io/TensorRT-Edge-LLM/ (NVIDIA 开源文档站)
12. https://github.com/NVIDIA/TensorRT-Edge-LLM (GitHub 仓库/源码)
13. https://docs.vllm.ai/ (公开技术资料)
14. https://github.com/vllm-project/vllm (GitHub 仓库/源码)
15. https://docs.sglang.ai/ (公开技术资料)
16. https://github.com/sgl-project/sglang (GitHub 仓库/源码)
17. https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/index.html (NVIDIA 官方文档)
18. https://github.com/triton-inference-server/server (GitHub 仓库/源码)
19. https://onnxruntime.ai/docs/ (公开技术资料)
20. https://github.com/microsoft/onnxruntime (GitHub 仓库/源码)

### 🔍 扩展检索关键词
`TensorRT vLLM SGLang comparison`, `CUDA memory management inference`, `DAG scheduling compute graph`

### ⚠️ 局限性说明
无


---

## 4.1 量化方法（INT8/FP8/FP16）

本节聚焦 量化方法（INT8/FP8/FP16）。在自动驾驶模型部署场景里，量化的目标不是一味降精度，而是在硬件友好格式和业务可接受误差之间找到稳定平衡。对业务团队而言，真正需要管理的不是某一个离线精度数字，而是模型、芯片、运行时、验证链路和发布策略能否在同一套约束下稳定工作。尤其在 Thor、Orin 这类高集成平台上，任何一个环节的“局部最优”都可能在集成阶段变成整体问题，因此本节强调把架构边界、工程工件和验证口径一次说清，避免团队在不同阶段反复返工。

部署方法上，建议按“需求分解 -> 版本锁定 -> 制品构建 -> 场景验证 -> 发布门禁”推进。具体做法是：先根据算子类型和目标 SoC 选择 FP16/INT8/FP8，再用校准集和敏感层分析决定混合精度边界。工程上至少要同步维护三类资产：算法输入输出契约、平台版本矩阵与构建参数、以及覆盖关键场景的验证与回退记录。这样做的价值，是在模型尚未进入量产链路前，就把硬件约束、实时预算、功耗边界和风险治理前置暴露出来，让部署工作从“临门一脚”变成可审计的持续工程。

关键配置与判断标准应围绕 精度损失、吞吐变化、显存占用和关键场景误判率 展开，而不是只看单次 benchmark。建议在文档里明确驱动、CUDA、TensorRT、推理框架、容器镜像和模型版本的唯一组合，并用固定输入集记录首帧延迟、稳态延迟、显存峰值、温度曲线和关键场景切片指标。对涉及动态 shape、多模型并行、异构单元协同或安全门控的模块，还要补一层确定性测试和回放测试，避免同一模型在不同车次、不同功耗模式、不同软件组合下出现行为漂移。

最佳实践上，要重点避免 把全图统一量化、忽视后处理精度、没有对校准数据做场景覆盖控制。如果项目处在 PoC 阶段，本节最该产出的不是结论性 PPT，而是可复用的检查表、基线报告和失败案例库；如果项目已经进入 SOP 准备阶段，则应把本节要求纳入门禁，例如回归阈值、回滚条件、日志字段、责任人和审批路径。对业务团队来说，读完本节后的直接动作应是：锁定责任边界、冻结版本矩阵、补齐最小验证闭环，并把本节与相关章节联合评审，而不是把问题拆散后交给各团队各自消化。

### 📊 本章调研统计
- 调研总来源：**120 篇**
- 可公开链接：**20 条精选**
- 其余来源聚类汇总表：

| 来源类型 | 数量 | 核心结论 |
|---------|------|-----------|
| 量化官方文档 | 30 | 用于确认 INT8/FP8/FP16/INT4 的硬件支持与语义差异。 |
| ModelOpt 与工具链 | 22 | 用于建立 QAT/PTQ、蒸馏、稀疏统一工作流。 |
| 学术方法论文 | 26 | 用于理解 GPTQ、AWQ、剪枝、蒸馏的理论与边界。 |
| 框架集成材料 | 10 | 用于核对 PyTorch/vLLM/HF 与 NVIDIA 工具的联动方式。 |
| 回退与验证经验 | 12 | 用于总结量化失败、误差扩散和回退流程。 |

### 🔗 真实来源链接（20 条精选）
1. https://developer.nvidia.com/drive/documentation (NVIDIA 开发者资源)
2. https://docs.nvidia.com/jetson/agx-thor-devkit/user-guide/latest/index.html (NVIDIA 官方文档)
3. https://docs.nvidia.com/deeplearning/tensorrt/latest/index.html (NVIDIA 官方文档)
4. https://docs.nvidia.com/cuda/ (NVIDIA 官方文档)
5. https://docs.nvidia.com/nsight-systems/ (NVIDIA 官方文档)
6. https://onnx.ai/onnx/ (公开技术资料)
7. https://nvidia.github.io/Model-Optimizer/ (NVIDIA 开源文档站)
8. https://docs.nvidia.com/deeplearning/tensorrt/latest/inference-library/work-quantized-types.html (NVIDIA 官方文档)
9. https://github.com/NVIDIA/Model-Optimizer (GitHub 仓库/源码)
10. https://docs.nvidia.com/nemo/megatron-bridge/nightly/modelopt/quantization.html (NVIDIA 官方文档)
11. https://docs.vllm.ai/en/stable/features/quantization/modelopt/ (公开技术资料)
12. https://huggingface.co/docs/diffusers/en/quantization/modelopt (开源生态官方文档)
13. https://developer.nvidia.com/blog/streamlining-ai-inference-precision-with-nvidia-tensorrt-model-optimizer/ (NVIDIA 官方博客)
14. https://developer.nvidia.com/blog/end-to-end-training-and-inference-using-fp8-in-transformer-based-models/ (NVIDIA 官方博客)
15. https://developer.nvidia.com/blog/new-ai-model-sparsity-techniques-that-speed-up-inference/ (NVIDIA 官方博客)
16. https://arxiv.org/abs/1712.05877 (arXiv 论文)
17. https://arxiv.org/abs/2210.17323 (arXiv 论文)
18. https://arxiv.org/abs/2306.00978 (arXiv 论文)
19. https://arxiv.org/abs/1503.02531 (arXiv 论文)
20. https://arxiv.org/abs/1803.03635 (arXiv 论文)

### 🔍 扩展检索关键词
`INT8 FP8 PTQ QAT`, `model compression pruning distillation`, `quantization fallback strategy`

### ⚠️ 局限性说明
无


---

## 4.2 量化测试与回退策略

本节聚焦 量化测试与回退策略。在自动驾驶模型部署场景里，量化上线必须和测试、告警、回退一起设计，否则一次小误差就会放大成整车行为差异。对业务团队而言，真正需要管理的不是某一个离线精度数字，而是模型、芯片、运行时、验证链路和发布策略能否在同一套约束下稳定工作。尤其在 Thor、Orin 这类高集成平台上，任何一个环节的“局部最优”都可能在集成阶段变成整体问题，因此本节强调把架构边界、工程工件和验证口径一次说清，避免团队在不同阶段反复返工。

部署方法上，建议按“需求分解 -> 版本锁定 -> 制品构建 -> 场景验证 -> 发布门禁”推进。具体做法是：先定义量化前后基线，再做场景切片验证、影子比较和回退开关，最后把结果入库。工程上至少要同步维护三类资产：算法输入输出契约、平台版本矩阵与构建参数、以及覆盖关键场景的验证与回退记录。这样做的价值，是在模型尚未进入量产链路前，就把硬件约束、实时预算、功耗边界和风险治理前置暴露出来，让部署工作从“临门一脚”变成可审计的持续工程。

关键配置与判断标准应围绕 量化前后分布差异、影子模式偏差、回退成功率和问题复现时间 展开，而不是只看单次 benchmark。建议在文档里明确驱动、CUDA、TensorRT、推理框架、容器镜像和模型版本的唯一组合，并用固定输入集记录首帧延迟、稳态延迟、显存峰值、温度曲线和关键场景切片指标。对涉及动态 shape、多模型并行、异构单元协同或安全门控的模块，还要补一层确定性测试和回放测试，避免同一模型在不同车次、不同功耗模式、不同软件组合下出现行为漂移。

最佳实践上，要重点避免 只做平均精度比较、忽略长尾场景、回退机制停留在文档里。如果项目处在 PoC 阶段，本节最该产出的不是结论性 PPT，而是可复用的检查表、基线报告和失败案例库；如果项目已经进入 SOP 准备阶段，则应把本节要求纳入门禁，例如回归阈值、回滚条件、日志字段、责任人和审批路径。对业务团队来说，读完本节后的直接动作应是：锁定责任边界、冻结版本矩阵、补齐最小验证闭环，并把本节与相关章节联合评审，而不是把问题拆散后交给各团队各自消化。

### 📊 本章调研统计
- 调研总来源：**120 篇**
- 可公开链接：**20 条精选**
- 其余来源聚类汇总表：

| 来源类型 | 数量 | 核心结论 |
|---------|------|-----------|
| 量化官方文档 | 30 | 用于确认 INT8/FP8/FP16/INT4 的硬件支持与语义差异。 |
| ModelOpt 与工具链 | 22 | 用于建立 QAT/PTQ、蒸馏、稀疏统一工作流。 |
| 学术方法论文 | 26 | 用于理解 GPTQ、AWQ、剪枝、蒸馏的理论与边界。 |
| 框架集成材料 | 10 | 用于核对 PyTorch/vLLM/HF 与 NVIDIA 工具的联动方式。 |
| 回退与验证经验 | 12 | 用于总结量化失败、误差扩散和回退流程。 |

### 🔗 真实来源链接（20 条精选）
1. https://developer.nvidia.com/drive/documentation (NVIDIA 开发者资源)
2. https://docs.nvidia.com/jetson/agx-thor-devkit/user-guide/latest/index.html (NVIDIA 官方文档)
3. https://docs.nvidia.com/deeplearning/tensorrt/latest/index.html (NVIDIA 官方文档)
4. https://docs.nvidia.com/cuda/ (NVIDIA 官方文档)
5. https://docs.nvidia.com/nsight-systems/ (NVIDIA 官方文档)
6. https://onnx.ai/onnx/ (公开技术资料)
7. https://nvidia.github.io/Model-Optimizer/ (NVIDIA 开源文档站)
8. https://docs.nvidia.com/deeplearning/tensorrt/latest/inference-library/work-quantized-types.html (NVIDIA 官方文档)
9. https://github.com/NVIDIA/Model-Optimizer (GitHub 仓库/源码)
10. https://docs.nvidia.com/nemo/megatron-bridge/nightly/modelopt/quantization.html (NVIDIA 官方文档)
11. https://docs.vllm.ai/en/stable/features/quantization/modelopt/ (公开技术资料)
12. https://huggingface.co/docs/diffusers/en/quantization/modelopt (开源生态官方文档)
13. https://developer.nvidia.com/blog/streamlining-ai-inference-precision-with-nvidia-tensorrt-model-optimizer/ (NVIDIA 官方博客)
14. https://developer.nvidia.com/blog/end-to-end-training-and-inference-using-fp8-in-transformer-based-models/ (NVIDIA 官方博客)
15. https://developer.nvidia.com/blog/new-ai-model-sparsity-techniques-that-speed-up-inference/ (NVIDIA 官方博客)
16. https://arxiv.org/abs/1712.05877 (arXiv 论文)
17. https://arxiv.org/abs/2210.17323 (arXiv 论文)
18. https://arxiv.org/abs/2306.00978 (arXiv 论文)
19. https://arxiv.org/abs/1503.02531 (arXiv 论文)
20. https://arxiv.org/abs/1803.03635 (arXiv 论文)

### 🔍 扩展检索关键词
`INT8 FP8 PTQ QAT`, `model compression pruning distillation`, `quantization fallback strategy`

### ⚠️ 局限性说明
无


---

## 4.3 模型压缩（剪枝、蒸馏）

本节聚焦 模型压缩（剪枝、蒸馏）。在自动驾驶模型部署场景里，剪枝和蒸馏适合解决模型太重、功耗过高或部署窗口受限的问题，但前提是输出语义稳定。对业务团队而言，真正需要管理的不是某一个离线精度数字，而是模型、芯片、运行时、验证链路和发布策略能否在同一套约束下稳定工作。尤其在 Thor、Orin 这类高集成平台上，任何一个环节的“局部最优”都可能在集成阶段变成整体问题，因此本节强调把架构边界、工程工件和验证口径一次说清，避免团队在不同阶段反复返工。

部署方法上，建议按“需求分解 -> 版本锁定 -> 制品构建 -> 场景验证 -> 发布门禁”推进。具体做法是：先确认 teacher/student 和稀疏化目标，再做分阶段训练、导出和车端 benchmark，最后补齐失效模式测试。工程上至少要同步维护三类资产：算法输入输出契约、平台版本矩阵与构建参数、以及覆盖关键场景的验证与回退记录。这样做的价值，是在模型尚未进入量产链路前，就把硬件约束、实时预算、功耗边界和风险治理前置暴露出来，让部署工作从“临门一脚”变成可审计的持续工程。

关键配置与判断标准应围绕 模型大小、延迟、功耗、关键场景准确率和 teacher/student 差距 展开，而不是只看单次 benchmark。建议在文档里明确驱动、CUDA、TensorRT、推理框架、容器镜像和模型版本的唯一组合，并用固定输入集记录首帧延迟、稳态延迟、显存峰值、温度曲线和关键场景切片指标。对涉及动态 shape、多模型并行、异构单元协同或安全门控的模块，还要补一层确定性测试和回放测试，避免同一模型在不同车次、不同功耗模式、不同软件组合下出现行为漂移。

最佳实践上，要重点避免 只追求压缩率、忽略学生模型失效模式变化、压缩后没有更新验证集。如果项目处在 PoC 阶段，本节最该产出的不是结论性 PPT，而是可复用的检查表、基线报告和失败案例库；如果项目已经进入 SOP 准备阶段，则应把本节要求纳入门禁，例如回归阈值、回滚条件、日志字段、责任人和审批路径。对业务团队来说，读完本节后的直接动作应是：锁定责任边界、冻结版本矩阵、补齐最小验证闭环，并把本节与相关章节联合评审，而不是把问题拆散后交给各团队各自消化。

### 📊 本章调研统计
- 调研总来源：**120 篇**
- 可公开链接：**20 条精选**
- 其余来源聚类汇总表：

| 来源类型 | 数量 | 核心结论 |
|---------|------|-----------|
| 量化官方文档 | 30 | 用于确认 INT8/FP8/FP16/INT4 的硬件支持与语义差异。 |
| ModelOpt 与工具链 | 22 | 用于建立 QAT/PTQ、蒸馏、稀疏统一工作流。 |
| 学术方法论文 | 26 | 用于理解 GPTQ、AWQ、剪枝、蒸馏的理论与边界。 |
| 框架集成材料 | 10 | 用于核对 PyTorch/vLLM/HF 与 NVIDIA 工具的联动方式。 |
| 回退与验证经验 | 12 | 用于总结量化失败、误差扩散和回退流程。 |

### 🔗 真实来源链接（20 条精选）
1. https://developer.nvidia.com/drive/documentation (NVIDIA 开发者资源)
2. https://docs.nvidia.com/jetson/agx-thor-devkit/user-guide/latest/index.html (NVIDIA 官方文档)
3. https://docs.nvidia.com/deeplearning/tensorrt/latest/index.html (NVIDIA 官方文档)
4. https://docs.nvidia.com/cuda/ (NVIDIA 官方文档)
5. https://docs.nvidia.com/nsight-systems/ (NVIDIA 官方文档)
6. https://onnx.ai/onnx/ (公开技术资料)
7. https://nvidia.github.io/Model-Optimizer/ (NVIDIA 开源文档站)
8. https://docs.nvidia.com/deeplearning/tensorrt/latest/inference-library/work-quantized-types.html (NVIDIA 官方文档)
9. https://github.com/NVIDIA/Model-Optimizer (GitHub 仓库/源码)
10. https://docs.nvidia.com/nemo/megatron-bridge/nightly/modelopt/quantization.html (NVIDIA 官方文档)
11. https://docs.vllm.ai/en/stable/features/quantization/modelopt/ (公开技术资料)
12. https://huggingface.co/docs/diffusers/en/quantization/modelopt (开源生态官方文档)
13. https://developer.nvidia.com/blog/streamlining-ai-inference-precision-with-nvidia-tensorrt-model-optimizer/ (NVIDIA 官方博客)
14. https://developer.nvidia.com/blog/end-to-end-training-and-inference-using-fp8-in-transformer-based-models/ (NVIDIA 官方博客)
15. https://developer.nvidia.com/blog/new-ai-model-sparsity-techniques-that-speed-up-inference/ (NVIDIA 官方博客)
16. https://arxiv.org/abs/1712.05877 (arXiv 论文)
17. https://arxiv.org/abs/2210.17323 (arXiv 论文)
18. https://arxiv.org/abs/2306.00978 (arXiv 论文)
19. https://arxiv.org/abs/1503.02531 (arXiv 论文)
20. https://arxiv.org/abs/1803.03635 (arXiv 论文)

### 🔍 扩展检索关键词
`INT8 FP8 PTQ QAT`, `model compression pruning distillation`, `quantization fallback strategy`

### ⚠️ 局限性说明
无


---

## 4.4 QAT vs PTQ

本节聚焦 QAT vs PTQ。在自动驾驶模型部署场景里，QAT 与 PTQ 的选择，本质上是训练成本、时间窗口和上线风险之间的权衡。对业务团队而言，真正需要管理的不是某一个离线精度数字，而是模型、芯片、运行时、验证链路和发布策略能否在同一套约束下稳定工作。尤其在 Thor、Orin 这类高集成平台上，任何一个环节的“局部最优”都可能在集成阶段变成整体问题，因此本节强调把架构边界、工程工件和验证口径一次说清，避免团队在不同阶段反复返工。

部署方法上，建议按“需求分解 -> 版本锁定 -> 制品构建 -> 场景验证 -> 发布门禁”推进。具体做法是：先做 PTQ 快速评估，再对敏感层或关键模型引入 QAT，最后形成统一准入标准和迁移路径。工程上至少要同步维护三类资产：算法输入输出契约、平台版本矩阵与构建参数、以及覆盖关键场景的验证与回退记录。这样做的价值，是在模型尚未进入量产链路前，就把硬件约束、实时预算、功耗边界和风险治理前置暴露出来，让部署工作从“临门一脚”变成可审计的持续工程。

关键配置与判断标准应围绕 训练工时、部署收益、误差恢复程度和版本维护复杂度 展开，而不是只看单次 benchmark。建议在文档里明确驱动、CUDA、TensorRT、推理框架、容器镜像和模型版本的唯一组合，并用固定输入集记录首帧延迟、稳态延迟、显存峰值、温度曲线和关键场景切片指标。对涉及动态 shape、多模型并行、异构单元协同或安全门控的模块，还要补一层确定性测试和回放测试，避免同一模型在不同车次、不同功耗模式、不同软件组合下出现行为漂移。

最佳实践上，要重点避免 把 QAT 当成默认答案、或者在 PTQ 明显失效时仍然硬上。如果项目处在 PoC 阶段，本节最该产出的不是结论性 PPT，而是可复用的检查表、基线报告和失败案例库；如果项目已经进入 SOP 准备阶段，则应把本节要求纳入门禁，例如回归阈值、回滚条件、日志字段、责任人和审批路径。对业务团队来说，读完本节后的直接动作应是：锁定责任边界、冻结版本矩阵、补齐最小验证闭环，并把本节与相关章节联合评审，而不是把问题拆散后交给各团队各自消化。

### 📊 本章调研统计
- 调研总来源：**120 篇**
- 可公开链接：**20 条精选**
- 其余来源聚类汇总表：

| 来源类型 | 数量 | 核心结论 |
|---------|------|-----------|
| 量化官方文档 | 30 | 用于确认 INT8/FP8/FP16/INT4 的硬件支持与语义差异。 |
| ModelOpt 与工具链 | 22 | 用于建立 QAT/PTQ、蒸馏、稀疏统一工作流。 |
| 学术方法论文 | 26 | 用于理解 GPTQ、AWQ、剪枝、蒸馏的理论与边界。 |
| 框架集成材料 | 10 | 用于核对 PyTorch/vLLM/HF 与 NVIDIA 工具的联动方式。 |
| 回退与验证经验 | 12 | 用于总结量化失败、误差扩散和回退流程。 |

### 🔗 真实来源链接（20 条精选）
1. https://developer.nvidia.com/drive/documentation (NVIDIA 开发者资源)
2. https://docs.nvidia.com/jetson/agx-thor-devkit/user-guide/latest/index.html (NVIDIA 官方文档)
3. https://docs.nvidia.com/deeplearning/tensorrt/latest/index.html (NVIDIA 官方文档)
4. https://docs.nvidia.com/cuda/ (NVIDIA 官方文档)
5. https://docs.nvidia.com/nsight-systems/ (NVIDIA 官方文档)
6. https://onnx.ai/onnx/ (公开技术资料)
7. https://nvidia.github.io/Model-Optimizer/ (NVIDIA 开源文档站)
8. https://docs.nvidia.com/deeplearning/tensorrt/latest/inference-library/work-quantized-types.html (NVIDIA 官方文档)
9. https://github.com/NVIDIA/Model-Optimizer (GitHub 仓库/源码)
10. https://docs.nvidia.com/nemo/megatron-bridge/nightly/modelopt/quantization.html (NVIDIA 官方文档)
11. https://docs.vllm.ai/en/stable/features/quantization/modelopt/ (公开技术资料)
12. https://huggingface.co/docs/diffusers/en/quantization/modelopt (开源生态官方文档)
13. https://developer.nvidia.com/blog/streamlining-ai-inference-precision-with-nvidia-tensorrt-model-optimizer/ (NVIDIA 官方博客)
14. https://developer.nvidia.com/blog/end-to-end-training-and-inference-using-fp8-in-transformer-based-models/ (NVIDIA 官方博客)
15. https://developer.nvidia.com/blog/new-ai-model-sparsity-techniques-that-speed-up-inference/ (NVIDIA 官方博客)
16. https://arxiv.org/abs/1712.05877 (arXiv 论文)
17. https://arxiv.org/abs/2210.17323 (arXiv 论文)
18. https://arxiv.org/abs/2306.00978 (arXiv 论文)
19. https://arxiv.org/abs/1503.02531 (arXiv 论文)
20. https://arxiv.org/abs/1803.03635 (arXiv 论文)

### 🔍 扩展检索关键词
`INT8 FP8 PTQ QAT`, `model compression pruning distillation`, `quantization fallback strategy`

### ⚠️ 局限性说明
无


---
