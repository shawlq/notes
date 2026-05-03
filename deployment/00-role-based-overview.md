## 0. 总览索引表（按角色）

### 检索计划与中间汇总

本章是整套自动驾驶模型部署知识库的角色导航页，检索目标不是替代后续 52 个二级章节的深度分析，而是建立“谁应该先读什么、为什么先读”的业务索引。检索关键词按 8 组展开：`NVIDIA DRIVE AGX Thor DriveOS 7 TensorRT 10`, `CUDA Graphs automotive inference`, `TensorRT quantization FP8 INT8 INT4 NVFP4`, `ONNX Runtime TensorRT TVM OpenXLA`, `autonomous driving end-to-end VLA deployment`, `ISO 26262 SOTIF AI safety explainability`, `autonomous driving MLOps OTA shadow mode`, `CARLA Omniverse HIL deterministic replay`。网站范围优先级为 NVIDIA 官方文档与技术博客、arXiv/GitHub、ONNX/TVM/OpenXLA 官方文档、仿真与 MLOps 公开资料、行业标准解读与课程材料；预期数量为 NVIDIA 35 条、论文/综述 20 条、GitHub/开源项目 15 条、开源框架文档 15 条、博客/课程/行业材料 15 条。

50 条资料中间汇总：资料已覆盖 Thor/DriveOS、TensorRT/CUDA、量化与模型压缩、ONNX/TVM/OpenXLA、安全与可解释性六个核心簇。初步结论是：部署工程师需要先掌握 TensorRT、CUDA Graphs、显存与量化；系统架构师需要先理解 Thor/Orin 算力、传感器 I/O、实时隔离和跨芯片边界；功能安全经理需要把 ISO 26262、SOTIF、ISO/SAE 21434、AI 可解释性放在模型指标之前。100 条资料完成汇总：新增 MLOps、OTA、仿真/HIL、功耗热管理、端到端 VLA 与 DAG 调度资料后，可以把全书导航拆成 5 类角色，每类角色都对应一个可执行的阅读路径和交付物。

本章正文采用业务团队视角：先定义角色，再给优先章节，再说明读完后应能产出的工程决策。部署工程师的主要目标是把模型稳定地跑在车端目标平台上，所以优先阅读第 3、4、5、6 章：第 3 章解决推理框架与计算平台选型，第 4 章解决 FP16/FP8/INT8/INT4/NVFP4 等精度策略，第 5 章解决 NVIDIA 工具链和开源编译栈衔接，第 6 章解决瓶颈定位与调优闭环。对这类角色，阅读时不要只看“平均延迟”，还要建立三张表：算子支持/回退表、动态 shape 与 engine cache 表、精度回退与校准数据表。尤其在 Thor/DriveOS 7 相关部署中，TensorRT 10、DriveOS LLM SDK、TensorRT Edge-LLM、CUDA Graphs、Nsight Systems 和 Model Optimizer 是高频工具；它们分别对应 engine 构建、LLM/VLM C++ 运行时、边缘低依赖推理、CPU launch overhead 降低、端到端 trace、量化/剪枝/蒸馏。

系统架构师的主要目标是决定“模型如何嵌入整车计算系统”，所以优先阅读第 1、2、7、13 章。第 1 章帮助判断 Thor 的 Blackwell GPU、Arm CPU、内存带宽、传感器接口与安全能力能否承载业务路线；第 2 章把两段式部署、端到端部署和混合 VLA 架构放到同一张决策图里；第 7 章处理低延迟、确定性、隔离机制和调度约束；第 13 章处理 Thor/Orin/其他芯片之间的算子兼容、量化对齐和 IR 边界。架构师读完后应能输出三类决策：是否把关键链路放到 TensorRT 原生 engine，是否允许 ONNX Runtime/TensorRT EP 作为过渡层，是否需要 TVM/MLIR/OpenXLA 作为跨芯片适配层。对业务团队而言，架构师还要明确哪些指标是芯片能力问题，哪些是模型结构问题，哪些是流水线编排问题。

功能安全经理的主要目标是降低发布风险，所以优先阅读第 8、14 章，并结合第 7、15 章复核实时性与验证证据。自动驾驶 AI 部署不能只用 mAP、NDS、collision rate 或 closed-loop score 做放行依据；还要建立安全 case、可解释性证据、功耗热管理策略和降级策略。ISO 26262 关注系统性故障和随机硬件故障，SOTIF 关注“系统按设计运行但能力不足”导致的风险，ISO/SAE 21434 与 UN R155/R156 关注网络安全和软件更新治理。对 AI 模型而言，可解释性不是展示热力图那么简单，而是要服务于缺陷定位、审计追踪、失效复盘和安全论证。功耗热管理也应进入安全经理视野，因为批量大小、推理频率、显存带宽、低精度策略都会影响热节流，热节流又会反向影响实时性与功能降级。

MLOps 负责人的主要目标是让模型可复现、可回滚、可灰度，所以优先阅读第 12、15 章。自动驾驶模型部署的 MLOps 不只是训练平台管理，而是云端训练、仿真验证、车端部署、OTA、影子模式、金丝雀发布和回滚的闭环。推荐从模型版本哈希、训练数据血缘、量化配置、TensorRT engine 构建参数、目标硬件/DriveOS/TensorRT/CUDA 版本、校准集版本开始建立部署元数据。随后把仿真、HIL、确定性回放、性能回归、精度回归、安全回归接入发布门禁。业务上最重要的不是“能不能发版”，而是“线上发现问题后能不能快速定位是哪一批数据、哪一个模型、哪一个 engine、哪一个车端环境造成的”。

技术负责人需要同时理解路线选择、团队能力和前瞻趋势，所以优先阅读第 9、11、16 章，并按问题回查第 10 章 FAQ。第 9 章覆盖多任务学习、自适应更新和兼容性，第 11 章给团队学习地图与验证清单，第 16 章跟踪 2025-2026 年端上学习、联邦学习、实时适应、下一代芯片和合规趋势。技术负责人要把知识库当成路线图使用：短期看部署稳定性，中期看跨平台迁移成本，长期看端到端/VLA/世界模型是否会改变现有模块边界。建议建立月度技术雷达，把 NVIDIA 官方发布、arXiv 综述、核心 GitHub 项目、DriveOS/TensorRT/CUDA 版本变化和法规动态放在同一张看板里。

| 角色 | 优先阅读章节 | 核心关注点 | 阅读后应产出 |
|------|-------------|-----------|-------------|
| 部署工程师 | 3, 4, 5, 6 | 推理框架、量化、工具链、调优 | engine 构建参数表、量化回退表、性能 profile 报告 |
| 系统架构师 | 1, 2, 7, 13 | 芯片、部署范式、实时性、跨平台 | 硬件/框架选型矩阵、实时预算、跨芯片适配边界 |
| 功能安全经理 | 8, 14 | 安全性、功耗 | 安全 case、可解释性证据、热节流降级策略 |
| MLOps 负责人 | 12, 15 | 版本管理、仿真测试 | 模型血缘、OTA/灰度策略、HIL/回放发布门禁 |
| 技术负责人 | 9, 11, 16 | 高级话题、团队建设、前瞻方向 | 技术路线图、能力建设清单、季度趋势雷达 |

### 📊 本章调研统计
- 调研总来源：**108 篇**
- 可公开链接：**20 条精选**
- 其余来源聚类汇总表：

| 来源类型 | 数量 | 核心结论 |
|---------|------|-----------|
| NVIDIA DRIVE/Thor/DriveOS 官方资料 | 14 | Thor/DriveOS 7 是全书硬件与软件基线，需优先跟踪 TensorRT、CUDA、DriveWorks、Nsight 与安全配置。 |
| TensorRT/CUDA/Nsight 推理优化资料 | 15 | 车端部署的首要瓶颈通常来自动态 shape、engine 构建、CPU launch overhead、显存复用和多流调度。 |
| 量化、压缩与 Model Optimizer 资料 | 10 | FP16/FP8/INT8/INT4/NVFP4 需要以校准集、Q/DQ 图、回退策略和逐层误差分析共同管理。 |
| 开源编译与跨平台部署资料 | 8 | ONNX Runtime、TVM、OpenXLA/StableHLO 能降低迁移成本，但不能替代目标芯片上的端到端 profiling。 |
| 自动驾驶模型架构论文/综述 | 12 | 两段式、端到端、VLA、世界模型会长期共存，部署选型应按安全边界、延迟预算和可解释性拆分。 |
| 安全、可解释性与合规资料 | 10 | ISO 26262、SOTIF、ISO/SAE 21434、UN R155/R156 与 XAI 证据要进入发布门禁，而非事后补文档。 |
| MLOps、OTA、灰度发布资料 | 9 | 车端模型发布必须记录模型、数据、量化、engine、硬件和系统版本血缘，支持影子模式与快速回滚。 |
| 仿真、HIL、确定性回放资料 | 10 | CARLA、Omniverse、AlpaSim、HIL 和芯片级回放是部署前验证闭环，适合与性能/安全回归绑定。 |

### 🔗 真实来源链接（20 条精选）
1. https://developer.nvidia.com/drive/documentation (NVIDIA DRIVE/Thor/DriveOS 官方文档入口)
2. https://developer.nvidia.com/drive/os (NVIDIA DriveOS SDK 官方介绍)
3. https://developer.nvidia.com/blog/accelerate-autonomous-vehicle-development-with-the-nvidia-drive-agx-thor-developer-kit/ (2025 Thor 开发套件与 DriveOS 7 技术博客)
4. https://docs.nvidia.com/deeplearning/tensorrt/latest/index.html (TensorRT 官方文档)
5. https://docs.nvidia.com/deeplearning/tensorrt/10.16.1/performance/optimization.html (TensorRT 性能优化指南)
6. https://docs.nvidia.com/deeplearning/tensorrt/latest/inference-library/work-quantized-types.html (TensorRT 量化类型与 Q/DQ 指南)
7. https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/cuda-graphs.html (CUDA Graphs 编程指南)
8. https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/ (CUDA C++ 最佳实践指南)
9. https://developer.nvidia.com/nsight-systems/get-started (Nsight Systems 官方入口)
10. https://github.com/NVIDIA/Model-Optimizer (NVIDIA Model Optimizer 开源仓库)
11. https://nvidia.github.io/TensorRT-LLM/ (TensorRT-LLM 官方文档)
12. https://developer.nvidia.com/blog/streamline-llm-deployment-for-autonomous-vehicle-applications-with-nvidia-driveos-llm-sdk/ (DriveOS LLM SDK 技术博客)
13. https://developer.nvidia.com/blog/accelerating-llm-and-vlm-inference-for-automotive-and-robotics-with-nvidia-tensorrt-edge-llm/ (2026 TensorRT Edge-LLM 技术博客)
14. https://github.com/NVIDIA/TensorRT-Edge-LLM (TensorRT Edge-LLM 开源仓库)
15. https://onnxruntime.ai/docs/execution-providers/TensorRT-ExecutionProvider.html (ONNX Runtime TensorRT Execution Provider 文档)
16. https://tvm.apache.org/docs/get_started/tutorials/quick_start.html (Apache TVM 快速开始文档)
17. https://openxla.org/ (OpenXLA/StableHLO/PJRT 官方入口)
18. https://arxiv.org/abs/2402.10086 (2024 自动驾驶可解释 AI 系统综述)
19. https://carla.readthedocs.io/en/latest/start_introduction/ (CARLA 自动驾驶仿真文档)
20. https://www.nvidia.com/en-us/use-cases/autonomous-vehicle-simulation/ (NVIDIA 自动驾驶仿真用例与 Omniverse/AlpaSim 入口)

### 🔍 扩展检索关键词
`DRIVE AGX Thor TensorRT 10 DriveOS 7`, `CUDA Graphs deterministic inference automotive`, `TensorRT FP8 INT4 NVFP4 quantization`, `autonomous driving VLA deployment`, `ISO 26262 SOTIF XAI autonomous driving`, `CARLA Omniverse HIL deterministic replay`

### ⚠️ 局限性说明
无

---
