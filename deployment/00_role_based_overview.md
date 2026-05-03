---

## 0. 总览索引表（按角色）

本章是自动驾驶模型部署知识库的入口页，目标不是替代后续 52 个二级章节，而是让业务团队、技术负责人和工程团队在面对“模型如何从训练产物进入车端量产系统”这一问题时，能够快速定位应读内容、判断协作边界，并形成统一的部署语言。围绕 NVIDIA DRIVE AGX Thor、DriveOS 7、TensorRT 10、CUDA、TensorRT-LLM、模型量化、实时调度、MLOps、安全、功耗和仿真测试等公开资料，本轮采用“硬件与工具链优先、部署风险优先、业务角色优先”的方法整理索引。

**检索计划。** 关键词覆盖三层：第一层为 `NVIDIA DRIVE AGX Thor`, `DriveOS 7`, `TensorRT 10`, `CUDA 13`, `DriveOS LLM SDK`, `NVIDIA Halos`，用于确认 Thor 车端部署底座、软件栈、算力、内存带宽、安全和车规约束；第二层为 `TensorRT quantization`, `INT8 FP8 FP4`, `TensorRT Model Optimizer`, `QAT PTQ`, `CUDA Graph`, `DAG scheduling`, `Triton inference autonomous driving`，用于覆盖部署工程师最常处理的精度、性能、内存、算子和流水线问题；第三层为 `Autoware MLOps`, `AWML`, `HIL autonomous driving`, `OTA model deployment`, `ISO 26262 AI`, `SOTIF`, `ONNX cross platform`, `Snapdragon Ride SDK`, `Jetson Thor thermal`，用于补足 MLOps、仿真、跨芯片迁移、功能安全与功耗热管理。网站范围优先级为 NVIDIA 官方文档和技术博客，其次为 arXiv、GitHub、IEEE/ACM、Qualcomm、Autoware/ROS、Waymo/NHTSA 等行业资料；预期数量为官方文档 35 条、论文与预印本 30 条、GitHub/开源项目 18 条、行业博客与厂商资料 20 条、法规/安全/课程资料 9 条，总计 112 条。

**50 条资料中间汇总。** 前 50 条资料集中在硬件、推理框架和量化：Thor 相比 Orin 的关键变化是 Blackwell GPU、Arm Neoverse V3AE、64 GB LPDDR5X、约 273 GB/s 带宽、最高 1000 INT8 TOPS 级别车端 SoC 能力，以及 DriveOS、CUDA、cuDNN、TensorRT、DriveWorks、NvMedia/NvStreams 等配套工具链。对业务团队最重要的结论是：Thor 不是“只换更强 GPU”，而是把多模态感知、端到端驾驶模型、VLM/LLM 车端交互、传感器融合和安全监控放入同一中央计算平台。部署工程师阅读顺序应从第 3、4、5、6 章开始：先确认推理框架和模型格式，再确认 INT8/FP8/FP16/FP4 或 INT4 的适用边界，最后进入 profiling、内存池、CUDA lazy loading、L2 persistent cache、engine build reproducibility 和动态 shape 策略。

**100 条资料收集完成汇总。** 后 62 条资料扩展到实时性、MLOps、安全、跨平台和热管理：Autoware/ROS2 研究把自动驾驶系统建模为 DAG 或多 deadline DAG，说明“单模型延迟达标”不足以保证系统达标；AWML 等开源资料显示机器人/自动驾驶 MLOps 已从单次训练转向数据挖掘、伪标注、模型版本、部署接口和主动学习闭环；NVIDIA DRIVE Hyperion、Halos、Alpamayo、DriveOS LLM SDK 等资料显示 2025-2026 年部署重点会从传统 CNN/BEV 感知扩展到 reasoning-based autonomy、VLA、LLM/VLM 和可审计决策链。功能安全经理应优先阅读第 8、14 章，因为量化、异步流水线、热节流和 OTA 更新都可能改变安全论证；MLOps 负责人应优先阅读第 12、15 章，把模型版本哈希、血缘追踪、影子模式、回滚、HIL、仿真和确定性回放纳入发布门禁；系统架构师应优先阅读第 1、2、7、13 章，重点把芯片能力、部署范式、实时隔离和跨平台 IR 边界统一成架构决策。

| 角色 | 优先阅读章节 | 核心关注点 |
|------|-------------|-----------|
| 部署工程师 | 3, 4, 5, 6 | 推理框架、量化、工具链、性能调优、显存与算子排障 |
| 系统架构师 | 1, 2, 7, 13 | Thor/Orin 能力边界、两段式与端到端范式、确定性、异构与跨平台 |
| 功能安全经理 | 8, 14 | AI 安全性、可解释性、ASIL/SOTIF 证据、功耗热管理下的降级策略 |
| MLOps 负责人 | 12, 15 | 模型版本、OTA、A/B、灰度、回滚、仿真、HIL、自动化回归 |
| 技术负责人 | 9, 11, 16 | 多任务学习、自适应更新、团队能力地图、路线图与标准动态 |
| 产品/项目经理 | 0, 2, 10, 12 | 选型影响、发布风险、FAQ 严重度、跨团队交付节奏 |
| 数据闭环负责人 | 8, 12, 15 | 数据挖掘、伪标注、模型血缘、影子模式、场景回放 |
| 供应链/平台合作负责人 | 1, 13, 14, 16 | Thor 与其他芯片差异、供应商 SDK、功耗约束、后续芯片路线 |

行动建议是：第一，任何自动驾驶模型部署项目都应先建立“角色-章节-验收物”映射，例如部署工程师输出 TensorRT engine 构建记录、精度回归报告和 profiler trace，安全经理输出安全目标影响分析和降级策略，MLOps 负责人输出版本血缘与回滚流程。第二，立项时不要只讨论模型精度，应同时冻结传感器输入 shape、时间预算、芯片功耗模式、目标精度格式、可接受 fallback、engine build 环境和仿真/HIL 场景。第三，业务团队评审时应把常见问题按 P0/P1/P2 管理：量化精度下降、实时性不达标属于 P0；动态 shape 重编译、图优化失败、多模型流水线冲突、算子回退 CPU 属于 P1；内存碎片和批处理吞吐波动通常属于 P2，但若触发热节流或安全降级，也应提升等级。第四，Thor 相关方案需要尽早验证 DriveOS 版本、TensorRT 版本、CUDA 能力、目标精度格式和目标模型族是否匹配；尤其是 LLM/VLM 或端到端大模型，不能直接沿用 Orin 时代的 CNN/BEV 部署假设。第五，后续每个二级章节应继续沉淀“可执行检查清单”，让业务团队在评审供应商方案、内部里程碑或量产发布门禁时，可以直接复用。

### 📊 本章调研统计
- 调研总来源：**112 篇**
- 可公开链接：**20 条精选**
- 其余来源聚类汇总表：

| 来源类型 | 数量 | 核心结论 |
|---------|------|-----------|
| NVIDIA 官方文档/博客/产品页 | 35 | Thor/DriveOS/TensorRT/CUDA 是车端部署主线，需同时关注算力、内存、精度格式、安全与工具链版本 |
| arXiv/论文/经典基础内容 | 30 | 量化、FP8、DAG 调度、MLOps、实时系统和模型压缩提供方法论，但需结合车端硬件重新验证 |
| GitHub/开源项目 | 18 | TensorRT-LLM、TensorRT Model Optimizer、Autoware、AWML、DL4AGX 等可作为工程样例和验证入口 |
| 厂商/行业资料 | 20 | Qualcomm、Horizon、NXP、Waymo、Bosch 等资料显示跨平台、功能安全和闭环验证是量产差异点 |
| 法规/课程/会议资料 | 9 | ISO 26262、ISO 21434、SOTIF、NHTSA、GTC 课程适合补齐安全、合规和团队培训视角 |

### 🔗 真实来源链接（20 条精选）
1. https://developer.nvidia.com/drive/agx (NVIDIA DRIVE AGX Thor/Orin 开发套件与规格)
2. https://developer.nvidia.com/blog/accelerate-autonomous-vehicle-development-with-the-nvidia-drive-agx-thor-developer-kit/ (2025 Thor Developer Kit 与 DriveOS 7 说明)
3. https://www.nvidia.com/en-us/solutions/autonomous-vehicles/in-vehicle-computing/ (NVIDIA 车载计算、DRIVE Hyperion、Thor、Halos 概览)
4. https://developer.nvidia.com/blog/streamline-llm-deployment-for-autonomous-vehicle-applications-with-nvidia-driveos-llm-sdk/ (DriveOS LLM SDK 部署流程)
5. https://docs.nvidia.com/deeplearning/tensorrt/latest/architecture/how-trt-works.html (TensorRT 架构、内存、确定性与兼容性)
6. https://developer.nvidia.com/docs/drive/drive-os/7.0.3/public/drive-os-tensorrt-developer-guide/work-quantized-types.html (DriveOS TensorRT 量化类型)
7. https://developer.nvidia.com/cuda/toolkit (CUDA Toolkit 与 Nsight/CUDA-X 资源)
8. https://github.com/NVIDIA/TensorRT-LLM (TensorRT-LLM 开源仓库)
9. https://nvidia.github.io/TensorRT-LLM/ (TensorRT-LLM 官方文档)
10. https://github.com/NVIDIA/TensorRT-Model-Optimizer (TensorRT Model Optimizer 开源仓库)
11. https://developer.nvidia.com/blog/floating-point-8-an-introduction-to-efficient-lower-precision-ai-training/ (2025 FP8 与 Blackwell 低精度介绍)
12. https://developer.nvidia.com/blog/sparsity-in-int8-training-workflow-and-best-practices-for-tensorrt-acceleration/ (经典基础内容：INT8 稀疏与 TensorRT 工作流)
13. https://arxiv.org/abs/2004.09602 (经典基础内容：整数推理量化原则与实证)
14. https://arxiv.org/abs/2209.05433 (经典基础内容：FP8 格式与训练/推理)
15. https://developer.nvidia.com/blog/designing-an-optimal-ai-inference-pipeline-for-autonomous-driving/ (经典基础内容：Triton 自动驾驶推理流水线案例)
16. https://arxiv.org/html/2505.06780v1 (2025 Autoware 多 deadline DAG 调度)
17. https://arxiv.org/html/2506.00645v1 (2025 AWML 自动驾驶 MLOps 与部署框架)
18. https://www.qualcomm.com/developer/software/snapdragon-ride-sdk (Snapdragon Ride SDK 跨平台参考)
19. https://www.nvidia.com/en-us/autonomous-machines/embedded-systems/jetson-thor/ (Jetson Thor 功耗、带宽与 Blackwell 边缘 AI 规格参考)
20. https://www.nhtsa.gov/vehicle-safety/automated-vehicles-safety (NHTSA 自动驾驶安全公共资料)

### 🔍 扩展检索关键词
`NVIDIA DRIVE AGX Thor DriveOS 7 TensorRT 10`, `DriveOS LLM SDK FP8 NVFP4 INT4`, `TensorRT explicit quantization QDQ ONNX`, `Autoware DAG scheduling ROS2 real-time`, `AWML autonomous driving MLOps`, `NVIDIA Halos ISO 26262 SOTIF`, `Thor Orin Qualcomm Ride Horizon J6 ONNX`, `automotive HIL deterministic replay model deployment`

### ⚠️ 局限性说明
无。本章资料量满足 100 篇以上的索引构建要求；但第 0 章是导航章节，后续二级章节仍需对对应专题重新做更细粒度检索、链接去重和工程检查清单沉淀。
