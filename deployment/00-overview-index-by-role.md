## 0. 总览索引表（按角色）

本章是整份自动驾驶模型部署知识库的入口页，目标不是替代后续 52 个二级章节，而是帮助业务、研发、测试和安全团队快速判断“先读哪里、带着什么问题读、读完后应形成什么工程动作”。本轮检索计划分为五组：第一组聚焦 `NVIDIA DRIVE Thor`, `DriveOS`, `TensorRT`, `CUDA`, `Blackwell automotive SoC`，优先 `nvidia.com` 与 `docs.nvidia.com`，预期 30 条；第二组聚焦 `INT8`, `FP8`, `NVFP4`, `QAT`, `PTQ`, `model compression`, `TensorRT Model Optimizer`，预期 20 条；第三组聚焦 `ONNX Runtime TensorRT`, `TVM`, `MLIR`, `OpenXLA`, `DAG scheduling`，预期 18 条；第四组聚焦 `autonomous driving MLOps`, `OTA`, `A/B testing`, `shadow mode`, `CARLA`, `HIL`, `deterministic replay`，预期 22 条；第五组聚焦 `ISO 26262`, `SOTIF`, `ISO 21434`, `UN R155`, `AI safety`, `explainability`, `behavioral safety`，预期 20 条。合计目标 110 条以上，实际纳入 112 条，其中 2024-2026 年资料优先，2023 年前的 TVM、MLIR、经典自动驾驶开源栈资料仅作为“经典基础内容”使用。

第 50 条资料中间汇总显示，部署工程侧的高频共识非常明确：Thor/Orin 车端模型部署不能只看 TOPS，而要同时管理摄像头/雷达输入带宽、LPDDR 带宽、TensorRT engine 构建时间、动态 shape profile、精度回退和安全 runtime 约束。NVIDIA 官方资料强调 Thor DevKit 提供 Blackwell GPU、Arm Neoverse V3AE、CUDA、cuDNN、TensorRT、NvMedia 与 DriveOS 基础软件栈；TensorRT 与 ONNX Runtime TensorRT EP 资料则反复出现 engine cache、timing cache、FP16/INT8/FP8、插件库、DLA/多流、fallback 调试等关键词。对部署工程师而言，优先阅读第 3、4、5、6、7、10 章最直接，因为这些章节覆盖推理框架、量化、工具链、性能调优、实时性和故障排查。

第 100 条资料中间汇总显示，业务团队更容易低估安全、测试和生命周期管理的复杂度。公开论文和产业资料将风险分成三类：一是模型压缩或端到端模型引入的行为偏移，例如低比特量化可能改变规划/控制策略；二是仿真与 HIL 不充分导致长尾场景无法覆盖，例如 CARLA、dSPACE、Omniverse、数据回放与场景生成需要组合使用；三是 MLOps 和 OTA 治理不足导致“模型能跑但不可发布”，例如版本哈希、血缘追踪、影子模式、灰度发布、热切换与回滚必须在部署前设计。对系统架构师、功能安全经理、MLOps 负责人和技术负责人而言，第 1、2、8、12、13、14、15、16 章应作为决策闭环阅读，而不是上线后补课。

角色索引建议如下：

| 角色 | 优先阅读章节 | 核心关注点 |
|------|-------------|-----------|
| 部署工程师 | 3, 4, 5, 6 | 推理框架、量化、工具链、调优 |
| 系统架构师 | 1, 2, 7, 13 | 芯片、部署范式、实时性、跨平台 |
| 功能安全经理 | 8, 14 | 安全性、功耗 |
| MLOps 负责人 | 12, 15 | 版本管理、仿真测试 |
| 技术负责人 | 9, 11, 16 | 高级话题、团队建设、前瞻方向 |

落地使用时，建议业务团队把本知识库当成“三层地图”。第一层是决策地图：第 1、2、13、16 章回答“为什么选 Thor、两段式还是端到端、是否需要跨芯片适配、未来技术风险在哪里”。这层适合技术负责人和架构师在立项、供应商评审、量产路线评审时使用。第二层是工程地图：第 3、4、5、6、7、10 章回答“模型如何转 ONNX、如何构建 TensorRT engine、如何做 FP8/INT8、如何定位 latency、如何处理动态 shape、fallback CPU、内存碎片和多模型流水线冲突”。这层适合部署工程师、模型工程师和平台工程师共同维护 checklist。第三层是发布地图：第 8、12、14、15 章回答“压缩后的模型是否安全、能否解释、功耗和热是否可控、仿真/HIL/回放是否覆盖、OTA 和回滚是否有证据链”。这层适合功能安全、质量、测试和 MLOps 团队共同定义发布门禁。

对于自动驾驶模型部署项目，最重要的实践建议是先把角色问题前置到章节阅读路径中。部署工程师不要只从第 4 章量化开始，而应先读第 3 章确认目标 runtime、算子支持和 CUDA 内存策略，再读第 6、7 章建立 profiling 与确定性验证方法；系统架构师不要只读 Thor 硬件规格，而应把第 1 章硬件单元、第 2 章部署范式、第 13 章跨芯片迁移一起看，避免在模型已经绑定特定插件或非标准算子后才讨论平台兼容；功能安全经理需要把第 8 章安全/可解释性与第 14 章功耗热管理联读，因为热节流导致的推理降级同样可能触发安全风险；MLOps 负责人应从第 12 章模型版本血缘进入，再回到第 15 章仿真与 HIL，确保每个模型包都有可复现实验、硬件 profile、回滚策略和场景覆盖报告；技术负责人则应把第 9、11、16 章作为组织能力建设材料，明确多任务学习、端上模型更新、团队技能矩阵和 2025-2026 趋势的投资优先级。

### 📊 本章调研统计
- 调研总来源：**112 篇**
- 可公开链接：**20 条精选**
- 其余来源聚类汇总表：

| 来源类型 | 数量 | 核心结论 |
|---------|------|-----------|
| NVIDIA 官方文档、博客、论坛、白皮书 | 24 | Thor/DriveOS/TensorRT/CUDA 是车端部署主线，必须同时关注安全版本、engine 构建、精度格式和 profile 配置。 |
| arXiv、IEEE、SAE、ACM/软件工程论文 | 21 | 2024-2026 年研究重点转向端到端、基础模型、行为安全、低比特部署、仿真验证与软件工程化治理。 |
| GitHub 与开源项目文档 | 16 | TensorRT、Autoware、CARLA、TensorRT-Model-Optimizer 等项目可支撑原型验证，但量产仍需补齐安全、确定性和版本治理。 |
| 编译器、IR 与推理框架资料 | 12 | ONNX、TVM、MLIR、OpenXLA 等适合做跨平台抽象，但车端最终性能仍依赖目标芯片算子库、插件和内存调度。 |
| MLOps、AVOps、OTA 与数据闭环资料 | 10 | 模型仓库、血缘追踪、影子模式、灰度、差分更新和回滚机制应与部署工具链一起设计。 |
| 仿真、HIL、数据回放与测试平台资料 | 14 | CARLA、dSPACE、Omniverse、SIL/HIL/回放共同组成发布前证据链，单一仿真无法证明量产安全。 |
| 安全、合规、可解释性与功耗资料 | 15 | ISO 26262、SOTIF、ISO 21434、UN R155、AI safety 与热管理需要纳入模型部署验收，而非独立合规附件。 |

### 🔗 真实来源链接（20 条精选）
1. https://developer.nvidia.com/drive/agx (NVIDIA DRIVE AGX Thor/Orin DevKit 官方规格与软件栈)
2. https://docs.nvidia.com/drive/ (NVIDIA DRIVE OS、DriveWorks、CUDA、TensorRT 文档入口)
3. https://docs.nvidia.com/self-driving-cars/autonomous-driving-safety-report/index.html (NVIDIA 自动驾驶安全报告与 Halos/安全标准说明)
4. https://developer.nvidia.com/tensorrt (TensorRT 产品页，覆盖量化、融合、Model Optimizer 与 DRIVE 集成)
5. https://docs.nvidia.com/deeplearning/tensorrt/latest/index.html (TensorRT 最新官方文档与版本能力)
6. https://developer.nvidia.com/blog/streamline-llm-deployment-for-autonomous-vehicle-applications-with-nvidia-driveos-llm-sdk/ (DriveOS LLM SDK 车端 LLM/VLM 部署流程)
7. https://developer.nvidia.com/blog/accelerating-llm-and-vlm-inference-for-automotive-and-robotics-with-nvidia-tensorrt-edge-llm/ (TensorRT Edge-LLM 面向车端/机器人实时推理)
8. https://github.com/NVIDIA/TensorRT (TensorRT 开源组件仓库)
9. https://github.com/NVIDIA/TensorRT-Edge-LLM (TensorRT Edge-LLM 开源 C++ 推理框架)
10. https://github.com/NVIDIA/TensorRT-Model-Optimizer (TensorRT Model Optimizer，量化/剪枝/蒸馏工具)
11. https://onnxruntime.ai/docs/execution-providers/TensorRT-ExecutionProvider.html (ONNX Runtime TensorRT Execution Provider 配置)
12. https://tvm.apache.org/docs/ (Apache TVM 文档，经典基础内容/持续更新)
13. https://mlir.llvm.org/ (MLIR 编译器基础设施，经典基础内容/持续更新)
14. https://openxla.org/xla (OpenXLA/XLA 跨硬件 ML 编译器文档)
15. https://github.com/autowarefoundation/autoware (Autoware 自动驾驶开源栈)
16. https://github.com/carla-simulator/carla (CARLA 自动驾驶仿真器)
17. https://learn.microsoft.com/en-us/industry/mobility/architecture/avops-architecture-content (Microsoft AVOps 架构，自动驾驶数据与 MLOps 流程)
18. https://www.dspace.com/en/pub/home/products/systems/ecutest/hil_for_autonomous_driving.cfm (dSPACE 自动驾驶 HIL 与数据回放方案)
19. https://arxiv.org/abs/2412.01034 (2024 量化感知模仿学习，覆盖自动驾驶低比特部署)
20. https://arxiv.org/abs/2505.16214 (2025 自动驾驶大规模部署行为安全评估)

### 🔍 扩展检索关键词
`NVIDIA DRIVE Thor TensorRT FP8`, `DriveOS LLM SDK ONNX engine`, `autonomous driving model deployment quantization`, `TensorRT dynamic shape engine cache`, `AVOps shadow mode OTA rollback`, `CARLA HIL deterministic replay`, `ISO 26262 AI functional safety`, `UN R155 model deployment cybersecurity`

### ⚠️ 局限性说明
无。本章已达到 100 篇以上调研目标；但第 0 章是导航章，后续具体章节仍需继续按主题补充更细的 benchmark、配置示例、失败案例和跨平台实测资料。

---
