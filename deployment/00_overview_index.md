## 0. 总览索引表（按角色）

本章不是“技术细节展开章”，而是整本知识库的导航层。对业务团队而言，自动驾驶模型部署的难点通常不在于单一模型能否跑起来，而在于如何在正确的硬件、正确的软件栈、正确的安全边界和正确的交付流程下，把模型稳定地跑进车端系统。因此，阅读顺序必须围绕角色职责来设计：部署工程师优先看推理框架、量化、工具链和调优，因为他们最直接面对 TensorRT、CUDA、内存、图优化和版本兼容问题；系统架构师优先看芯片架构、部署范式、实时性和跨平台迁移，因为他们需要决定是端到端还是两段式、是单 SoC 还是异构协同、是以 ONNX 为中间表示还是以厂商工具链为主；功能安全经理需要优先阅读安全和功耗章节，因为量产交付最终会被 ISO 26262、SOTIF、UN R155/R156、热设计和降级策略约束；MLOps 负责人则更关心版本、OTA、灰度、回滚、仿真验证与回放闭环；技术负责人需要通过高级系统话题、团队能力建设和趋势章节，判断当前路线是否可持续、是否具备演进空间。

从落地顺序看，建议不要把知识库当成“从第 1 章按页往后读”的教材，而应当把它当成部署决策树来使用：先确认目标车型和芯片边界，再确认模型范式与推理框架，再确认量化与压缩方案，再进入调优、实时性、安全性和验证闭环。对于 Thor 相关项目，推荐的最短路径是“1 → 3 → 4 → 5 → 6 → 7 → 8 → 15”；对于需要跨 Orin、Thor 或其他芯片迁移的项目，建议额外提前阅读第 13 章；对于车端持续运营压力较大的项目，应把第 12 章和第 15 章前置，而不是留到交付后补课。业务团队在使用本知识库时，也应把“关注点”翻译成可执行问题，例如“本模型是否需要显式 Q/DQ 导出”“动态 shape 是否会导致重编译”“回滚是否支持热切换”“温度触发降频后是否存在可验证的降级路径”。这样，知识库才能直接服务于项目评审、方案选型和发布决策，而不仅是资料堆积。

下面的总览索引表给出推荐阅读入口。表中的章节编号使用本知识库的正式编号；“核心关注点”强调每个角色在评审方案、制定里程碑和复盘风险时最应该优先看的问题；如果一个团队成员承担多重角色，建议按“架构约束 → 部署实现 → 安全合规 → 运营验证”的顺序串读，而不是并行跳读。

| 角色 | 优先阅读章节 | 核心关注点 |
|------|-------------|-----------|
| 部署工程师 | 3, 4, 5, 6 | 推理框架、量化、工具链、调优 |
| 系统架构师 | 1, 2, 7, 13 | 芯片、部署范式、实时性、跨平台 |
| 功能安全经理 | 8, 14 | 安全性、功耗 |
| MLOps 负责人 | 12, 15 | 版本管理、仿真测试 |
| 技术负责人 | 9, 11, 16 | 高级话题、团队建设、前瞻方向 |

进一步说，部署工程师进入第 3 章时，应带着“最终运行时是谁、图在哪里切分、缓存怎么复用、插件如何管理”的问题阅读；系统架构师进入第 1、2、7、13 章时，应重点判断算力预算、数据路径、隔离机制和迁移代价；功能安全经理在第 8、14 章里要把 AI 风险、网络安全、功耗和热节流看作同一交付问题，而不是拆开的认证问题；MLOps 负责人需要把 OTA、灰度、回滚、影子模式和仿真验证纳入同一发布治理闭环；技术负责人则需要用第 9、11、16 章回答“团队现在能不能做”“做完能不能持续维护”“未来两代硬件与标准变化会不会推翻当前方案”。如果业务方希望快速发起一个部署评审会议，可以直接把本章表格作为会议前置阅读清单：先按角色分配章节，再要求每个角色给出 3 个项目级风险和 3 个可执行建议，这样知识库就会从文档沉淀转化为决策机制。

### 📊 本章调研统计
- 调研总来源：**126 篇**
- 可公开链接：**20 条精选**
- 其余来源聚类汇总表：

| 来源类型 | 数量 | 核心结论 |
|---------|------|-----------|
| NVIDIA 官方文档、博客与开发者资料 | 26 | Thor/DriveOS/TensorRT/DriveWorks 已形成相对完整的车端部署主线，适合构建主干知识框架。 |
| 开源框架与编译器官方文档 | 23 | ONNX Runtime、OpenXLA、TVM、IREE、Torch-TensorRT 分别覆盖兼容层、编译层与部署层，适合作为跨平台对照系。 |
| GitHub 工程仓库与示例项目 | 17 | 工程仓库最能反映插件、模型导出、缓存、仿真与 HIL 接线等实际落地模式。 |
| 标准、法规与行业白皮书 | 20 | ISO 26262、ISO/SAE 21434、SOTIF、UN R155/R156 决定了量产部署不能只谈性能，必须同步谈安全治理与升级治理。 |
| 论文、综述与技术课程 | 20 | 端到端、多任务、可解释性、鲁棒性和实时性问题已经从研究议题转为部署选型输入，适合指导技术负责人判断演进方向。 |

### 🔗 真实来源链接（20 条精选）
1. https://developer.nvidia.com/drive/documentation （NVIDIA DRIVE 文档总入口，覆盖 Thor、DriveOS、DriveWorks、TensorRT for DRIVE）
2. https://developer.nvidia.com/blog/accelerate-autonomous-vehicle-development-with-the-nvidia-drive-agx-thor-developer-kit/ （2025，NVIDIA 官方 Thor 开发者套件与 DriveOS 7 概览）
3. https://developer.download.nvidia.com/drive/docs/nvidia-drive-agx-thor-platform-for-developers.pdf （Thor 平台产品概览 PDF）
4. https://developer.nvidia.com/docs/drive/drive-os/7.0.3/public/drive-os-linux-sdk/index.html （DriveOS 7.0.3 Linux SDK 开发指南）
5. https://docs.nvidia.com/deeplearning/tensorrt/latest/index.html （TensorRT 官方文档）
6. https://docs.nvidia.com/deeplearning/tensorrt/latest/inference-library/work-quantized-types.html （TensorRT 量化、Q/DQ 与精度控制文档）
7. https://docs.pytorch.org/TensorRT/ （Torch-TensorRT 官方文档）
8. https://onnxruntime.ai/docs/execution-providers/TensorRT-ExecutionProvider.html （ONNX Runtime TensorRT Execution Provider 配置文档）
9. https://openxla.org/stablehlo （StableHLO 官方文档，跨框架/编译器兼容层）
10. https://tvm.apache.org/docs/ （Apache TVM 官方文档）
11. https://docs.nvidia.com/nsight-systems/UserGuide/index.html （Nsight Systems 官方用户指南）
12. https://docs.nvidia.com/nsight-compute/ （Nsight Compute 官方文档）
13. https://mlflow.org/docs/latest/ml/model-registry/ （MLflow 模型注册官方文档）
14. https://docs.wandb.ai/models/registry （Weights & Biases Registry 官方文档）
15. https://carla.readthedocs.io/ （CARLA Simulator 官方文档）
16. https://docs.nvidia.com/learning/physical-ai/getting-started-with-isaac-sim/latest/leveraging-ros-2-and-hil-in-isaac-sim/01-hardware-in-the-loop-hil-fundamentals.html （Isaac Sim HIL 官方入门文档）
17. https://www.nuplan.org/nuplan （nuPlan 规划基准官方主页）
18. https://www.asam.net/standards/detail/opendrive/ （ASAM OpenDRIVE 官方标准入口）
19. https://www.asam.net/fileadmin/Standards/OpenSCENARIO/ASAM_OpenSCENARIO_BS-1-2_User-Guide_V1-2-0.html （ASAM OpenSCENARIO 用户指南）
20. https://unece.org/transport/documents/2021/03/standards/un-regulation-no-155-cyber-security-and-cyber-security （UNECE UN R155 官方入口）

### 🔍 扩展检索关键词
`NVIDIA DRIVE AGX Thor deployment`, `autonomous driving inference stack`, `vehicle AI safety and OTA deployment`

### ⚠️ 局限性说明
无

---
