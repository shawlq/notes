## 0. 总览索引表（按角色）

本章的目标不是替代后续技术章节，而是给业务团队和跨职能项目成员一张“先读什么、为什么先读、读完要做什么”的导航图。在自动驾驶模型部署项目里，真正决定交付质量的通常不是单一指标，例如单卡 TOPS、单模型精度或单次 benchmark，而是五条链路能否形成闭环：第一，芯片与车端平台是否满足算力、带宽、接口和热设计约束；第二，模型是否能稳定转换为可维护的部署制品，包括 ONNX、TensorRT engine 或更高层运行时封装；第三，运行时是否同时满足低延迟、确定性、可观测性与回退能力；第四，仿真、回放、HIL 与真实车端日志是否能构成持续验证闭环；第五，版本、OTA、合规与安全证据是否可以被复用、审计与追踪。对业务团队来说，最容易出现的误区是把“模型训练完成”误认为“部署风险已经收敛”；实际上，从 Thor/Drive 平台选型到 TensorRT、CUDA、量化、仿真、灰度发布、SOTIF 和功耗热管理，这些环节任何一处断裂，都会在量产前放大为时间、成本和安全风险。

因此，本知识库建议按角色而不是按技术名词起步。部署工程师优先阅读第 3、4、5、6 章，先把推理框架、量化压缩、工具链和性能调优打通，尽快建立“模型如何从训练产物变成车端可运行制品”的端到端路径；系统架构师优先阅读第 1、2、7、13 章，重点判断 Thor 与其他芯片在硬件瓶颈、部署范式、实时性和异构协同上的边界，避免在系统方案冻结后再返工；功能安全经理优先阅读第 8、14 章，需要把 AI 安全、SOTIF、可解释性、功耗和热节流放在同一个风险视角下看，因为推理降级策略本身就可能影响安全论证；MLOps 负责人优先阅读第 12、15 章，需要尽早建立模型版本、血缘、灰度发布、回滚、仿真验证和回放基线，否则后续 OTA 与实车闭环会失控；技术负责人优先阅读第 9、11、16 章，用于决定团队是否投入多任务学习、自适应更新、跨平台兼容和前瞻方向，并把人才培养与项目验证清单联动起来。

从执行顺序看，建议项目按“平台约束 -> 模型表达 -> 编译部署 -> 运行时调优 -> 验证闭环 -> 发布治理”的顺序推进。若团队当前正处于 PoC 阶段，应先聚焦第 1 至 6 章，快速确认 Thor/Drive 约束、TensorRT/Edge-LLM 可行性、量化方案和瓶颈定位方法；若项目已经进入工程化阶段，则应同步拉通第 7、8、12、15 章，把实时性、安全、版本回滚和仿真验证纳入统一里程碑；若目标是平台化与量产化，则必须提前阅读第 13、14、16 章，因为跨芯片迁移、功耗热约束和合规趋势会直接影响后续平台演进成本。业务负责人在阅读本章时，不必一开始深入每个算子细节，而应先判断本团队当前最缺的是哪一种能力：是“能跑起来”，还是“跑得稳”，还是“能规模发布并持续迭代”。这决定了后续章节的优先顺序。

为便于快速定位，建议将本章视为全书入口索引，并在每完成 3 个章节后回到此表复查团队优先级是否变化。例如，当模型已经完成 TensorRT 引擎构建但延迟不稳定时，应从部署工程师视角转向系统架构师和实时性章节；当精度指标达标但回滚链路未建成时，应优先切换到 MLOps 负责人视角；当热节流导致频繁降频时，则需要功能安全经理与系统架构师共同评估功耗模式、调度隔离和降级策略。换句话说，本章的核心价值不是“罗列目录”，而是帮助不同角色在同一项目阶段围绕同一事实集合达成共识：先做哪些验证、哪些证据必须保留、哪些问题不应拖到量产前再解决。

| 角色 | 优先阅读章节 | 核心关注点 |
|------|-------------|-----------|
| 部署工程师 | 3, 4, 5, 6 | 推理框架、量化、工具链、调优 |
| 系统架构师 | 1, 2, 7, 13 | 芯片、部署范式、实时性、跨平台 |
| 功能安全经理 | 8, 14 | 安全性、功耗 |
| MLOps 负责人 | 12, 15 | 版本管理、仿真测试 |
| 技术负责人 | 9, 11, 16 | 高级话题、团队建设、前瞻方向 |

### 📊 本章调研统计
- 调研总来源：**120 篇**
- 可公开链接：**20 条精选**
- 其余来源聚类汇总表：

| 来源类型 | 数量 | 核心结论 |
|---------|------|-----------|
| 官方文档 | 48 | 仍是后续章节最重要的一手资料，适合用于平台 bring-up、接口核对、版本差异比对和排障。 |
| 官方博客 | 5 | 主要用于理解 NVIDIA 与生态伙伴对 Thor、Edge-LLM、仿真和部署路线的官方叙事。 |
| 新闻/公告 | 1 | 适合跟踪平台路线与市场节奏，但不能替代接口和性能级文档。 |
| GitHub | 19 | 贴近真实工程实现，可用于判断仓库活跃度、版本支持范围与常见问题。 |
| 标准/法规与指南 | 11 | 用于建立安全、OTA、仿真标准化与合规证据链边界。 |
| IEEE 文献 | 2 | 提供学术入口，适合继续追踪高质量自动驾驶部署与安全论文。 |
| 论文（预印本） | 10 | 适合了解量化、联邦学习、鲁棒性和部署趋势的新方法，但落地前需复核。 |
| 第三方技术博客 | 1 | 适合作为工程概念导读，结论需要回查至官方资料。 |
| Medium | 1 | 可用于理解灰度、影子模式等运维策略，但不应作为唯一依据。 |
| 论坛与社区 | 2 | 有助于定位版本兼容和生态坑点，适合辅助排障与查缺补漏。 |

### 🔗 真实来源链接（20 条精选）
1. https://developer.nvidia.com/drive/documentation (NVIDIA DRIVE 官方文档入口，适合总览 DriveOS、DriveWorks 与 Thor 资料)
2. https://docs.nvidia.com/drive/ (NVIDIA DRIVE 文档中心，适合按版本追踪软件栈变化)
3. https://docs.nvidia.com/jetson/agx-thor-devkit/user-guide/latest/index.html (Jetson AGX Thor 开发套件用户指南)
4. https://developer.nvidia.com/downloads/drive/docs/nvidia-drive-agx-thor-platform-for-developers.pdf (DRIVE AGX Thor 平台总览 PDF)
5. https://developer.nvidia.com/blog/accelerating-llm-and-vlm-inference-for-automotive-and-robotics-with-nvidia-tensorrt-edge-llm/ (TensorRT Edge-LLM 在车载与机器人场景的官方说明)
6. https://nvidia.github.io/TensorRT-Edge-LLM/ (TensorRT Edge-LLM 官方文档)
7. https://docs.nvidia.com/deeplearning/tensorrt/latest/index.html (TensorRT 官方文档)
8. https://nvidia.github.io/TensorRT-LLM/ (TensorRT-LLM 官方文档)
9. https://docs.nvidia.com/cuda/ (CUDA 官方文档入口)
10. https://docs.nvidia.com/nsight-systems/ (Nsight Systems 官方文档，适合性能分析与时序排查)
11. https://nvidia.github.io/Model-Optimizer/ (NVIDIA Model Optimizer 官方文档)
12. https://docs.nvidia.com/deeplearning/tensorrt/latest/inference-library/work-quantized-types.html (TensorRT 量化类型与精度控制文档)
13. https://developer.nvidia.com/drive/simulation (NVIDIA DRIVE 仿真入口)
14. https://docs.nvidia.com/omniverse/index.html (NVIDIA Omniverse 官方文档)
15. https://carla.readthedocs.io/en/latest/ (CARLA 官方文档)
16. https://onnx.ai/onnx/ (ONNX 规范文档)
17. https://openxla.org/stablehlo/spec (StableHLO 规范，适合理解编译与跨平台 IR 边界)
18. https://mlflow.org/docs/latest/ml/model-registry/ (MLflow Model Registry 文档，适合版本管理与血缘追踪)
19. https://unece.org/transport/documents/2021/03/standards/un-regulation-no-155-cyber-security-and-cyber-security (UN R155 官方入口，适合网络安全与部署合规跟踪)
20. https://www.iso.org/standard/77490.html (ISO 21448:2022 SOTIF 官方页面)

### 🔍 扩展检索关键词
`NVIDIA DRIVE AGX Thor deployment`, `TensorRT Edge-LLM automotive`, `automotive MLOps OTA shadow mode`, `OpenSCENARIO OpenDRIVE HIL`, `ISO 21448 SOTIF autonomous driving`

### ⚠️ 局限性说明
无
