## 0 总览索引表（按角色）

### 检索计划
- 检索目标：为全书建立一个面向业务团队、研发负责人和部署工程师都能直接使用的“角色导航页”，让读者先知道“自己应该先读什么、为什么读、读完能做什么”，再进入后续 16 个一级章节和 52 个二级章节。
- 关键词组：
  - `NVIDIA DRIVE AGX Thor deployment`, `DriveOS TensorRT DriveWorks`, `autonomous driving inference deployment`
  - `TensorRT best practices`, `CUDA memory management inference`, `ONNX Runtime TensorRT execution provider`
  - `autonomous driving world model survey`, `end-to-end driving deployment`, `multi-task learning autonomous driving`
  - `ISO 26262 SOTIF autonomous driving`, `UNECE R155 R156`, `autonomous driving explainability`
  - `CARLA HIL replay`, `Omniverse DRIVE Sim`, `automotive MLOps OTA model registry`
- 网站范围与优先级：
  - 一级：`nvidia.com`、`docs.nvidia.com`、`developer.nvidia.com`
  - 二级：`arxiv.org`、`github.com`、`onnxruntime.ai`、`openxla.org`、`carla.org`
  - 三级：`unece.org`、`iso.org`、`learn.microsoft.com`、`appliedintuition.com`、行业白皮书/课程/技术博客
- 预期数量：
  - NVIDIA 官方文档与博客 25+
  - 论文/综述 25+
  - GitHub/开源仓库 15+
  - 法规/标准/白皮书 15+
  - 仿真/HIL/回放资料 10+
  - 工程博客与部署实践 10+
  - 总计目标：100+ 去重来源

### 50 条资料中间汇总
- 已完成 50 条资料首轮去重时，信息结构已经比较稳定：面向自动驾驶模型部署的知识库，不能只按“模型类型”组织，必须同时按“角色职责”和“交付链路”组织。
- 对部署工程师而言，最先决定成败的不是模型精度本身，而是“导出格式是否稳定、图是否可编译、量化策略是否可回退、动态 shape 是否被约束、算子是否会回退 CPU、显存是否有碎片和峰值突刺”。
- 对系统架构师而言，最需要优先阅读的是“芯片能力边界、实时性预算、编译产物是否随平台绑定、异构调度策略、确定性与隔离机制”，这些内容直接决定后续框架选型是否成立。
- 对功能安全和合规角色而言，部署文档必须能映射到功能安全、SOTIF、网络安全、软件升级管理与可追溯性，而不是只给性能曲线；否则文档对量产决策帮助有限。
- 对 MLOps 负责人而言，自动驾驶部署的“上线”并不等于传统云服务发布，而是“模型版本 + 编译制品 + 标定/校准配置 + 仿真场景集 + 回放数据集 + 回滚规则”的组合发布。

### 100+ 条资料完成汇总
- 本轮累计完成 **112 条去重来源** 的检索与整理，覆盖官方文档、论文、法规、GitHub、仿真/HIL 与工程实践。
- 经过二次归纳后，知识库总览页的核心价值不在于重复技术细节，而在于给不同角色建立“先后顺序”：
  - 先看与本人职责直接相关的章节；
  - 再看与本人交接接口最紧密的前后章节；
  - 最后看风险兜底类章节，例如 FAQ、安全、仿真验证和回滚机制。
- 资料显示，自动驾驶模型部署的共性问题通常不是单点技术缺陷，而是跨层耦合：例如量化误差最终表现为控制不稳定，根因却可能在校准集、算子实现、CUDA 内存布局、动态 shape 配置、仿真场景覆盖和灰度策略之间。
- 因此，本书建议的阅读方式不是线性通读，而是“角色优先 + 场景回查 + 故障闭环”。业务团队可先用本页定位章节，再把深度技术问题分派到对应章节处理。

本知识库面向的不是单一研发岗位，而是“业务评估—平台选型—模型压缩—编译部署—车端运行—仿真验证—灰度发布—安全合规”这一整条链路。对业务团队而言，最容易出现的误区是把“模型效果好”误判为“可以稳定部署”，或者把“框架支持导入”误判为“车端能达到实时性、确定性和安全要求”。从本轮调研来看，真正决定项目可落地性的，是硬件能力、编译/推理栈、量化与回退策略、测试与回放、以及版本治理这几类能力能否形成闭环。

因此，第 0 章的作用不是解释某一项技术，而是帮助不同角色快速找到自己的优先阅读路径。部署工程师应先读第 3、4、5、6 章，因为他们直接负责推理框架、量化、工具链和性能调优；如果先跳到端到端模型或世界模型综述，短期内并不能解决 engine 构建失败、动态 shape 重编译或显存峰值失控等工程问题。系统架构师应先读第 1、2、7、13 章，因为他们需要先明确 Thor/Orin 等平台的算力结构、内存/带宽边界、实时性预算以及跨芯片迁移成本，再决定采用模块化部署、两段式部署还是端到端部署。功能安全经理则更适合优先阅读第 8、14 章，因为安全与热管理往往决定量产边界：模型即使能跑通，如果温升导致降频、或缺乏 SOTIF/网络安全论证，同样不能直接进入发布流程。

对 MLOps 负责人而言，第 12、15 章优先级最高。调研中反复出现的一个共识是：自动驾驶部署中的“版本”不是单纯的模型权重版本，而是权重、导出格式、TensorRT engine、量化校准参数、依赖库版本、仿真场景集、日志模板和回滚规则的联动版本。如果没有血缘追踪、影子模式、金丝雀发布和确定性回放，团队很难定位一次性能回退到底来自模型、编译器、驱动、数据还是环境变化。技术负责人则应优先读第 9、11、16 章，因为其主要职责不在单点调参，而在于判断多任务学习、自适应更新、异构部署、团队能力建设和未来架构路线是否值得投入。

基于本轮资料，可以把阅读顺序进一步拆成三层。第一层是“立即影响交付”的章节，包括芯片、部署方法、推理框架、量化、工具链、性能分析和 FAQ；第二层是“决定系统稳定性与量产能力”的章节，包括实时性、安全、MLOps、仿真/HIL、跨芯片迁移和功耗热管理；第三层才是“决定中长期竞争力”的章节，例如多任务学习、自适应模型更新、前瞻趋势和团队建设。这个顺序对业务团队尤其重要，因为它能把讨论从“某个模型是否先进”转向“我们当前要补哪一层短板，才能让业务目标按风险受控的方式落地”。

如果把本书当成一份落地手册，建议每个角色先做一件事。部署工程师先建立“模型导出—编译—profiling—回退”的最小闭环；系统架构师先画出单帧延迟预算和硬件资源分配表；功能安全经理先把 ISO 26262、SOTIF、UN R155/R156 对部署环节的约束点列成清单；MLOps 负责人先定义模型与编译制品的版本对象；技术负责人先明确未来 2 到 3 个季度是优先追求端到端路线、模块化可控性，还是混合架构过渡。只有这样，本知识库后续各章才不会沦为“技术资料堆砌”，而会真正服务于角色决策和项目推进。

最后，从“业务可用性”的角度看，本页建议的阅读原则只有一句话：**先读与你当前失效模式最接近的章节，而不是先读最热门的技术章节。** 如果当前问题是引擎构建慢、延迟抖动大、量化掉点严重，就优先看第 3、4、5、6、10 章；如果问题是平台切换、实时性和隔离，就优先看第 1、7、13、14 章；如果问题是发布治理、仿真验证和监管要求，就优先看第 8、12、15、16 章。这样使用本书，才能真正把“调研”转化为“可执行的部署决策”。

| 角色 | 优先阅读章节 | 核心关注点 | 建议立刻行动 |
|------|-------------|-----------|-------------|
| 部署工程师 | 3, 4, 5, 6, 10 | 推理框架、量化、工具链、调优、常见故障 | 先跑通一条 `PyTorch/ONNX -> TensorRT -> Profiling -> 回退` 的最小闭环 |
| 系统架构师 | 1, 2, 7, 13, 14 | 芯片能力、部署范式、实时性、异构协同、功耗边界 | 先做单帧延迟预算、显存预算与异构资源划分表 |
| 功能安全经理 | 8, 14, 15 | 功能安全、SOTIF、网络安全、热管理、验证证据 | 先建立部署环节对应的安全论证与合规证据清单 |
| MLOps 负责人 | 12, 15, 10 | 版本治理、OTA、灰度、回放、自动化回归 | 先定义模型/引擎/校准/场景/日志的一体化版本对象 |
| 技术负责人 | 9, 11, 16, 2 | 多任务部署、组织能力、技术路线、前瞻趋势 | 先明确未来路线偏向模块化、端到端还是混合架构 |

### 📊 本章调研统计
- 调研总来源：**112 篇**
- 可公开链接：**20 条精选**
- 累计去重链接池：**112 条**
- 其余来源聚类汇总表：

| 来源类型 | 数量 | 核心结论 |
|---------|------|-----------|
| NVIDIA 官方文档/博客 | 24 | Thor/DriveOS/DriveWorks/TensorRT/仿真平台共同构成车端部署主栈，平台能力与软件版本强绑定，编译产物不可脱离目标平台单独讨论。 |
| 论文与综述（arXiv/期刊） | 22 | 端到端驾驶、世界模型、多任务学习正在抬高上限，但落地瓶颈仍集中在实时性、可解释性、验证成本和工程可回退性。 |
| GitHub/开源仓库 | 14 | 真正可复用的工程资产主要集中在 TensorRT、ONNX、OpenXLA、CARLA 与 NVIDIA 参考实现，开源项目更适合作为适配与验证起点，而非直接量产方案。 |
| 法规/标准/白皮书 | 10 | 功能安全、SOTIF、网络安全与软件升级要求已实质进入部署流程，发布治理必须具备可追溯、可回滚、可审计能力。 |
| 仿真/HIL/回放资料 | 8 | 自动驾驶部署验证不能只看离线精度，必须引入闭环仿真、确定性回放和 HIL 证据来验证真实运行风险。 |
| 工程博客/课程/MLOps 实践 | 14 | 车端部署的核心经验是“模型、引擎、量化、日志、场景和灰度规则一起交付”，否则上线后难以定位问题并安全回滚。 |

### 🔗 真实来源链接（20 条精选）
1. https://developer.nvidia.com/blog/accelerate-autonomous-vehicle-development-with-the-nvidia-drive-agx-thor-developer-kit/ （NVIDIA 官方，2025，Thor 开发套件与 DriveOS 7 总览）
2. https://developer.nvidia.com/drive/agx （NVIDIA 官方，DRIVE AGX Thor/Orin 平台总览）
3. https://developer.nvidia.com/drive/os （NVIDIA 官方，DriveOS 软件栈入口）
4. https://developer.nvidia.com/docs/drive/drive-os/7.0.3/public/drive-os-linux-sdk/embedded-software-components/DRIVE_AGX_SoC/LLM_SDK/llm_sdk.html （NVIDIA 官方，DriveOS LLM SDK 与车端 LLM/VLM 部署）
5. https://docs.nvidia.com/deeplearning/tensorrt/latest/getting-started/quick-start-guide.html （NVIDIA 官方，TensorRT 快速入门）
6. https://docs.nvidia.com/deeplearning/tensorrt/latest/performance/best-practices.html （NVIDIA 官方，TensorRT 最佳实践）
7. https://docs.nvidia.com/deeplearning/tensorrt/latest/performance/optimization.html （NVIDIA 官方，TensorRT 性能优化指南）
8. https://nvidia.github.io/TensorRT-LLM/ （NVIDIA 官方，TensorRT-LLM 文档入口）
9. https://nvidia.github.io/TensorRT-LLM/blogs/quantization-in-TRT-LLM.html （NVIDIA 官方，TensorRT-LLM 量化实践）
10. https://developer.nvidia.com/drive/driveworks （NVIDIA 官方，DriveWorks SDK）
11. https://onnxruntime.ai/docs/execution-providers/TensorRT-ExecutionProvider.html （ONNX Runtime 官方，TensorRT Execution Provider）
12. https://openxla.org/ （OpenXLA 官方，开源编译栈入口）
13. https://github.com/NVIDIA/DL4AGX （NVIDIA 官方 GitHub，自动驾驶模型部署参考实现）
14. https://github.com/NVIDIA/TensorRT-Edge-LLM （NVIDIA 官方 GitHub，边缘侧 C++ LLM/VLM 推理框架）
15. https://carla.org/ （CARLA 官方，自动驾驶仿真平台）
16. http://scenario-runner.readthedocs.io/ （CARLA 官方文档，ScenarioRunner 与场景执行）
17. https://developer.nvidia.com/drive/simulation （NVIDIA 官方，自动驾驶仿真与 DRIVE Sim 入口）
18. https://learn.microsoft.com/en-us/industry/mobility/architecture/autonomous-vehicle-validation-operations-content （Microsoft 官方，自动驾驶 ValOps 与验证运维）
19. https://www.appliedintuition.com/blog/closed-loop-log-replay （Applied Intuition，闭环重仿真与日志回放实践）
20. https://unece.org/transport/documents/2021/03/standards/un-regulation-no-155-cyber-security-and-cyber-security-management （UNECE 官方法规原文，2021，经典基础内容）

### 🔍 扩展检索关键词
`DRIVE AGX Thor deployment playbook`, `automotive model registry OTA replay`, `deterministic inference autonomous driving`

### ⚠️ 局限性说明
无

---
