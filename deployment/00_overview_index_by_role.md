## 0. 总览索引表（按角色）

这一章不是替代后续 16 个一级章节的技术细节，而是给业务团队、架构团队和部署团队一个“先读哪里、先做什么、先看哪些风险”的导航页。自动驾驶模型部署在 2025-2026 年的复杂度已经明显高于传统视觉模型上线：一方面，NVIDIA DRIVE AGX Thor、Jetson Thor/Orin、TensorRT 10、CUDA Graph、Torch-TensorRT、DriveOS LLM SDK、TensorRT Edge-LLM 等工具链把车端算力利用率推到更高水平；另一方面，端到端规划、多任务联合建模、低比特量化、影子模式、OTA 差分更新、SOTIF/UN R155 合规、仿真与 HIL 回放测试也把“能跑”升级为“可验证、可回滚、可审计、可持续运维”。因此，知识库首页必须解决两个问题：第一，不同角色该按什么顺序阅读，避免把时间花在不相关的章节上；第二，跨角色协作时，哪些章节应作为共同语言，避免部署、算法、安全、MLOps 各自优化却在发布阶段互相牵制。

| 角色 | 优先阅读章节 | 核心关注点 |
|------|-------------|-----------|
| 部署工程师 | 3, 4, 5, 6 | 推理框架、量化、工具链、调优 |
| 系统架构师 | 1, 2, 7, 13 | 芯片、部署范式、实时性、跨平台 |
| 功能安全经理 | 8, 14 | 安全性、功耗 |
| MLOps 负责人 | 12, 15 | 版本管理、仿真测试 |
| 技术负责人 | 9, 11, 16 | 高级话题、团队建设、前瞻方向 |

对部署工程师而言，最有效的阅读路径不是先看模型结构，而是先锁定运行时边界：目标芯片是 Thor 还是 Orin，主推理栈是 TensorRT 还是 ONNX Runtime + TensorRT EP，量化是 PTQ、QAT 还是混合精度，调优目标是单帧延迟、尾延迟还是吞吐。第 3、4、5、6 章组合在一起，构成“从模型导出到线上稳定运行”的主通道。工程上建议先建立一份最小可运行基线：固定输入 shape、固定 batch、固定时钟与功耗模式、固定标定集，再讨论 FP16/INT8/FP8/NVFP4 等压缩策略，否则调优数据会失真。部署工程师还应把第 10 章 FAQ 当作日常排障索引，特别是动态 shape 重编译、算子回退 CPU、图优化失败、显存碎片和实时性不达标这些高频问题。

对系统架构师而言，最关键的不是某个模型快了多少，而是整车计算预算如何被系统性分配。第 1 章帮助回答 Thor 的关键硬件单元、带宽瓶颈、内存层次和可利用的精度格式；第 2 章帮助判断两段式部署与端到端部署在功能边界、可解释性、回退策略和组织协作上的差别；第 7 章聚焦低延迟和确定性，适合拿来定义 P50/P95/P99 延迟目标、资源隔离策略以及多模型流水线的调度原则；第 13 章则直接决定未来是否能把同一套模型资产迁移到 Orin、高通 SA 或地平线 J6。架构师应优先形成三张图：任务拓扑图、算力预算图、故障降级图。没有这三张图，后续的量化、图编译、MLOps 和安全评审都会缺少统一约束。

对功能安全经理而言，第 8 章和第 14 章必须联读。原因是车规部署中的风险从来不只来自“模型答错”，还包括“模型在热节流后答得更慢”“模型 OTA 后缺乏可追踪回退”“量化后在长尾场景出现系统性误差”等系统性问题。第 8 章会覆盖功能安全、SOTIF、对抗攻击、不确定性量化、解释性与安全论证；第 14 章则把功耗、热设计、频率调节、热触发降级与报告模板连接起来。安全经理在项目初期就应推动团队把“推理降级策略”写成可执行规则，例如高温时优先降低哪些辅助任务频率、何时退回保守规划、何时触发人工接管提示。这样安全策略才不会停留在评审材料里。

对 MLOps 负责人而言，第 12 章与第 15 章是发布体系的骨架。模型版本管理不应只保存原始权重，还要保存设备类型、编译参数、量化配置、校准集哈希、运行时版本以及来源血缘；否则同一个模型名在 Thor、Orin 和仿真环境中实际上对应的是不同产物。第 12 章将聚焦 OTA、灰度发布、影子模式、热切换与回滚；第 15 章则回答“上线前如何证明这次发布没有把实时性、安全性和精度一起带崩”。对 MLOps 团队最有价值的实践，是把性能、精度、安全、可回放性统一进回归流水线，而不是只做单一离线精度评估。

对技术负责人而言，第 9、11、16 章负责回答“接下来一年团队该投什么”。第 9 章覆盖多任务学习、自适应更新、兼容集成；第 11 章把学习路径和项目验证清单标准化；第 16 章则用于持续跟踪端上学习、联邦学习、Thor 后续路线、ISO 26262/UN R155 等行业变化。技术负责人不应只关注模型 SOTA，还要判断组织是否具备把新技术放进量产链路的能力，包括算子兼容、验证预算、法规约束和回滚成本。

如果团队刚开始建设自动驾驶部署体系，建议采用“四步法”使用本知识库。第一步，所有角色先看本章，明确自己优先读哪些章节，并确认哪些章节需要跨团队共同评审；第二步，部署工程师和系统架构师联合阅读第 1-7 章，建立统一的芯片、模型、工具链和时延预算假设；第三步，功能安全经理与 MLOps 负责人基于第 8、12、14、15 章补齐验证、回放、灰度和回滚机制；第四步，技术负责人再用第 9、11、16 章决定是否引入端到端、多任务、在线更新或跨芯片迁移等高级主题。这样阅读顺序的好处是：先把发布约束钉牢，再决定模型复杂度，而不是反过来让部署链路为研究原型被动兜底。

从调研结果看，2025-2026 年自动驾驶部署最值得业务团队形成共识的结论有五点：一是 Thor/DriveOS 生态正在把高算力、低比特量化和边缘 LLM 能力带到车端，但引擎构建、精度回退和热管理成本同步上升；二是端到端模型正在走向闭环验证和量产试探期，但两段式架构仍然在可解释性、故障隔离和认证沟通上更稳健；三是部署工具链正在从单一推理引擎扩展为“编译器 + 运行时 + 分析器 + 发布系统”的组合；四是安全与合规要求已经从附属项变成主线约束；五是仿真、HIL、影子模式和回滚机制不再是上线后的补救工具，而是部署设计的一部分。基于这些判断，本知识库后续各章会按“能指导工程决策、能支撑跨团队沟通、能直接落地实施”的标准展开，而本章的作用就是帮助读者在进入细节之前找到最短路径。

### 📊 本章调研统计
- 调研总来源：**112 篇**
- 可公开链接：**20 条精选**
- 其余来源聚类汇总表：

| 来源类型 | 数量 | 核心结论 |
|---------|------|-----------|
| NVIDIA 官方文档/技术博客 | 26 | Thor、DriveOS 7、TensorRT 10、CUDA Graph、Nsight、Edge-LLM 构成车端部署主栈，硬件能力提升同时提高了引擎构建、量化验证和热管理复杂度。 |
| 开源框架与 GitHub 仓库 | 24 | Torch-TensorRT、ONNX Runtime、ONNX-MLIR、TVM、OpenXLA、Autoware、Apollo、Bench2DriveZoo 等项目显示部署已从“单模型优化”转向“生态联调”。 |
| 论文/期刊/综述 | 28 | 端到端、多任务、对抗鲁棒性、不确定性量化、DVFS 与热感知调度正在成为量产部署的前置研究，而不是纯研究议题。 |
| 标准/法规/协会资料 | 11 | ISO 26262、SOTIF、UN R155、ASAM OpenSCENARIO/OpenDRIVE 共同决定了部署验证、回放测试和安全论证边界。 |
| 技术博客/案例/课程 | 23 | 行业实践普遍采用角色分层、灰度发布、影子模式、差分 OTA 和闭环仿真来降低发布风险。 |

### 🔗 真实来源链接（20 条精选）
1. https://developer.nvidia.cn/blog/accelerate-autonomous-vehicle-development-with-the-nvidia-drive-agx-thor-developer-kit/ (NVIDIA Thor 开发者套件与 DriveOS 7 概览，2025)
2. https://developer.nvidia.cn/drive/documentation (NVIDIA DRIVE 文档入口)
3. https://docs.nvidia.com/deeplearning/tensorrt/latest/getting-started/quick-start-guide.html (TensorRT 官方快速入门)
4. https://docs.nvidia.com/deeplearning/tensorrt/latest/inference-library/work-quantized-types.html (TensorRT 量化类型与工作流)
5. https://nvidia.github.io/Torch-TensorRT/ (Torch-TensorRT 官方文档)
6. https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/ (Triton Inference Server 官方文档)
7. https://docs.nvidia.com/dl-cuda-graph/troubleshooting/performance-issues.html (CUDA Graph 性能问题排查)
8. https://developer.nvidia.com/nsight-systems (Nsight Systems 官方入口)
9. https://nvidia.github.io/TensorRT-Edge-LLM/0.4.0/developer_guide/01.2_Quick_Start_Guide.html (TensorRT Edge-LLM 在 Thor/边缘侧的快速开始)
10. https://developer.nvidia.com/blog/streamline-llm-deployment-for-autonomous-vehicle-applications-with-nvidia-driveos-llm-sdk/ (DriveOS LLM SDK 面向自动驾驶的部署路径，2025)
11. https://arxiv.org/abs/2404.18573 (自动驾驶安全失效预测与不确定性量化，2024)
12. https://arxiv.org/abs/2411.13778 (LiDAR 感知对抗鲁棒性综述，2024)
13. https://arxiv.org/abs/2603.21908 (SparseDVFS：面向边缘推理的稀疏性感知 DVFS，2026)
14. https://arxiv.org/abs/2604.04349 (云辅助自动驾驶系统对抗鲁棒性分析，2026)
15. https://github.com/NVIDIA/Model-Optimizer (NVIDIA 模型优化库：量化、剪枝、蒸馏)
16. https://github.com/onnx/onnx-mlir (ONNX-MLIR 开源编译器)
17. https://github.com/opendrivelab/end-to-end-autonomous-driving (端到端自动驾驶综述与资源索引)
18. https://github.com/Thinklab-SJTU/Bench2DriveZoo (闭环 CARLA 评测与部署基准)
19. https://carla.readthedocs.io/en/latest/ecosys_simready/ (CARLA 与 NVIDIA Omniverse/SimReady 生态集成)
20. https://unece.org/transport/documents/2021/03/standards/un-regulation-no-155-cyber-security-and-cyber-security (UN R155 网络安全法规入口，经典基础内容)

### 🔍 扩展检索关键词
`DRIVE AGX Thor deployment playbook`, `autonomous driving inference safety case`, `vehicle edge OTA canary rollback`

### ⚠️ 局限性说明
无
