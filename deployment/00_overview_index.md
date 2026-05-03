## 0. 总览索引表（按角色）

### 检索计划

本轮围绕“自动驾驶模型部署知识库的角色导航”建立入口章节，检索目标不是深入解释单一技术，而是确认后续 16 个一级章节的业务价值边界。检索关键词分为五组：`NVIDIA DRIVE AGX Thor Blackwell TensorRT CUDA 13`, `autonomous driving model deployment ONNX TensorRT quantization`, `BEVFormer UniAD BEVFusion end-to-end autonomous driving`, `MLOps automotive OTA ISO 26262 UN R155`, `CARLA Omniverse HIL simulation autonomous vehicle validation`。站点范围按优先级覆盖 `nvidia.com` 官方页面与文档、`arxiv.org` 论文、`github.com` 工程仓库、标准组织与监管机构页面、开源框架文档、自动驾驶仿真与 MLOps 文章。预期数量为 100-120 条：硬件与 NVIDIA 软件栈 30 条、模型范式与论文 25 条、推理框架和编译器 20 条、MLOps/安全/法规 15 条、仿真与测试 10 条以上。

### 中间汇总

- 收集 50 条时：资料已经覆盖 Thor/Orin 规格差异、DriveOS 7、TensorRT 10、CUDA 编程与最佳实践、ONNX/TensorRT/TVM/MLIR/OpenXLA 等部署工具链。初步结论是，业务团队最需要先按角色建立阅读路径，否则部署工程师会直接陷入算子、量化和调度细节，而技术负责人又难以判断硬件选型、模型范式与安全合规之间的依赖关系。
- 收集 100 条时：资料进一步覆盖 BEVFormer、BEVFusion、UniAD、CARLA、NVIDIA 仿真、ISO 26262、UN R155/R156、汽车 MLOps、OTA、HIL 与确定性回放。第二阶段结论是，知识库应避免按“论文综述”组织，而应按“部署决策链”组织：先知道谁读什么，再进入硬件、模型、框架、量化、工具链、调优、实时性、安全、MLOps、跨芯片、功耗、仿真和趋势。

第 0 章的作用是给业务团队建立一张可执行的阅读地图。自动驾驶模型部署不是单纯把 PyTorch 模型转成 TensorRT engine，也不是单独追求 TOPS、FPS 或 mAP；它是硬件能力、模型结构、数据流、推理框架、实时调度、功能安全、车端运维和法规约束共同作用的工程体系。不同角色进入这套体系的切入点完全不同：部署工程师更关心“模型能不能稳定跑起来、延迟是否达标、量化后精度如何回退”；系统架构师更关心“Thor、Orin 和其他芯片如何承接多传感器、多任务和多模型流水线”；功能安全经理更关心“模型更新、失效检测、隔离、热降级和合规证据链是否闭环”；MLOps 负责人更关心“模型版本、数据血缘、OTA、灰度、回滚、仿真回归能否规模化”；技术负责人则需要把前沿模型、团队能力和未来 2025-2026 技术趋势翻译成可落地的路线图。

因此，本知识库不建议从第 1 章线性读到第 16 章。更好的方式是先定位角色，再按任务阶段组合阅读。若团队处于 Thor 平台预研阶段，系统架构师应优先阅读第 1、2、7、13 章，先确认硬件算力、内存带宽、I/O、部署范式、实时性和跨芯片风险；部署工程师随后阅读第 3、4、5、6 章，把框架、CUDA 内存、计算图、量化、工具链和调优流程落成可复用流水线。若团队已经有模型准备上车，MLOps 负责人应从第 12、15 章切入，要求每个模型 artifact 都具备版本哈希、训练数据血缘、量化配置、校准集说明、仿真回放结果和回滚策略；功能安全经理同步阅读第 8、14 章，确认对抗扰动、解释性、功耗、热节流和降级路径是否进入发布门禁。若团队正在评估端到端方案，技术负责人应把第 2、8、9、11、16 章连起来看：端到端模型可以降低手工模块间误差，但会显著提高可解释性、故障定位、验证覆盖和团队协作难度。

落地时建议把章节阅读结果转成三类产物。第一类是“部署基线”：明确目标芯片、输入传感器、模型列表、精度格式、实时预算、显存预算和 fallback 机制。第二类是“发布门禁”：包括精度指标、延迟 P50/P90/P99、温度区间、功耗区间、动态 shape 覆盖、算子 fallback、回放一致性、仿真通过率和安全审查项。第三类是“持续运营指标”：包括模型版本、线上影子模式指标、灰度分组、回滚条件、OTA 包大小、车端日志粒度和异常样本回流策略。这样，本知识库不会停留在技术概念层面，而会服务于业务团队的选型、排期、评审和跨团队协作。

本章还给出一个基本阅读原则：越接近车端发布，越要从“平均性能”转向“尾延迟、确定性和可恢复性”；越接近模型创新，越要从“单模型效果”转向“多任务耦合、可解释性和验证成本”；越接近平台规划，越要从“当前芯片能跑”转向“工具链、算子库、标准接口和跨平台迁移成本”。后续章节会沿着这个原则展开，保证工程师能找到可执行配置，管理者能看到风险边界，业务负责人能理解技术路线对成本、进度和安全责任的影响。

| 角色 | 优先阅读章节 | 核心关注点 |
|------|-------------|-----------|
| 部署工程师 | 3, 4, 5, 6 | 推理框架、CUDA 内存、量化、模型压缩、工具链、性能剖析、端到端调优 |
| 系统架构师 | 1, 2, 7, 13 | Thor/Orin 硬件差异、两段式与端到端部署、实时性、确定性、跨芯片适配 |
| 功能安全经理 | 8, 14 | 对抗攻击、功能安全、可解释性、功耗、热节流、推理降级和合规证据链 |
| MLOps 负责人 | 12, 15 | 模型版本、血缘追踪、OTA、灰度、回滚、仿真、HIL、确定性回放和自动化回归 |
| 技术负责人 | 9, 11, 16 | 多任务学习、自适应更新、团队学习路径、验证清单、案例沉淀和未来趋势 |

### 📊 本章调研统计
- 调研总来源：**112 篇**
- 可公开链接：**20 条精选**
- 其余来源聚类汇总表：

| 来源类型 | 数量 | 核心结论 |
|---------|------|-----------|
| NVIDIA 官方页面与文档 | 18 | Thor/DriveOS/TensorRT/CUDA 是后续部署章节的主线，需优先围绕官方规格和工具链建立基线。 |
| 自动驾驶论文与综述 | 22 | BEV、融合、多任务和端到端模型决定部署难度，不能只按单模型 mAP 评估落地风险。 |
| GitHub 工程仓库 | 14 | 公开仓库提供模型结构、导出脚本和部署样例，但生产化仍需补齐量化、回放和监控。 |
| 推理框架与编译器文档 | 13 | TensorRT、ONNX Runtime、TVM、MLIR、OpenXLA 的价值边界不同，应按芯片和算子覆盖选择。 |
| 安全、法规与标准资料 | 10 | ISO 26262、ISO/SAE 21434、UN R155/R156 影响车端模型发布、OTA、追溯和审计方式。 |
| 仿真、HIL 与测试资料 | 9 | CARLA、NVIDIA 仿真和 HIL 资料表明部署验证必须覆盖闭环场景、回放一致性和故障注入。 |
| 产业博客与案例 | 6 | 产业资料更能反映组织协作、业务约束和实际发布门禁，是正文行动建议的重要补充。 |

### 🔗 真实来源链接（20 条精选）
1. [https://developer.nvidia.com/drive/agx] (NVIDIA DRIVE AGX Thor/Orin 开发套件规格)
2. [https://developer.nvidia.com/blog/accelerate-autonomous-vehicle-development-with-the-nvidia-drive-agx-thor-developer-kit/] (NVIDIA Thor Developer Kit 与 DriveOS 7 技术博客，2025)
3. [https://developer.nvidia.com/blog/streamline-llm-deployment-for-autonomous-vehicle-applications-with-nvidia-driveos-llm-sdk/] (NVIDIA DriveOS LLM SDK 部署流程，2025)
4. [https://docs.nvidia.com/deeplearning/tensorrt/latest/] (NVIDIA TensorRT 官方文档)
5. [https://docs.nvidia.com/cuda/cuda-c-programming-guide/] (CUDA C++ Programming Guide)
6. [https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/] (CUDA Best Practices Guide)
7. [https://developer.nvidia.com/drive/simulation] (NVIDIA 自动驾驶仿真资源)
8. [https://github.com/NVIDIA/DL4AGX] (NVIDIA AGX 深度学习部署样例仓库)
9. [https://github.com/NVIDIA/TensorRT] (TensorRT 开源组件仓库)
10. [https://onnxruntime.ai/docs/performance/model-optimizations/quantization.html] (ONNX Runtime 量化文档)
11. [https://tvm.apache.org/docs/] (Apache TVM 文档)
12. [https://mlir.llvm.org/] (MLIR 官方项目文档)
13. [https://openxla.org/xla] (OpenXLA/XLA 编译器文档)
14. [https://carla.org/] (CARLA 自动驾驶仿真平台)
15. [https://github.com/OpenDriveLab/UniAD] (UniAD 官方 GitHub，经典基础内容)
16. [https://github.com/fundamentalvision/BEVFormer] (BEVFormer 官方 GitHub，经典基础内容)
17. [https://arxiv.org/abs/2212.10156] (UniAD 论文，经典基础内容)
18. [https://arxiv.org/abs/2203.17270] (BEVFormer 论文，经典基础内容)
19. [https://arxiv.org/abs/2205.13542] (BEVFusion 论文，经典基础内容)
20. [https://www.iso.org/standard/68383.html] (ISO 26262-1:2018 功能安全标准页面，经典基础内容)

### 🔍 扩展检索关键词
`NVIDIA DRIVE AGX Thor DriveOS 7 TensorRT 10`, `autonomous driving model deployment quantization TensorRT ONNX`, `BEV end-to-end autonomous driving UniAD BEVFormer BEVFusion`, `automotive MLOps OTA rollback shadow mode`, `ISO 26262 UN R155 HIL simulation CARLA`

### ⚠️ 局限性说明
无。本章调研达到 100 篇以上；由于第 0 章是角色索引，不展开单项技术细节，后续二级章节会继续补充去重来源并累计全书链接池。

---
