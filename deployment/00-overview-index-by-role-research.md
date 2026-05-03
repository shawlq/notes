# 第 0 章研究日志：总览索引表（按角色）

## 检索计划

### 目标
- 为第 0 章建立一份面向业务团队的角色导航页。
- 覆盖全书后续章节涉及的 10 个主题面。
- 去重后来源数不少于 100 条，优先 2024-2026 年公开资料。

### 检索关键词
- `NVIDIA DRIVE AGX Thor deployment`
- `Jetson AGX Thor user guide`
- `DriveOS 7.0.3 TensorRT 10 DriveWorks`
- `TensorRT Edge-LLM automotive robotics`
- `CUDA Nsight Systems deterministic inference`
- `NVIDIA Model Optimizer PTQ QAT`
- `automotive MLOps OTA shadow mode rollback`
- `CARLA Omniverse HIL deterministic replay`
- `OpenSCENARIO OpenDRIVE VTD`
- `ISO 26262 ISO 21448 SOTIF UN R155 UN R156`
- `ONNX MLIR OpenXLA heterogeneous scheduling`
- `Jetson Thor power thermal DVFS`
- `federated learning autonomous vehicles 2025`

### 网站范围
- `nvidia.com`
- `docs.nvidia.com`
- `developer.nvidia.com`
- `arxiv.org`
- `github.com`
- `onnx.ai`
- `openxla.org`
- `tvm.apache.org`
- `carla.org`
- `carla.readthedocs.io`
- `asam.net`
- `unece.org`
- `iso.org`
- `mlflow.org`
- `kubernetes.io`
- `istio.io`
- `argo-rollouts.readthedocs.io`

### 预期数量
| 主题 | 预期来源数 |
|------|-----------|
| Thor / DRIVE / Jetson Thor | 15 |
| TensorRT / CUDA / Nsight / cuDNN / CUTLASS | 20 |
| 量化 / 压缩 / QAT / PTQ | 10 |
| MLOps / OTA / 灰度 / 回滚 | 10 |
| 仿真 / HIL / 回放 / 回归 | 15 |
| 实时性 / 确定性 / 隔离 | 8 |
| 安全 / 可解释性 / SOTIF | 12 |
| 跨芯片 / 异构 / ONNX / MLIR / OpenXLA / TVM | 12 |
| 功耗 / 热管理 / DVFS | 8 |
| 2025-2026 趋势 / 联邦学习 / 合规 | 10 |
| 合计 | 120 |

## 中间汇总（完成约 50 条资料后）

### 覆盖情况
- 已覆盖 Thor/DRIVE、TensorRT/CUDA、量化、MLOps、仿真五个主干主题。
- 已拿到 DRIVE 文档中心、Jetson Thor 用户指南、TensorRT/Edge-LLM 文档、ModelOpt 文档、CARLA/Omniverse 资料等高优先级一手入口。
- 已发现第 0 章不适合堆砌实现细节，更适合强调“不同角色为什么应该先读哪些章节”。

### 阶段性判断
1. 系统架构师最应优先关注平台与实时性，因为 Thor/Orin、DriveOS/Jetson、异构调度和热设计会决定后续部署上限。
2. 部署工程师最应优先关注 TensorRT、量化和性能分析工具，否则无法把训练产物稳定变成车端制品。
3. MLOps 负责人不能只看版本管理，还必须把 OTA、影子模式、回滚和仿真验证打通。
4. 功能安全经理在自动驾驶部署里必须同时关注 SOTIF 与推理降级，因为热节流和回退策略会直接影响安全论证。

### 当时已确认的高价值来源类别
- NVIDIA 官方文档与官方博客
- GitHub 上游仓库
- ASAM / UNECE / ISO 标准页
- CARLA / Omniverse 公开文档

## 完成汇总（去重后 120 条）

### 去重总数
- 去重后总来源：120
- 其中正文精选：20
- 其余来源：100

### 全量来源类型统计
| 来源类型 | 数量 | 说明 |
|---------|------|------|
| 官方文档 | 62 | 车端平台、编译器、推理框架、内核与平台工具的一手说明 |
| 官方博客 | 6 | 平台路线、Edge-LLM、仿真与趋势解读 |
| 新闻/公告 | 1 | 平台路线与生态进展 |
| GitHub | 20 | 上游实现、issue、发布说明与样例 |
| 标准/法规与指南 | 15 | ASAM、UNECE、ISO、NIST 等权威材料 |
| IEEE 文献 | 2 | 学术入口和专题论文线索 |
| 论文（预印本） | 10 | 新方法与趋势补充 |
| 第三方技术博客 | 1 | 工程导读 |
| Medium | 1 | 发布策略经验总结 |
| 论坛与社区 | 2 | 版本兼容与真实坑点 |

### 余下 100 条来源类型统计
| 来源类型 | 数量 | 说明 |
|---------|------|------|
| 官方文档 | 48 | 用于后续各章继续下钻 |
| 官方博客 | 5 | 作为路线与设计背景补充 |
| 新闻/公告 | 1 | 趋势侧参考 |
| GitHub | 19 | 工程实现与排障线索 |
| 标准/法规与指南 | 11 | 合规与验证侧支撑 |
| IEEE 文献 | 2 | 进一步扩读入口 |
| 论文（预印本） | 10 | 新方法和趋势判断 |
| 第三方技术博客 | 1 | 工程概念辅材 |
| Medium | 1 | 发布治理方法参考 |
| 论坛与社区 | 2 | 生态差异和兼容性问题追踪 |

### 本章写作结论
- 第 0 章的价值不在技术细节，而在于帮助角色对后续章节排序。
- 本章必须明确“平台约束 -> 编译部署 -> 运行调优 -> 验证闭环 -> 发布治理”的主路径。
- 业务团队最需要的是按职责快速定位，而不是先通读全书。
