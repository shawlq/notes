# 第 0 章研究日志：总览索引表（按角色）

## 1. 检索计划

### 1.1 目标
- 为第 0 章构建“按角色阅读导航”。
- 覆盖部署工程师、系统架构师、功能安全经理、MLOps 负责人、技术负责人五类角色。
- 建立后续章节可复用的一手来源池，优先 2024-2026 年公开资料。

### 1.2 关键词设计
- 硬件与平台：`NVIDIA DRIVE AGX Thor`, `DriveOS 7`, `DriveWorks`, `DRIVE Hyperion`, `Blackwell automotive`
- 推理与工具链：`TensorRT`, `Torch-TensorRT`, `TensorRT-LLM`, `ONNX Runtime TensorRT`, `Polygraphy`, `ModelOpt`
- 编译器与 IR：`OpenXLA`, `StableHLO`, `MLIR`, `TVM`, `IREE`, `ONNX`
- 量化与压缩：`INT8`, `FP8`, `FP4`, `QAT`, `PTQ`, `model compression`, `3D perception quantization`
- 实时性与调优：`CUDA memory management`, `CUDA Graphs`, `Nsight Systems`, `Nsight Compute`, `deterministic inference`
- 安全与合规：`ISO 26262`, `ISO 21448`, `ISO/PAS 8800`, `ISO/SAE 21434`, `UN R155`, `UN R156`
- MLOps 与运营：`model registry`, `lineage`, `OTA`, `shadow deployment`, `canary release`, `rollback`
- 仿真与验证：`CARLA`, `nuPlan`, `Omniverse`, `Isaac Sim HIL`, `OpenSCENARIO`, `OpenDRIVE`, `VTD`
- 高级话题：`end-to-end autonomous driving`, `multi-task learning deployment`, `explainable autonomous driving`, `adversarial robustness`

### 1.3 网站范围
- 一级优先：`nvidia.com`, `developer.nvidia.com`, `docs.nvidia.com`
- 二级优先：`arxiv.org`, `github.com`, `onnxruntime.ai`, `openxla.org`, `tvm.apache.org`, `onnx.ai`
- 三级优先：`iso.org`, `unece.org`, `asam.net`, `nuplan.org`, `carla.readthedocs.io`
- 补充来源：高质量行业白皮书、技术博客、课程页面、研究机构资料

### 1.4 预期数量
- 目标总来源：100-130 条
- 目标一手公开链接：至少 20 条
- 预期来源结构：
  - 官方文档/博客 35+
  - 论文/综述 20+
  - GitHub/示例仓库 15+
  - 标准/法规/白皮书 15+
  - 仿真与验证资料 10+

## 2. 中间汇总（累计 50 条资料）

### 2.1 已覆盖主题
- Thor 硬件与 DriveOS/DriveWorks 主线已建立。
- TensorRT、Torch-TensorRT、ONNX Runtime TensorRT EP、TensorRT-LLM、Triton 等推理链路已覆盖。
- CUDA/Nsight 性能分析与低延迟优化资料已具备。
- OpenXLA、StableHLO、TVM、MLIR 基础来源已收集。
- 量化、QAT/PTQ、FP8/INT8 文档与部分自动驾驶论文已纳入。
- ISO 26262、ISO 21448、ISO/PAS 8800、UN R155 基本入口已覆盖。

### 2.2 阶段判断
- 50 条时已足够支撑“角色导航”的初稿。
- 但 MLOps、仿真/HIL、异构迁移、可解释性四类内容仍偏薄，不足以支撑后续章节复用。
- 因此继续扩展资料池，而不是直接写作。

### 2.3 阶段性结论
1. Thor 相关公开资料的最佳入口是 `DRIVE Documentation + Thor devkit blog + DriveOS 7.0.3 docs`。
2. 部署工程师阅读主线可稳定落在 `TensorRT + CUDA/Nsight + ModelOpt + ONNX Runtime/Torch-TensorRT`。
3. 架构师阅读主线必须同时包含 `IR/编译器` 与 `实时性/隔离性`，仅看推理框架不够。
4. 功能安全与部署已经高度耦合，不能拆成“后验认证”问题。

## 3. 中间汇总（累计 100 条资料）

### 3.1 新增覆盖
- MLOps：MLflow、W&B Registry、影子模式、金丝雀发布、回滚治理。
- 仿真与验证：CARLA、Isaac Sim HIL、nuPlan、OpenSCENARIO、OpenDRIVE、VTD。
- 异构与迁移：Thor/Orin 差异、ONNX 边界、跨平台执行提供程序和编译路径。
- 技术负责人视角：端到端自动驾驶、多任务学习、可解释性、对抗鲁棒性。

### 3.2 阶段判断
- 100 条后已可支撑第 0 章正文与后续章节导航。
- 资料池中高复用来源足够多，可减少后续章节重复检索压力。
- 部分 ISO 页面存在反爬或仅提供标准介绍页，因此标准类来源以官方入口页、法规正文和行业权威解读混合使用。

### 3.3 阶段性结论
1. 第 0 章不应写成“摘要目录”，而应写成“角色驱动的阅读策略”。
2. 真正能服务业务团队的导航，不是简单列章节，而是把角色问题映射到章节。
3. 后续章节应优先沿“官方文档 + 论文综述 + GitHub 工程仓库 + 标准法规”的四层证据结构展开。

## 4. 本轮来源聚类摘要

| 聚类 | 代表来源 | 用途 |
|------|----------|------|
| NVIDIA 车端官方栈 | DRIVE Documentation、Thor blog、DriveOS 7、DriveWorks、TensorRT for DRIVE | 建立 Thor/DriveOS 部署主线 |
| 通用推理与编译 | TensorRT、Torch-TensorRT、ORT、OpenXLA、TVM、IREE、ONNX | 建立框架/IR/编译器对照 |
| 性能分析与低延迟 | CUDA Toolkit、Nsight Systems、Nsight Compute | 建立调优与实时性阅读入口 |
| 安全合规 | ISO 26262、ISO 21448、ISO/SAE 21434、ISO/PAS 8800、UN R155/R156 | 建立功能安全与网络安全入口 |
| MLOps 与版本治理 | MLflow、W&B Registry、AWS deployment guidance | 建立版本、灰度、回滚阅读入口 |
| 仿真与验证 | CARLA、Isaac Sim、nuPlan、ASAM 标准、VTD | 建立仿真/HIL/回放入口 |
| 高级研究议题 | 端到端自动驾驶综述、多任务学习、可解释性、对抗鲁棒性 | 建立技术负责人阅读入口 |

## 5. 去重统计口径
- 本轮候选来源总数：126
- 用于正文精选公开链接：20
- 其余来源按聚类统计入表，不逐条展开
- 去重以 URL 为准；同一主题的中英文镜像页视为不同 URL，但正文优先保留英文原始页或官方入口页
