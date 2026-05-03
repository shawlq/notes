# 第 0 章《总览索引表（按角色）》检索台账

## 一、检索计划

### 1. 检索目标
- 为知识库首页建立“按角色导航”的证据基础，而不是只做章节目录复述。
- 覆盖芯片/硬件、部署方法、推理框架、量化优化、工具链、调优、实时性、安全性、MLOps、仿真/HIL、跨芯片迁移、功耗热管理、趋势与合规。
- 优先 2024-2026 年公开资料；如引用更早资料，仅作为经典基础入口。

### 2. 关键词矩阵
- NVIDIA / 芯片：`DRIVE AGX Thor`, `DriveOS 7`, `TensorRT 10`, `TensorRT Edge-LLM`, `DriveOS LLM SDK`, `Jetson Thor`, `Jetson Orin`
- 部署方法：`end-to-end autonomous driving deployment`, `two-stage deployment`, `BEV planning inference`
- 推理与编译：`Torch-TensorRT`, `ONNX Runtime TensorRT EP`, `ONNX-MLIR`, `TVM`, `OpenXLA`, `MLIR`, `CUDA Graph`
- 优化：`INT8 FP8 NVFP4`, `PTQ QAT`, `model optimizer`, `memory management`, `DAG scheduling`
- 验证与安全：`SOTIF`, `ISO 26262`, `UN R155`, `adversarial robustness autonomous driving`, `uncertainty quantification`
- 仿真与发布：`CARLA Omniverse`, `Autoware CARLA`, `Apollo simulation`, `HIL replay`, `OTA canary rollback shadow mode`

### 3. 网站范围
- 官方/标准：`nvidia.com`, `docs.nvidia.com`, `developer.nvidia.com`, `openxla.org`, `onnx.ai`, `onnxruntime.ai`, `tvm.apache.org`, `mlir.llvm.org`, `asam.net`, `unece.org`, `autoware.org`, `apollo.auto`, `carla.org`
- 论文/期刊：`arxiv.org`, `sciencedirect.com`, `ieeexplore.ieee.org`, `dl.acm.org`, `openreview.net`
- 开源：`github.com`
- 工程案例：行业技术博客、厂商博客、课程与案例页

### 4. 预期数量
| 类别 | 预期条数 |
|---|---:|
| NVIDIA 官方文档/博客 | 25 |
| 开源框架与仓库 | 20 |
| 论文/综述/期刊 | 25 |
| 标准/法规/协会资料 | 10 |
| 工程博客/案例/课程 | 20 |
| 合计 | 100 |

---

## 二、中间汇总（第 1 轮，累计 54 条）

### 1. 覆盖情况
- 已覆盖主题：Thor/DriveOS、TensorRT/Torch-TensorRT/Triton、ONNX/TVM/MLIR/OpenXLA、端到端/BEV、量化、CARLA/Autoware/Apollo、UN R155、SOTIF、Captum/SHAP。
- 已识别高价值角色映射：
  - 部署工程师最依赖推理栈、量化与性能分析资料。
  - 系统架构师最依赖芯片架构、时延预算、跨平台兼容资料。
  - 安全经理最依赖 SOTIF/UN R155/解释性/不确定性量化资料。
  - MLOps 负责人最依赖版本、仿真、灰度、回滚资料。

### 2. 阶段性结论
1. Thor 相关资料集中在 NVIDIA 博客、DRIVE 文档与 Jetson/Thor 边缘文档，足以支撑首页对芯片与工具链的角色化导航。
2. 端到端与两段式并存，首页不能只做“趋势判断”，必须强调不同角色的阅读路径差异。
3. 部署生态已经从单一引擎扩展为编译器、运行时、分析器、发布系统与仿真系统的组合。
4. 安全与功耗不应被放在附录，而应进入角色导航主线。

### 3. 阶段性来源分布
| 来源类型 | 数量 |
|---|---:|
| 官方文档/博客 | 16 |
| GitHub/开源项目 | 12 |
| 论文/综述 | 14 |
| 标准/法规 | 5 |
| 工程案例/博客 | 7 |
| 合计 | 54 |

---

## 三、中间汇总（第 2 轮，累计 112 条）

### 1. 最终覆盖情况
- 芯片与硬件：Thor、Jetson Thor/Orin、DriveOS、TensorRT 10、CUDA Graph、Nsight
- 部署方法：两段式、端到端、BEV、规划导向、多任务统一模型
- 推理与编译：TensorRT、Torch-TensorRT、Triton、ONNX Runtime、ONNX-MLIR、TVM、OpenXLA、MLIR
- 优化：量化、蒸馏、剪枝、QAT/PTQ、内存管理、热感知 DVFS
- 验证：CARLA、Omniverse、Autoware、Apollo、OpenSCENARIO/OpenDRIVE、HIL、回放
- 安全与运维：ISO 26262、SOTIF、UN R155、解释性、对抗鲁棒性、影子模式、灰度发布、回滚、OTA

### 2. 最终结论
1. 首页应按角色建立阅读入口，而不是按技术栈罗列名词，否则业务团队无法快速判断“谁先看什么”。
2. 2025-2026 年的自动驾驶部署主线已经变成“芯片能力 + 编译/推理工具链 + 发布治理 + 安全验证”的联合优化。
3. 角色导航页必须强调跨团队耦合：部署与架构共同决定预算，安全与 MLOps 共同决定发布边界，技术负责人负责主题取舍与组织能力建设。
4. 资料充足，可支撑后续章节逐章展开，无需在第 0 章堆砌实现细节。

### 3. 最终来源分布
| 来源类型 | 数量 |
|---|---:|
| NVIDIA 官方文档/技术博客 | 26 |
| 开源框架与 GitHub 仓库 | 24 |
| 论文/期刊/综述 | 28 |
| 标准/法规/协会资料 | 11 |
| 技术博客/案例/课程 | 23 |
| 合计 | 112 |

---

## 四、来源聚类说明（用于后续章节断点续传）
- **芯片与官方生态入口**：Thor、DriveOS、TensorRT、CUDA Graph、Nsight、Edge-LLM、LLM SDK
- **开源部署入口**：Torch-TensorRT、Triton、ONNX Runtime、ONNX-MLIR、TVM、OpenXLA、MLIR
- **自动驾驶方法入口**：UniAD、VAD、BEVFormer、BEVDet、LSS、Bench2DriveZoo、End-to-end Autonomous Driving
- **仿真/回放入口**：CARLA、Autoware CARLA Interface、Apollo、Cyber RT、ASAM OpenSCENARIO/OpenDRIVE
- **安全/合规入口**：UN R155、ISO 26262、SOTIF、UL 4600、对抗鲁棒性、不确定性量化、Captum、SHAP
- **运维发布入口**：模型注册、OTA 差分更新、影子模式、金丝雀发布、回滚与热切换

## 五、后续章节建议
- 第 1 章优先从 Thor 关键硬件单元切入，继续沿用“官方文档 + 论文/博客 + GitHub 工具链 + 调优案例”的混合采样方式。
- 每章继续保持：检索计划 -> 50 条中间汇总 -> 100 条完成汇总 -> 正文章节。
