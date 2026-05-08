# infer_repo 模型加载/部署/编排/流水线梳理（v2）

## 0. 梳理口径

- 只覆盖 `infer_repo` 顶层 5 个独立仓库：
  - `Apollo-Vision-Net-Deployment`
  - `OpenPCDet`
  - `lift-splat-shoot`
  - `mmdetection3d`
  - `bevfusion`
- 重点维度：模型加载、部署、CUDA Graph、编排、pipeline、多模型/多模态协调。
- 结论先行：全仓未发现显式 CUDA Graph（`cuda graph` / `CUDAGraph` / `torch.cuda.graph` 等均无命中）。

---

## 1. 仓库级流程图（文字版）

### 1.1 `Apollo-Vision-Net-Deployment`

**A) PyTorch -> ONNX**

`tools/pth2onnx.py`
-> 读取 config（含 plugin 动态导入）
-> `det2trt.convert.pytorch2onnx(...)`
-> 输出 ONNX 到 `config.ONNX_PATH`

**B) ONNX -> TensorRT**

`tools/2d/onnx2trt.py` / `tools/bevformer/onnx2trt.py` / `tools/bevdet/onnx2trt.py`
-> `load_tensorrt_plugin()`
-> 读取 dynamic shape 配置
->（可选）构建 calibrator（INT8）
-> `build_engine(...)`
-> 输出 `.trt` 到 `config.TENSORRT_PATH`

**C) TensorRT 推理评测**

`tools/*/evaluate_trt.py`
-> `create_engine_context(...)` 反序列化引擎
-> `allocate_buffers(...)` 分配 host/device buffer
-> `do_inference(...)`（`execute_async_v2`）
-> 借助 PyTorch 模型 `post_process` 做结果解码与评估

**D) 关键设计**

- 解码与评估尽量复用原 PyTorch 后处理，减少 TRT 分支重复逻辑。
- BEVFormer 分支显式维护时序状态（`prev_bev`、`can_bus` 增量、`scene_token`）。
- INT8 校准中先用 FP32 引擎跑出时序 `prev_bev`，再喂给 calibrator。

---

### 1.2 `OpenPCDet`

**A) 构建与加载**

`tools/demo.py` / `tools/test.py`
-> `build_network(model_cfg, num_class, dataset)`
-> `model.load_params_from_file(...)`
-> `model.cuda().eval()`

**B) 推理主链**

`load_data_to_gpu(batch_dict)`
-> `model.forward(batch_dict)`
-> `Detector3DTemplate.post_processing(...)`（NMS、多头标签映射）

**C) 模块编排**

`Detector3DTemplate.module_topology`：

`vfe -> backbone_3d -> map_to_bev_module -> pfe -> backbone_2d -> dense_head -> point_head -> roi_head`

**D) 关键设计**

- 明确的“数据-模型分离 + 模块拼装”范式。
- checkpoint 加载内置 spconv 版本差异适配（权重 shape 转换）。
- 仓内无完整 ONNX/TensorRT 导出-构建-评估工具链脚本。

---

### 1.3 `lift-splat-shoot`

**A) 统一入口**

`main.py`（Fire 命令分发）
-> `train` / `eval_model_iou` / `viz_model_preds` / `lidar_check` / `cumsum_check`

**B) 模型前向**

`LiftSplatShoot.forward(...)`
-> `get_voxels(...)`
-> `get_geometry(...)`（相机到 ego 几何）
-> `get_cam_feats(...)`（相机特征 + 深度分布）
-> `voxel_pooling(...)`（Lift + Splat）
-> `bevencode(...)`

**C) 关键设计**

- 多相机融合是核心，不涉及多模型协同调度。
- 通过 `QuickCumsum` 优化 BEV pooling 的聚合路径。
- 偏研究原型，部署链路较轻。

---

### 1.4 `mmdetection3d`

**A) API 模型加载**

`mmdet3d/apis/inference.py::init_model`
-> 读 config
-> SyncBN 转普通 BN
-> `MODELS.build(config.model)`
-> `load_checkpoint(...)`
-> 挂 `dataset_meta`
-> `model.to(device).eval()`

**B) 推理接口族**

- `inference_detector`（点云）
- `inference_multi_modality_detector`（点云+图像+ann）
- `inference_mono_3d_detector`（单目）
- `inference_segmentor`（点云分割）

共性：
构建 test pipeline（Compose）
-> 组装 data dict
-> `model.test_step(collate_data)`

**C) Inferencer 编排**

`Base3DInferencer` 将流程固定为：

`preprocess -> forward -> visualize -> postprocess`

支持预测落盘、可视化落盘、终端打印。

**D) 部署定位**

- `docs/en/user_guides/model_deployment.md`：部署依赖 MMDeploy。
- 框架侧主要负责模型/数据接口规范，不在仓内自维护完整 TRT 工具链。

---

### 1.5 `bevfusion`（独立仓库）

**A) 训练/测试编排**

- 训练：`tools/train.py`
  - `configs.load + recursive_eval`
  - `build_dataset`
  - `build_model`
  - `train_model(...)`
- 测试：`tools/test.py`
  - `build_dataset/build_dataloader`
  - `build_model + load_checkpoint`
  - `multi_gpu_test / single_gpu_test`

**B) 模型结构（核心）**

`mmdet3d/models/fusion_models/bevfusion.py`

- `encoders`：camera / lidar / radar（可选）
- `fuser`：融合器（如 ConvFuser）
- `decoder`：共享 BEV 解码器
- `heads`：object / map 多任务头

前向（`forward_single`）：

1. 逐 sensor 提取特征（camera 走 `backbone+neck+vtransform`，lidar/radar 走 `voxelize+backbone`）
2. `fuser(features)` 融合
3. `decoder`
4. 多 head 产出（训练返回 loss dict，推理返回 boxes/masks）

**C) 导出与部署**

- ONNX 导出脚本：`tools/export.py`（`torch.onnx.export` + `onnxsim.simplify`）。
- TensorRT 实战方案在 README 指向外部 CUDA-BEVFusion 工程。

---

## 2. 多模型 / 多模态协调方式对照

| 仓库 | 多模型形态 | 跨模型协调 | 多模态协调 | 时序协调 |
|---|---|---|---|---|
| Apollo-Vision-Net-Deployment | 多模型家族并行支持（2D/BEVDet/BEVFormer） | 以“分脚本分配置”切换，不做在线 ensemble | 有（BEV 系列） | 强（BEVFormer `prev_bev`） |
| OpenPCDet | 多 detector 共存 | 运行时通常单模型 | 有（部分模型如 BEVFusion/CaDDN） | 有（部分时序模型如 MPPNet） |
| lift-splat-shoot | 单主模型 | 无 | 多相机融合到 BEV | 无显式跨帧状态模块 |
| mmdetection3d | 框架级多模型 | 通过配置切换，不内置调度器 | 完整支持（LiDAR/Mono/Multi-modality） | 依具体模型实现 |
| bevfusion | 单框架多配置（camera/lidar/fusion + det/seg） | 通过 head/fuser 在模型内统一 | 强（camera/lidar/radar 可插拔） | 依配置和具体子模块 |

---

## 3. pipeline 设计差异（工程视角）

### 3.1 配置驱动强度

- 最高：`mmdetection3d`、`bevfusion`（数据、模型、训练策略都在 config/yaml）。
- 中等：`OpenPCDet`（yaml + 明确模块拓扑）。
- 脚本驱动更明显：`Apollo-Vision-Net-Deployment`、`lift-splat-shoot`。

### 3.2 部署链闭环程度

- 完整闭环（导出+构建+评测）：`Apollo-Vision-Net-Deployment`
- 半闭环（导出有、部署实战外链）：`bevfusion`
- 平台委托型（MMDeploy）：`mmdetection3d`
- 弱部署、强训练评测：`OpenPCDet`、`lift-splat-shoot`

### 3.3 后处理复用策略

- Apollo 在 TRT 推理后复用 PyTorch 后处理（降低结果偏差风险）。
- mmdetection3d/bevfusion 统一通过 head 接口规范输出。
- OpenPCDet 在 `Detector3DTemplate.post_processing` 中集中管理 NMS 和 recall 统计。

---

## 4. CUDA Graph 专项结论

对 `infer_repo` 全量源码检索以下关键词均无命中：

- `cuda graph`
- `CUDAGraph`
- `torch.cuda.graph`
- `make_graphed_callables`
- `graph capture`

结论：当前 5 仓库均**未显式实现 CUDA Graph 推理/训练捕获**。  
若线上有 CUDA Graph，通常应在仓外部署层（服务框架、推理网关、私有 runtime）实现。

---

## 5. 可直接落地的二轮深挖建议

1. **Apollo**：补每个 `samples/*.sh` 的“命令-产物”矩阵（onnx/trt/metric 对齐）。
2. **OpenPCDet**：按常用模型（SECOND/PVRCNN/CenterPoint）生成最小推理调用链差异图。
3. **mmdetection3d**：补 `Inferencer` 与 `demo` 的接口边界（输入组织与输出落盘格式）。
4. **bevfusion**：单独拆 camera/lidar/radar 三支的张量形态与融合点位。
5. **跨仓统一**：输出“从训练 checkpoint 到线上推理”的标准迁移模板（包含导出、校验、回归项）。

