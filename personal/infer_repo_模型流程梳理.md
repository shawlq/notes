# infer_repo 模型加载/部署/编排/流水线梳理（第一版）

## 范围与结论总览

本次梳理覆盖 `infer_repo` 下 5 个独立仓库：

1. `Apollo-Vision-Net-Deployment`
2. `OpenPCDet`
3. `lift-splat-shoot`
4. `mmdetection3d`
5. `bevfusion`

核心结论：

- **CUDA Graph**：在 `infer_repo` 全局检索 `cuda graph / CUDAGraph / torch.cuda.graph / make_graphed_callables`，未发现显式实现。
- **部署能力最强**：`Apollo-Vision-Net-Deployment`（内置 PyTorch -> ONNX -> TensorRT 全链路）。
- **框架化最强**：`mmdetection3d`、`OpenPCDet`（配置驱动 + 注册构建）。
- **多模态融合设计最清晰**：`bevfusion`（独立版）与 `mmdetection3d/projects/BEVFusion`。
- **论文原型最直接**：`lift-splat-shoot`（训练/评估闭环完整，但部署链路较轻）。

## 1) Apollo-Vision-Net-Deployment

### 1.1 定位

- 仓库定位是 Apollo 视觉网络在 TensorRT 上部署，入口文档：`Apollo-Vision-Net-Deployment/README.md`。
- 覆盖 2D 检测、BEVDet、BEVFormer 等分支，提供量化与插件支持。

### 1.2 模型加载流程

- PyTorch 权重加载并导出 ONNX：
  - `tools/pth2onnx.py`
  - 内部调用 `det2trt.convert.pytorch2onnx`
  - 支持 `config.plugin` 动态导入（自定义算子/模型结构）
- 评测 PyTorch 模型：
  - `tools/bevformer/evaluate_pth.py`
  - 通过 `build_model(...) + load_checkpoint(...)` 加载权重，并走 TRT 友好的前向路径

### 1.3 部署流程（ONNX/TensorRT）

标准链路：

1. `pth2onnx.py`：导出 ONNX  
2. `tools/*/onnx2trt.py`：构建 TensorRT Engine  
3. `tools/*/evaluate_trt.py`：加载 `.trt` 做推理与评估  

关键实现：

- TRT 引擎加载：`det2trt/utils/tensorrt.py` 中 `deserialize_cuda_engine`
- 推理执行：`execute_async_v2` + CUDA stream
- 插件加载：`mmdeploy.backend.tensorrt.load_tensorrt_plugin`
- 量化：`tools/*/onnx2trt.py` + `det2trt/quantization/*` 支持 INT8 calibrator

### 1.4 编排与 pipeline

- 编排主要体现在：
  - `samples/*/*.sh`（大量预设脚本，串联导出/构建/评测）
  - config 驱动输入输出 shape（如 `configs/bevformer/*_trt.py` 中 `input_shapes`/`output_shapes`）
- 数据 pipeline 复用 MMDetection3D 生态（`third_party/bev_mmdet3d/datasets/pipelines/*`）

### 1.5 多模型/多子图协调

- 仓库级是“多模型家族并行支持”（2D/BEVDet/BEVFormer/Apollo BEV），不是 ensemble。
- 时序协调在 BEVFormer 中是关键：
  - `tools/bevformer/evaluate_trt.py` 维护 `prev_bev`、`scene_token`、`can_bus` 增量；
  - 每帧推理把上一帧 BEV 作为当前输入，属于状态传递式单模型时序推理。
- INT8 校准时也会先跑 FP32 引擎预生成 `prev_bev` 序列（`tools/bevformer/onnx2trt.py`）。

## 2) OpenPCDet

### 2.1 定位

- LiDAR 3D 检测框架，强调数据-模型解耦，入口：`OpenPCDet/README.md`。

### 2.2 模型加载与推理

- 构建模型：`pcdet/models/__init__.py` 的 `build_network(...)`
- 权重加载：`pcdet/models/detectors/detector3d_template.py` 的 `load_params_from_file(...)`
- Demo 推理：`tools/demo.py`  
  `build_network` -> `load_params_from_file` -> `load_data_to_gpu` -> `model.forward`

### 2.3 部署现状

- 仓库内没有完整 ONNX/TensorRT 工具链脚本（与 Apollo/独立部署仓不同）。
- README 提到部分模型可实现 TensorRT 实时速度，但实现更多偏外部或下游工程。

### 2.4 编排与 pipeline

- 训练/测试编排：`tools/train.py`、`tools/test.py`
- 支持分布式、断点评估、全 ckpt 轮询评估（`repeat_eval_ckpt`）
- 网络组装 pipeline：
  - `Detector3DTemplate.module_topology`
  - `vfe -> backbone_3d -> map_to_bev -> pfe -> backbone_2d -> dense_head -> point_head -> roi_head`

### 2.5 多模型协调

- 仓库层面支持多种 detector（SECOND/PV-RCNN/CenterPoint/BEVFusion 等），但通常单次运行单模型。
- 模型内部有多头协调机制（如 post-processing 中 multi-head class mapping + NMS）。

## 3) lift-splat-shoot

### 3.1 定位

- ECCV 2020 原始实现，偏研究原型，入口：`lift-splat-shoot/README.md`。

### 3.2 模型加载与结构

- 模型构建：`src/models.py` 的 `compile_model(...) -> LiftSplatShoot`
- Backbone 初始化：`CamEncode` 使用 `EfficientNet.from_pretrained("efficientnet-b0")`
- 权重加载：`src/explore.py` 中 `model.load_state_dict(torch.load(modelf))`

### 3.3 训练/评估/可视化编排

- 统一命令入口：`main.py`（Fire 路由）
  - `train`
  - `eval_model_iou`
  - `viz_model_preds`
  - `lidar_check` / `cumsum_check`
- 训练闭环：`src/train.py`（数据加载、loss、反向、定期验证、保存 checkpoint）

### 3.4 Pipeline 设计（核心）

主流程：

1. Camera Encoder 提取图像特征与深度分布  
2. Lift 到 frustum  
3. Splat 到 BEV voxel  
4. BEV encoder 输出任务结果

关键函数：

- `get_geometry`
- `get_cam_feats`
- `voxel_pooling`
- `forward`

### 3.5 多模型/多传感器协调

- 不是多模型协同；核心是多相机融合到统一 BEV。
- 多视角融合在 `voxel_pooling` 中完成。
- 没有独立部署脚本和 CUDA Graph 逻辑。

## 4) mmdetection3d

### 4.1 定位

- OpenMMLab 3D 框架，支持单模态/多模态，入口：`mmdetection3d/README.md`。

### 4.2 模型加载与推理 API

- 核心 API：`mmdet3d/apis/inference.py`
  - `init_model(config, checkpoint, device, ...)`
  - `inference_detector`（点云）
  - `inference_multi_modality_detector`（点云+图像+ann）
  - `inference_mono_3d_detector`
- 新版 inferencer：`mmdet3d/apis/inferencers/base_3d_inferencer.py`
  - 统一 preprocess/forward/visualize/postprocess
  - 支持结果落盘、可视化输出

### 4.3 部署

- 官方文档 `docs/en/user_guides/model_deployment.md` 明确：完全依赖 MMDeploy。
- 本仓库以框架能力 + 接口为主，部署实现下沉到 MMDeploy。

### 4.4 编排与 pipeline

- 强配置驱动：dataset pipeline / model / runtime 全在 config 中组合。
- demo 入口如 `demo/multi_modality_demo.py`：  
  `init_model` -> `inference_multi_modality_detector` -> visualizer

### 4.5 多模型/多模态协调

- 框架层支持多模态模型，但跨模型调度通常不内置。
- 在 `projects/BEVFusion` 中体现“阶段式参数迁移 + 融合训练”。

## 5) bevfusion（独立仓库）

### 5.1 定位

- MIT Han Lab 原始 BEVFusion 工程，入口：`bevfusion/README.md`。
- 提供 det/seg 两类任务，支持 camera-only、lidar-only、camera+lidar 配置。

### 5.2 模型加载与结构编排

- 模型核心：`mmdet3d/models/fusion_models/bevfusion.py`
- 顶层结构：
  - `encoders`（camera/lidar/radar 可选）
  - `fuser`
  - `decoder`
  - `heads`（object/map）
- 构建器：`mmdet3d/models/builder.py`（Registry 风格）

### 5.3 多传感器协调机制（重点）

- camera 分支：`backbone -> neck -> vtransform（到 BEV）`
- lidar/radar 分支：`voxelize -> sparse backbone`
- 融合：`features` 经过 `fuser`（如 ConvFuser）
- 多任务：`object head + map head` 可同时训练/推理，并可叠加 depth loss

### 5.4 部署与导出

- 仓库内有 ONNX 导出：`tools/export.py`（`torch.onnx.export` + `onnxsim.simplify`）。
- TensorRT 部署在 README 指向外部 NVIDIA 工程（CUDA-BEVFusion），非仓内完整链路。
- 测试入口：`tools/test.py`（分布式评估流程完整）。

### 5.5 数据与任务 pipeline

- `configs/nuscenes/default.yaml` 展示完整 pipeline：
  - 多视角图像 + 多 sweep 点云 + 标注 + 图像/3D 增广 + BEV seg + GTDepth + Collect
- 同时服务 det + seg，多任务输入输出在同一流水线中协调。

## 6) 跨仓对比：加载 / 部署 / 编排 / 多模型协调

### 6.1 模型加载风格

- 脚本直驱型：`Apollo-Vision-Net-Deployment`、`lift-splat-shoot`
- 框架注册型：`OpenPCDet`、`mmdetection3d`、`bevfusion`

### 6.2 部署成熟度

- 最完整内建部署链：`Apollo-Vision-Net-Deployment`
- 有导出能力、部署偏外链：`bevfusion`
- 部署交给 MMDeploy：`mmdetection3d`
- 偏训练/评测：`OpenPCDet`、`lift-splat-shoot`

### 6.3 编排层级

- Shell/脚本编排明显：Apollo（大量 `samples/*.sh`）
- 配置驱动编排最强：mmdetection3d、bevfusion
- 训练评估脚本清晰但偏单体：OpenPCDet、lift-splat-shoot

### 6.4 多模型/多传感器协调能力

- 多传感器融合最体系化：bevfusion（camera/lidar/radar + fuser + 多头任务）
- 时序状态协调最突出：Apollo BEVFormer（`prev_bev` 跨帧传递）
- 框架级多模态接口完善：mmdetection3d
- 多模型并存但单次单模型执行：OpenPCDet、lift-splat-shoot

### 6.5 CUDA Graph

- 当前 5 仓库中均未检索到显式 CUDA Graph 代码路径。

