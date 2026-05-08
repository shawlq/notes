# SparseDrive 模型流程与设计梳理

## 1. 总览结论

- **核心范式**：单模型骨干 + 多任务头联合（感知/建图/运动预测/规划）。
- **训练编排**：采用**两阶段训练**（stage1 感知+建图，stage2 加入运动规划）。
- **部署形态**：仓内以训练、评测、可视化为主，未提供完整 ONNX/TensorRT 部署链。
- **CUDA Graph**：检索未发现显式 `CUDAGraph`/`torch.cuda.graph` 实现。
- **多模型协调方式**：不是多独立模型在线串联，而是同一主干特征驱动多子头并行并在头间交互。

---

## 2. 模型加载与启动链路

## 2.1 训练入口

- 顶层脚本：`scripts/train.sh`
  - 先训 `projects/configs/sparsedrive_small_stage1.py`
  - 将 `work_dirs/sparsedrive_small_stage1/latest.pth` 拷贝为 `ckpt/sparsedrive_stage1.pth`
  - 再训 `projects/configs/sparsedrive_small_stage2.py`
- 分布式启动：`tools/dist_train.sh` -> `tools/train.py`（`torch.distributed.launch`）

## 2.2 测试入口

- `scripts/test.sh` -> `tools/dist_test.sh` -> `tools/test.py`
- `tools/test.py` 完成：
  - 读取 config
  - 构建 dataset/dataloader/model
  - `load_checkpoint`
  - 分布式推理与结果汇总
  - 按 `eval_mode` 执行多任务评估

## 2.3 主模型构建

- 主 detector：`projects/mmdet3d_plugin/models/sparsedrive.py`（类 `SparseDrive`）
  - 组件：`img_backbone` + `img_neck` + `head(SparseDriveHead)`
  - 训练：`extract_feat` -> `head` -> `head.loss`
  - 推理：`extract_feat` -> `head` -> `head.post_process`
- 插件导入：
  - 配置中 `plugin=True`, `plugin_dir="projects/mmdet3d_plugin/"`
  - 在 `tools/train.py`、`tools/test.py` 动态 import 注册模块

---

## 3. 多任务结构与协同

## 3.1 任务头总控

`SparseDriveHead`（`projects/mmdet3d_plugin/models/sparsedrive_head.py`）根据 `task_config` 管理：

- `det_head`：3D 检测/跟踪分支（`Sparse4DHead`）
- `map_head`：地图向量化分支（`Sparse4DHead`）
- `motion_plan_head`：运动预测与规划分支（`MotionPlanningHead`）

同一特征输入，多头并行前向，loss 合并，postprocess 结果合并。

## 3.2 两阶段任务开关

- `sparsedrive_small_stage1.py`：`with_motion_plan=False`
- `sparsedrive_small_stage2.py`：`with_motion_plan=True` 且 `load_from='ckpt/sparsedrive_stage1.pth'`

即：先收敛感知与地图，再引入规划任务。

## 3.3 跨任务交互

`MotionPlanningHead`（`models/motion/motion_planning_head.py`）显式接收：

- `det_output`
- `map_output`
- `feature_maps`

并执行：

- det/map top-k instance 筛选
- 时序队列融合（`InstanceQueue`）
- 跨任务注意力（`cross_gnn`）融合地图信息
- 同时输出 motion 与 planning

---

## 4. 时序建模机制

## 4.1 感知实例记忆：`InstanceBank`

文件：`projects/mmdet3d_plugin/models/instance_bank.py`

关键机制：

- 缓存历史 `cached_feature/cached_anchor`
- 使用 `timestamp` 与位姿矩阵投影历史 anchor 到当前帧
- 置信度衰减 + top-k 维护临时实例
- 维护并更新 `instance_id` 做跨帧关联

## 4.2 运动规划队列：`InstanceQueue`

文件：`projects/mmdet3d_plugin/models/motion/instance_queue.py`

关键机制：

- 保存 agent 历史特征/anchor 队列
- 保存 ego 历史特征/anchor 队列
- 依据 `instance_id` 做帧间匹配
- 产出 temporal memory 给 `MotionPlanningHead`

---

## 5. 数据 pipeline 与训练流程

## 5.1 数据准备

- `scripts/create_data.sh`
  - 调 `tools/data_converter/nuscenes_converter.py`
  - 生成 `data/infos/*.pkl` 与 map 标注
- `scripts/kmeans.sh`
  - 生成 `data/kmeans/*.npy`（det/map/motion/plan anchors）

## 5.2 训练 pipeline（配置）

主要在 `projects/configs/sparsedrive_small_stage1.py` / `stage2.py`：

- `LoadMultiViewImageFromFiles`
- `ResizeCropFlipImage`
- `MultiScaleDepthMapGenerator`
- `BBoxRotation`
- `PhotoMetricDistortionMultiViewImage`
- `VectorizeMap`
- `NuScenesSparse4DAdaptor`
- `Collect`（含 det/map/motion/planning 所需 GT）

## 5.3 测试 pipeline

- 保留多视角图像预处理与 adaptor
- `Collect` 中保留规划相关状态字段（如 `ego_status`、`gt_ego_fut_cmd`）

---

## 6. 评估编排

评估主入口在 `NuScenes3DDataset.evaluate`（`datasets/nuscenes_3d_dataset.py`），通过 `eval_mode` 控制：

- `with_det`：检测（可选 tracking）
- `with_map`：地图向量评估
- `with_motion`：运动预测评估
- `with_planning`：规划评估

最后汇总成统一结果字典并输出关键指标。

---

## 7. 部署与推理优化现状

## 7.1 仓内已有

- `tools/benchmark.py`：FLOPs/FPS/显存统计
- `tools/fuse_conv_bn.py`：卷积与 BN 融合
- `scripts/visualize.sh`：结果可视化

## 7.2 仓内缺失

- 无官方 ONNX 导出脚本
- 无 TensorRT 构建与推理脚本
- 无 MMDeploy 集成配置

结论：当前仓库更偏研究训练与评测实现，不是完整产品化部署工程。

---

## 8. CUDA Graph 检查结论

对仓库检索以下关键词无命中：

- `cuda graph`
- `CUDAGraph`
- `torch.cuda.graph`
- `make_graphed_callables`

结论：**SparseDrive 仓内未显式实现 CUDA Graph**。

---

## 9. 一句话总结

SparseDrive 通过“共享视觉特征 + 稀疏实例表示 + 多头协同 + 时序记忆队列”实现端到端多任务自动驾驶，但部署链路仍需在仓外或后续工程中补齐。

