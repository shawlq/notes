`results.pkl` 里保存的是**逐帧预测结果**，不是原始图像。  
按当前 `sparsedrive_small_stage2.py`（检测+地图+运动+规划都开）来看，结构是：

- **顶层**：`list`，长度=评测帧数（mini 通常 323）
- **每个元素**：`{"img_bbox": result_dict}`
- **`result_dict`** 是多任务结果合并后的字典（由 `SparseDriveHead.post_process` 合并）

主要字段如下：

- **检测/跟踪相关**
  - `boxes_3d`：3D 框（解码后，含位置尺寸朝向速度）
  - `scores_3d`：框置信度
  - `labels_3d`：类别 id
  - `instance_ids`：跟踪 id（有跟踪时）
  - `cls_scores`：可选，原始分类分数（有 quality 分支时）

- **地图向量相关**
  - `vectors`：每条地图线的点序列（list of numpy array）
  - `scores`：每条线分数
  - `labels`：每条线类别

- **运动预测相关**
  - `trajs_3d`：目标多模态未来轨迹
  - `trajs_score`：各轨迹模式分数
  - `anchor_queue`：时序队列中的历史锚框
  - `period`：对应时序间隔/周期信息

- **自车规划相关**
  - `planning_score`：规划模式分数
  - `planning`：候选规划轨迹
  - `final_planning`：最终选择的规划轨迹
  - `ego_period`：自车时序周期信息
  - `ego_anchor_queue`：自车历史锚框队列

补充两点：

- `results.pkl` **不包含原始图片像素**，可视化时会再通过数据集索引去读原图并叠加预测。
- 你现在的视频脚本主要用了其中一部分（如 `boxes_3d/scores_3d/labels_3d`），但 `pkl` 里通常比视频展示的信息更多。