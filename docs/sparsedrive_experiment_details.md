# 数据准备

## 数据下载， 实际只需要   "nuScenes-map-expansion-v1.3.zip"

```bash 
#!/usr/bin/env bash
# 从 Motional NuScenes 公开桶下载 v1.0 数据；可重复执行：已存在的文件跳过。

set -euo pipefail

DEST="${NUSCENES_DATA_DIR:-/media/c62664/DATA/sparsedrive/data}"
BUCKET_PREFIX="s3://motional-nuscenes/public/v1.0"
REGION="ap-northeast-1"

FILES=(
  "can_bus.zip"
  "md5.checksum"
  "nuScenes-map-expansion-v1.0.zip"
  "nuScenes-map-expansion-v1.1.zip"
  "nuScenes-map-expansion-v1.2.zip"
  "nuScenes-map-expansion-v1.3.zip"
  "v1.0-mini.tgz"
  "v1.0-test_blobs.tgz"
  "v1.0-test_blobs_camera.tgz"
  "v1.0-test_blobs_lidar.tgz"
  "v1.0-test_blobs_radar.tgz"
  "v1.0-test_meta.tgz"
  "v1.0-trainval01_blobs.tgz"
  "v1.0-trainval01_blobs_camera.tgz"
  "v1.0-trainval01_blobs_lidar.tgz"
  "v1.0-trainval01_blobs_radar.tgz"
  "v1.0-trainval01_keyframes.tgz"
  "v1.0-trainval02_blobs.tgz"
  "v1.0-trainval02_blobs_camera.tgz"
  "v1.0-trainval02_blobs_lidar.tgz"
  "v1.0-trainval02_blobs_radar.tgz"
  "v1.0-trainval02_keyframes.tgz"
  "v1.0-trainval03_blobs.tgz"
  "v1.0-trainval03_blobs_camera.tgz"
  "v1.0-trainval03_blobs_lidar.tgz"
  "v1.0-trainval03_blobs_radar.tgz"
  "v1.0-trainval03_keyframes.tgz"
  "v1.0-trainval04_blobs.tgz"
  "v1.0-trainval04_blobs_camera.tgz"
  "v1.0-trainval04_blobs_lidar.tgz"
  "v1.0-trainval04_blobs_radar.tgz"
  "v1.0-trainval04_keyframes.tgz"
  "v1.0-trainval05_blobs.tgz"
  "v1.0-trainval05_blobs_camera.tgz"
  "v1.0-trainval05_blobs_lidar.tgz"
  "v1.0-trainval05_blobs_radar.tgz"
  "v1.0-trainval05_keyframes.tgz"
  "v1.0-trainval06_blobs.tgz"
  "v1.0-trainval06_blobs_camera.tgz"
  "v1.0-trainval06_blobs_lidar.tgz"
  "v1.0-trainval06_blobs_radar.tgz"
  "v1.0-trainval06_keyframes.tgz"
  "v1.0-trainval07_blobs.tgz"
  "v1.0-trainval07_blobs_camera.tgz"
  "v1.0-trainval07_blobs_lidar.tgz"
  "v1.0-trainval07_blobs_radar.tgz"
  "v1.0-trainval07_keyframes.tgz"
  "v1.0-trainval08_blobs.tgz"
  "v1.0-trainval08_blobs_camera.tgz"
  "v1.0-trainval08_blobs_lidar.tgz"
  "v1.0-trainval08_blobs_radar.tgz"
  "v1.0-trainval08_keyframes.tgz"
  "v1.0-trainval09_blobs.tgz"
  "v1.0-trainval09_blobs_camera.tgz"
  "v1.0-trainval09_blobs_lidar.tgz"
  "v1.0-trainval09_blobs_radar.tgz"
  "v1.0-trainval09_keyframes.tgz"
  "v1.0-trainval10_blobs.tgz"
  "v1.0-trainval10_blobs_camera.tgz"
  "v1.0-trainval10_blobs_lidar.tgz"
  "v1.0-trainval10_blobs_radar.tgz"
  "v1.0-trainval10_keyframes.tgz"
  "v1.0-trainval_meta.tgz"
)

mkdir -p "${DEST}"

for name in "${FILES[@]}"; do
  dst="${DEST}/${name}"
  if [[ -f "${dst}" ]]; then
    echo "[skip] 已存在: ${dst}"
    continue
  fi
  echo "[get] ${name}"
  aws s3 cp --no-sign-request --region "${REGION}" \
    "${BUCKET_PREFIX}/${name}" "${dst}"
done

echo "完成。共 ${#FILES[@]} 个条目已检查/下载。"
```

## 数据解压

解压到 nuscenes,

### mini 

解压canbus 和 v1.0-mini，同时解压 nuScenes-map-expansion-v1.3.zip 到 nuscenes/maps下，会增加 basemap，expansion，prediction等目录

# 按步骤执行训练脚本

## 参考

url: https://github.com/swc-17/SparseDrive/blob/main/docs/quick_start.md

## 注意事项

### mini训练&验证
1. `sh scripts/create_data.sh ` 会报错，主要是数据不全，缺少map之类的，参考解压步骤

2. `sh scripts/kmeans.sh` 会报错 `ValueError: need at least one array to concatenate`，原因 mini数据集未生产

3. `sh scripts/train.sh` 会报错:

- import module 失败，需要添加路径 `import sys; sys.path.append("/home/c62664/workdir/gitcode/SparseDrive")`
- 没有 trainval，需要注释掉 `# version = 'trainval' `
- 主要是多进程抢占1个GPU异常 
- CXXABI_1.3.15 找不到  --- 不要直接安装影响系统库，在conda环境中解决，conda install libgcc-ng libstdcxx-ng，然后确认添加进去
```bash
echo $LD_LIBRARY_PATH
# 或者强制：
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
```

2和3的解决参考, **修改后本地能启动训练**：

```
commit e7bc973f2b427a6285ddc7b5989b9ff3173ab2b5 (HEAD -> liuchang/sandhill_test_0506)
Author: liuchang <chang.liu230863@seres.cn>
Date:   Thu May 7 14:28:47 2026 +0800

    适配单卡GPU

diff --git a/projects/configs/sparsedrive_small_stage1.py b/projects/configs/sparsedrive_small_stage1.py
index 1cbb38b..e54f62b 100644
--- a/projects/configs/sparsedrive_small_stage1.py
+++ b/projects/configs/sparsedrive_small_stage1.py
@@ -1,6 +1,6 @@
 # ================ base config ===================
 version = 'mini'
-version = 'trainval'
+# version = 'trainval'  # liuchang
 length = {'trainval': 28130, 'mini': 323}
 
 plugin = True
@@ -9,8 +9,9 @@ dist_params = dict(backend="nccl")
 log_level = "INFO"
 work_dir = None
 
-total_batch_size = 64
-num_gpus = 8
+# liuchang for single gpu — per-step batch matches one rank in the original 8-GPU setup (64/8).
+num_gpus = 1
+total_batch_size = 8 * num_gpus
 batch_size = total_batch_size // num_gpus
 num_iters_per_epoch = int(length[version] // (num_gpus * batch_size))
 num_epochs = 100
@@ -682,7 +683,7 @@ data = dict(
 # ================== training ========================
 optimizer = dict(
     type="AdamW",
-    lr=4e-4,
+    lr=5e-5,  # liuchang for single gpu: scaled from 4e-4 by (total_batch_size / 64)
     weight_decay=0.001,
     paramwise_cfg=dict(
         custom_keys={
diff --git a/projects/configs/sparsedrive_small_stage2.py b/projects/configs/sparsedrive_small_stage2.py
index 94cd085..a2398c5 100644
--- a/projects/configs/sparsedrive_small_stage2.py
+++ b/projects/configs/sparsedrive_small_stage2.py
@@ -9,8 +9,9 @@ dist_params = dict(backend="nccl")
 log_level = "INFO"
 work_dir = None
 
-total_batch_size = 48
-num_gpus = 8
+# liuchang for single gpu — per-GPU batch 6 (same as 48/8 in the 8-GPU setup).
+num_gpus = 1
+total_batch_size = 6 * num_gpus
 batch_size = total_batch_size // num_gpus
 num_iters_per_epoch = int(length[version] // (num_gpus * batch_size))
 num_epochs = 10
@@ -682,7 +683,7 @@ data = dict(
 # ================== training ========================
 optimizer = dict(
     type="AdamW",
-    lr=3e-4,
+    lr=3.75e-5,  # liuchang for single gpu: scaled from 3e-4 by (total_batch_size / 48)
     weight_decay=0.001,
     paramwise_cfg=dict(
         custom_keys={
diff --git a/scripts/test.sh b/scripts/test.sh
index c6861d2..caa27ae 100644
--- a/scripts/test.sh
+++ b/scripts/test.sh
@@ -1,7 +1,8 @@
+# liuchang for single gpu (GPUS=1 -> tools/dist_test.sh --nproc_per_node)
 bash ./tools/dist_test.sh \
     projects/configs/sparsedrive_small_stage2.py \
     ckpt/sparsedrive_stage2.pth \
-    8 \
+    1 \
     --deterministic \
     --eval bbox
     # --result_file ./work_dirs/sparsedrive_small_stage2/results.pkl
\ No newline at end of file
diff --git a/scripts/train.sh b/scripts/train.sh
index c1f6171..2665242 100644
--- a/scripts/train.sh
+++ b/scripts/train.sh
@@ -1,11 +1,13 @@
 ## stage1
+# liuchang for single gpu (GPUS=1 -> tools/dist_train.sh --nproc_per_node)
 bash ./tools/dist_train.sh \
    projects/configs/sparsedrive_small_stage1.py \
-   8 \
+   1 \
    --deterministic
 
 ## stage2
+# liuchang for single gpu (GPUS=1 -> tools/dist_train.sh --nproc_per_node)
 bash ./tools/dist_train.sh \
    projects/configs/sparsedrive_small_stage2.py \
-   8 \
+   1 \
    --deterministic
\ No newline at end of file
diff --git a/tools/kmeans/kmeans_plan.py b/tools/kmeans/kmeans_plan.py
index 33a74f7..c68ccae 100644
--- a/tools/kmeans/kmeans_plan.py
+++ b/tools/kmeans/kmeans_plan.py
@@ -25,7 +25,8 @@ for idx in tqdm(range(len(data_infos))):
     navi_trajs[cmd].append(plan_traj)
 
 clusters = []
-for trajs in navi_trajs:
+clusters.append(np.zeros((6,6,2))) # liuchang 打桩
+for trajs in navi_trajs[1:]: #liuchang 打桩
     trajs = np.concatenate(trajs, axis=0).reshape(-1, 12)
     cluster = KMeans(n_clusters=K).fit(trajs).cluster_centers_
     cluster = cluster.reshape(-1, 6, 2)
diff --git a/tools/train.py b/tools/train.py
index eef55ee..37ea0e9 100644
--- a/tools/train.py
+++ b/tools/train.py
@@ -24,6 +24,8 @@ from mmdet.apis import set_random_seed
 from torch import distributed as dist
 from datetime import timedelta
 
+import sys
+sys.path.append("/home/c62664/workdir/gitcode/SparseDrive")
 import cv2
 
 cv2.setNumThreads(8)
@@ -137,10 +139,8 @@ def main():
                 _module_dir = os.path.dirname(plugin_dir)
                 _module_dir = _module_dir.split("/")
                 _module_path = _module_dir[0]
-
                 for m in _module_dir[1:]:
                     _module_path = _module_path + "." + m
-                print(_module_path)
                 plg_lib = importlib.import_module(_module_path)
             else:
                 # import dir is the dirpath for the config file

```


# 其它

## 解压全量数据， 避免终端误关闭，最好后台运行

```bash

#!/usr/bin/env bash
# 将各 .tgz 解压到 tmp/<与归档同名（无 .tgz 后缀）>/ 目录下
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# 压缩包所在目录；默认与脚本同目录，例如：DATA_DIR=/path/to/archives ./sparsedrive_extract_data.sh
DATA_DIR="${DATA_DIR:-$ROOT}"
TMP="${ROOT}/tmp"
mkdir -p "${TMP}"

ARCHIVES=(
  "v1.0-trainval03_blobs_camera.tgz"
  "v1.0-trainval03_blobs_lidar.tgz"
  "v1.0-trainval03_blobs_radar.tgz"
  "v1.0-trainval03_blobs.tgz"
  "v1.0-trainval03_keyframes.tgz"
  "v1.0-trainval04_blobs_camera.tgz"
  "v1.0-trainval04_blobs_lidar.tgz"
  "v1.0-trainval04_blobs_radar.tgz"
  "v1.0-trainval04_blobs.tgz"
  "v1.0-trainval04_keyframes.tgz"
  "v1.0-trainval05_blobs_camera.tgz"
  "v1.0-trainval05_blobs_lidar.tgz"
  "v1.0-trainval05_blobs_radar.tgz"
  "v1.0-trainval05_blobs.tgz"
  "v1.0-trainval05_keyframes.tgz"
  "v1.0-trainval06_blobs_camera.tgz"
  "v1.0-trainval06_blobs_lidar.tgz"
  "v1.0-trainval06_blobs_radar.tgz"
  "v1.0-trainval06_blobs.tgz"
  "v1.0-trainval06_keyframes.tgz"
  "v1.0-trainval07_blobs_camera.tgz"
  "v1.0-trainval07_blobs_lidar.tgz"
  "v1.0-trainval07_blobs_radar.tgz"
  "v1.0-trainval07_blobs.tgz"
  "v1.0-trainval07_keyframes.tgz"
  "v1.0-trainval08_blobs_camera.tgz"
  "v1.0-trainval08_blobs_lidar.tgz"
  "v1.0-trainval08_blobs_radar.tgz"
  "v1.0-trainval08_blobs.tgz"
  "v1.0-trainval08_keyframes.tgz"
  "v1.0-trainval09_blobs_camera.tgz"
  "v1.0-trainval09_blobs_lidar.tgz"
  "v1.0-trainval09_blobs_radar.tgz"
  "v1.0-trainval09_blobs.tgz"
  "v1.0-trainval09_keyframes.tgz"
  "v1.0-trainval10_blobs_camera.tgz"
  "v1.0-trainval10_blobs_lidar.tgz"
  "v1.0-trainval10_blobs_radar.tgz"
  "v1.0-trainval10_blobs.tgz"
  "v1.0-trainval10_keyframes.tgz"
  "v1.0-trainval_meta.tgz"
)

for name in "${ARCHIVES[@]}"; do
  src="${DATA_DIR}/${name}"
  base="${name%.tgz}"
  dest="${TMP}/${base}"

  if [[ ! -f "${src}" ]]; then
    echo "跳过（文件不存在）: ${name}" >&2
    continue
  fi

  echo "解压: ${name} -> ${dest}/"
  mkdir -p "${dest}"
  tar -xzf "${src}" -C "${dest}"
done

echo "完成。"

```