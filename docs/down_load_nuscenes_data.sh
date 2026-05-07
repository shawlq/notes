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
