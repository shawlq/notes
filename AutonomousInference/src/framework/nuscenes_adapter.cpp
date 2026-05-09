#include "ai/framework/nuscenes_adapter.h"

namespace ai {

Tensor NuScenesAdapter::makeCameraTensor(const NuScenesFrame& frame, const TensorShape& shape) const {
  Tensor tensor("camera_images", shape);
  const float base = static_cast<float>(frame.camera_image_paths.size());
  for (std::size_t i = 0; i < tensor.data.size(); ++i) {
    tensor.data[i] = base + static_cast<float>(i % 255U) / 255.0F;
  }
  return tensor;
}

}  // namespace ai
