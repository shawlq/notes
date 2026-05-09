#pragma once

#include <string>
#include <vector>

#include "ai/common/tensor.h"

namespace ai {

struct NuScenesFrame {
  std::string sample_token;
  double timestamp_sec;
  std::vector<std::string> camera_image_paths;
  std::string lidar_path;
};

class NuScenesAdapter {
 public:
  Tensor makeCameraTensor(const NuScenesFrame& frame, const TensorShape& shape) const;
};

}  // namespace ai
