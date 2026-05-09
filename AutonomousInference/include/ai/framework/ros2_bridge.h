#pragma once

#include <string>

#include "ai/common/status.h"
#include "ai/common/tensor.h"

namespace ai {

class Ros2Bridge {
 public:
  Ros2Bridge();

  Status initialize(const std::string& node_name);
  Status publishTensorSummary(const TensorMap& outputs);
  bool enabled() const { return enabled_; }

 private:
  bool enabled_;
};

}  // namespace ai
