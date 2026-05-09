#pragma once

#include "ai/models/base_model.h"

namespace ai {

class LaneMarkModel : public BaseModel {
 public:
  Status infer(const TensorMap& inputs, TensorMap* outputs, const StreamHandle& stream) override;
  std::string modelKind() const override { return "LaneMark"; }
};

}  // namespace ai
