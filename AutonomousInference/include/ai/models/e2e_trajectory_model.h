#pragma once

#include "ai/models/base_model.h"

namespace ai {

class E2ETrajectoryModel : public BaseModel {
 public:
  Status infer(const TensorMap& inputs, TensorMap* outputs, const StreamHandle& stream) override;
  std::string modelKind() const override { return "E2E"; }
};

}  // namespace ai
