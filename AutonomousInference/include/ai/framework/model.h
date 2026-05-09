#pragma once

#include <string>

#include "ai/common/model_config.h"
#include "ai/common/status.h"
#include "ai/common/tensor.h"
#include "ai/runtime/device_runtime.h"

namespace ai {

class InferenceModel {
 public:
  virtual ~InferenceModel() {}

  virtual Status loadModel(const ModelConfig& config, DeviceRuntime* runtime) = 0;
  virtual Status infer(const TensorMap& inputs, TensorMap* outputs, const StreamHandle& stream) = 0;
  virtual Status release() = 0;
  virtual const ModelConfig& config() const = 0;
  virtual std::string modelKind() const = 0;
};

}  // namespace ai
