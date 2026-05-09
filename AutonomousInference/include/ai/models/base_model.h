#pragma once

#include <string>

#include "ai/framework/model.h"

namespace ai {

class BaseModel : public InferenceModel {
 public:
  BaseModel();
  virtual ~BaseModel();

  Status loadModel(const ModelConfig& config, DeviceRuntime* runtime) override;
  Status release() override;
  const ModelConfig& config() const override { return config_; }

 protected:
  Status validateInput(const TensorMap& inputs) const;
  Status writeOutput(TensorMap* outputs, float seed) const;

  ModelConfig config_;
  DeviceRuntime* runtime_;
  bool loaded_;
};

}  // namespace ai
