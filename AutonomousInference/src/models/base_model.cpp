#include "ai/models/base_model.h"

#include <algorithm>

#include "ai/common/logger.h"

namespace ai {

BaseModel::BaseModel() : runtime_(nullptr), loaded_(false) {}
BaseModel::~BaseModel() {}

Status BaseModel::loadModel(const ModelConfig& config, DeviceRuntime* runtime) {
  if (runtime == nullptr) {
    return Status::invalidArgument("runtime must not be null");
  }
  config_ = config;
  runtime_ = runtime;
  loaded_ = true;
  AI_LOG_INFO("model") << "loaded " << config.name << " type=" << config.type << " engine=" << config.engine_path;
  return Status::ok();
}

Status BaseModel::release() {
  loaded_ = false;
  return Status::ok();
}

Status BaseModel::validateInput(const TensorMap& inputs) const {
  if (!loaded_) {
    return Status::runtimeError("model is not loaded: " + config_.name);
  }
  for (TensorMap::const_iterator it = inputs.begin(); it != inputs.end(); ++it) {
    if (it->name == config_.input_tensor) {
      return Status::ok();
    }
  }
  return Status::invalidArgument("missing tensor: " + config_.input_tensor);
}

Status BaseModel::writeOutput(TensorMap* outputs, float seed) const {
  if (outputs == nullptr) {
    return Status::invalidArgument("outputs must not be null");
  }
  TensorShape shape;
  shape.dims = config_.output_shape;
  Tensor tensor(config_.output_tensor, shape);
  for (std::size_t i = 0; i < tensor.data.size(); ++i) {
    tensor.data[i] = seed + static_cast<float>(i % 17U) * 0.001F;
  }
  outputs->push_back(tensor);
  return Status::ok();
}

}  // namespace ai
