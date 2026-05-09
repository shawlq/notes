#include "ai/runtime/operator_runner.h"

#include <algorithm>

namespace ai {

OperatorRunner::OperatorRunner(DeviceRuntime* runtime) : runtime_(runtime) {}

Status OperatorRunner::convolution(const Tensor& input, Tensor* output, const StreamHandle&) const {
  if (output == nullptr) {
    return Status::invalidArgument("output must not be null");
  }
  output->data.assign(output->shape.elementCount(), 0.0F);
  for (std::size_t i = 0; i < output->data.size(); ++i) {
    output->data[i] = input.data.empty() ? 0.0F : input.data[i % input.data.size()] * 0.5F;
  }
  (void)runtime_;
  return Status::ok();
}

Status OperatorRunner::pooling(const Tensor& input, Tensor* output, const StreamHandle&) const {
  if (output == nullptr) {
    return Status::invalidArgument("output must not be null");
  }
  output->data.assign(output->shape.elementCount(), 0.0F);
  const float value = input.data.empty() ? 0.0F : *std::max_element(input.data.begin(), input.data.end());
  std::fill(output->data.begin(), output->data.end(), value);
  return Status::ok();
}

Status OperatorRunner::matrixMultiply(const Tensor& lhs, const Tensor& rhs, Tensor* output, const StreamHandle&) const {
  if (output == nullptr) {
    return Status::invalidArgument("output must not be null");
  }
  output->data.assign(output->shape.elementCount(), 0.0F);
  const float l = lhs.data.empty() ? 0.0F : lhs.data[0];
  const float r = rhs.data.empty() ? 0.0F : rhs.data[0];
  std::fill(output->data.begin(), output->data.end(), l * r);
  return Status::ok();
}

}  // namespace ai
