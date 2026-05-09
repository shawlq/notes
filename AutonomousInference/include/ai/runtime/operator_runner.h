#pragma once

#include "ai/common/status.h"
#include "ai/common/tensor.h"
#include "ai/runtime/device_runtime.h"

namespace ai {

class OperatorRunner {
 public:
  explicit OperatorRunner(DeviceRuntime* runtime);

  Status convolution(const Tensor& input, Tensor* output, const StreamHandle& stream) const;
  Status pooling(const Tensor& input, Tensor* output, const StreamHandle& stream) const;
  Status matrixMultiply(const Tensor& lhs, const Tensor& rhs, Tensor* output, const StreamHandle& stream) const;

 private:
  DeviceRuntime* runtime_;
};

}  // namespace ai
