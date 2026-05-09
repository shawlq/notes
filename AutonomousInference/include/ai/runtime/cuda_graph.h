#pragma once

#include <functional>
#include <string>

#include "ai/common/status.h"
#include "ai/runtime/device_runtime.h"

namespace ai {

class CudaGraphExecutor {
 public:
  CudaGraphExecutor();

  Status capture(const std::string& name, const StreamHandle& stream, const std::function<Status()>& work);
  Status launch(const StreamHandle& stream, const std::function<Status()>& fallback_work);
  void reset();
  bool captured() const { return captured_; }

 private:
  std::string name_;
  bool captured_;
#if defined(AI_ENABLE_CUDA)
  void* graph_;
  void* graph_exec_;
#endif
};

}  // namespace ai
