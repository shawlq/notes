#pragma once

#include <vector>

#include "ai/common/model_config.h"
#include "ai/common/status.h"
#include "ai/runtime/device_runtime.h"

namespace ai {

class StreamPool {
 public:
  StreamPool();
  ~StreamPool();

  Status initialize(DeviceRuntime* runtime, const PipelineConfig& config);
  Status getStream(int stream_id, StreamHandle* stream) const;
  Status synchronizeAll(DeviceRuntime* runtime) const;
  void shutdown(DeviceRuntime* runtime);

 private:
  std::vector<StreamHandle> streams_;
};

}  // namespace ai
