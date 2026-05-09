#pragma once

#include <map>
#include <memory>
#include <string>
#include <vector>

#include "ai/common/config.h"
#include "ai/framework/model.h"
#include "ai/framework/scheduler.h"
#include "ai/runtime/cuda_graph.h"
#include "ai/runtime/cuda_stream.h"
#include "ai/runtime/device_runtime.h"
#include "ai/runtime/memory_pool.h"

namespace ai {

class InferencePipeline {
 public:
  InferencePipeline();
  ~InferencePipeline();

  Status initialize(const PipelineConfig& config);
  Status initializeFromFile(const std::string& config_path);
  Status inferFrame(const TensorMap& frame_inputs, TensorMap* frame_outputs);
  Status release();
  const std::vector<ScheduleStep>& schedule() const { return scheduler_.steps(); }

 private:
  Status runOnce(const TensorMap& frame_inputs, TensorMap* frame_outputs);
  Status warmupAndCapture(const TensorMap& frame_inputs);
  InferenceModel* findModel(const std::string& name) const;

  PipelineConfig config_;
  std::unique_ptr<DeviceRuntime> runtime_;
  StreamPool streams_;
  MemoryPool memory_pool_;
  Scheduler scheduler_;
  std::vector<std::unique_ptr<InferenceModel> > models_;
  CudaGraphExecutor graph_;
  bool initialized_;
  bool graph_ready_;
};

}  // namespace ai
