#pragma once

#include <cstddef>
#include <string>
#include <vector>

namespace ai {

struct RuntimeConfig {
  RuntimeConfig()
      : device_id(0), enable_cuda_graph(false), warmup_runs(1), memory_pool_bytes(0U), max_concurrent_models(1) {}

  int device_id;
  bool enable_cuda_graph;
  int warmup_runs;
  std::size_t memory_pool_bytes;
  int max_concurrent_models;
};

struct ModelConfig {
  ModelConfig()
      : priority(0), stream_id(0), estimated_workspace_bytes(0U) {}

  std::string name;
  std::string type;
  std::string engine_path;
  std::string input_tensor;
  std::string output_tensor;
  int priority;
  int stream_id;
  std::size_t estimated_workspace_bytes;
  std::vector<int> input_shape;
  std::vector<int> output_shape;
};

struct PipelineEdge {
  std::string from;
  std::string to;
};

struct PipelineConfig {
  RuntimeConfig runtime;
  std::vector<ModelConfig> models;
  std::vector<PipelineEdge> edges;
};

}  // namespace ai
