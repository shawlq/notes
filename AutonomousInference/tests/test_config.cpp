#include <stdexcept>

#include "ai/common/config.h"

namespace {
void require(bool condition, const char* message) {
  if (!condition) {
    throw std::runtime_error(message);
  }
}
}

void testConfigLoader() {
  const std::string json =
      "{\"runtime\":{\"device_id\":0,\"enable_cuda_graph\":false,\"warmup_runs\":1,\"memory_pool_bytes\":1024,\"max_concurrent_models\":1},"
      "\"models\":[{\"name\":\"bev\",\"type\":\"BEV\",\"engine_path\":\"bev.plan\",\"input_tensor\":\"camera_images\",\"output_tensor\":\"bev_features\",\"priority\":0,\"stream_id\":0,\"estimated_workspace_bytes\":64,\"input_shape\":[1,2],\"output_shape\":[1,3]}],"
      "\"edges\":[]}";
  ai::PipelineConfig config;
  ai::ConfigLoader loader;
  ai::Status status = loader.loadFromString(json, &config);
  require(status.okStatus(), status.message().c_str());
  require(config.models.size() == 1U, "model size mismatch");
  require(config.models[0].output_shape[1] == 3, "shape mismatch");
}
