#include <stdexcept>

#include "ai/common/config.h"
#include "ai/framework/nuscenes_adapter.h"
#include "ai/framework/pipeline.h"

namespace {
void require(bool condition, const char* message) {
  if (!condition) {
    throw std::runtime_error(message);
  }
}
}

void testPipelineRuns() {
  ai::PipelineConfig config;
  ai::ConfigLoader loader;
  ai::Status status = loader.loadFromFile("../config/sample_pipeline.json", &config);
  require(status.okStatus(), status.message().c_str());

  ai::InferencePipeline pipeline;
  status = pipeline.initialize(config);
  require(status.okStatus(), status.message().c_str());
  require(pipeline.schedule().size() == 4U, "schedule size mismatch");
  require(pipeline.schedule()[0].model_name == "bev", "bev must run first");

  ai::NuScenesFrame frame;
  frame.sample_token = "sample";
  frame.camera_image_paths.assign(6U, "camera.jpg");
  ai::TensorShape shape;
  shape.dims = config.models[0].input_shape;
  ai::NuScenesAdapter adapter;
  ai::TensorMap inputs;
  inputs.push_back(adapter.makeCameraTensor(frame, shape));

  ai::TensorMap outputs;
  status = pipeline.inferFrame(inputs, &outputs);
  require(status.okStatus(), status.message().c_str());
  require(outputs.size() == 4U, "expected one output per model");
  require(outputs.back().name == "trajectory", "trajectory output missing");
}
