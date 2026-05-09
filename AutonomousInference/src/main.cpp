#include <iostream>
#include <string>

#include "ai/common/config.h"
#include "ai/framework/nuscenes_adapter.h"
#include "ai/framework/pipeline.h"

int main(int argc, char** argv) {
  const std::string config_path = argc > 1 ? argv[1] : "config/sample_pipeline.json";
  ai::PipelineConfig config;
  ai::ConfigLoader loader;
  ai::Status status = loader.loadFromFile(config_path, &config);
  if (!status) {
    std::cerr << "failed to load config: " << status.message() << std::endl;
    return 1;
  }

  ai::InferencePipeline pipeline;
  status = pipeline.initialize(config);
  if (!status) {
    std::cerr << "failed to initialize pipeline: " << status.message() << std::endl;
    return 2;
  }

  ai::TensorShape camera_shape;
  camera_shape.dims = config.models.empty() ? std::vector<int>{1, 6, 3, 256, 704} : config.models.front().input_shape;
  ai::NuScenesFrame frame;
  frame.sample_token = "demo-sample";
  frame.timestamp_sec = 0.0;
  frame.camera_image_paths.assign(6U, "camera.jpg");

  ai::NuScenesAdapter adapter;
  ai::TensorMap inputs;
  inputs.push_back(adapter.makeCameraTensor(frame, camera_shape));

  ai::TensorMap outputs;
  status = pipeline.inferFrame(inputs, &outputs);
  if (!status) {
    std::cerr << "failed to run inference: " << status.message() << std::endl;
    return 3;
  }

  std::cout << "Inference finished. Schedule=" << pipeline.schedule().size() << ", outputs=" << outputs.size() << std::endl;
  for (ai::TensorMap::const_iterator it = outputs.begin(); it != outputs.end(); ++it) {
    std::cout << " - " << it->name << " shape=" << it->shape.toString() << " elements=" << it->data.size() << std::endl;
  }
  return 0;
}
