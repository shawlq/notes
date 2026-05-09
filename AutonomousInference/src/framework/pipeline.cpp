#include "ai/framework/pipeline.h"

#include <algorithm>

#include "ai/common/logger.h"
#include "ai/framework/model_registry.h"
#include "ai/utils/timer.h"

namespace ai {

InferencePipeline::InferencePipeline() : initialized_(false), graph_ready_(false) {}
InferencePipeline::~InferencePipeline() { release(); }

Status InferencePipeline::initializeFromFile(const std::string& config_path) {
  PipelineConfig config;
  ConfigLoader loader;
  Status status = loader.loadFromFile(config_path, &config);
  if (!status) {
    return status;
  }
  return initialize(config);
}

Status InferencePipeline::initialize(const PipelineConfig& config) {
  release();
  config_ = config;
  registerBuiltinModels();
  runtime_ = createDefaultRuntime();
  Status status = runtime_->initialize(config_.runtime);
  if (!status) {
    return status;
  }
  status = memory_pool_.initialize(runtime_.get(), config_.runtime.memory_pool_bytes);
  if (!status) {
    return status;
  }
  status = streams_.initialize(runtime_.get(), config_);
  if (!status) {
    return status;
  }
  status = scheduler_.build(config_);
  if (!status) {
    return status;
  }
  for (std::vector<ModelConfig>::const_iterator it = config_.models.begin(); it != config_.models.end(); ++it) {
    std::unique_ptr<InferenceModel> model = ModelRegistry::instance().create(it->type);
    if (!model) {
      return Status::invalidArgument("unknown model type: " + it->type);
    }
    status = model->loadModel(*it, runtime_.get());
    if (!status) {
      return status;
    }
    models_.push_back(std::move(model));
  }
  initialized_ = true;
  AI_LOG_INFO("pipeline") << "initialized with runtime=" << runtime_->name() << ", models=" << models_.size();
  return Status::ok();
}

Status InferencePipeline::inferFrame(const TensorMap& frame_inputs, TensorMap* frame_outputs) {
  if (!initialized_) {
    return Status::runtimeError("pipeline is not initialized");
  }
  if (frame_outputs == nullptr) {
    return Status::invalidArgument("frame_outputs must not be null");
  }
  if (config_.runtime.enable_cuda_graph && !graph_ready_) {
    Status status = warmupAndCapture(frame_inputs);
    if (!status) {
      return status;
    }
  }
  if (graph_ready_) {
    StreamHandle stream;
    Status status = streams_.getStream(config_.models.empty() ? 0 : config_.models.front().stream_id, &stream);
    if (!status) {
      return status;
    }
    return graph_.launch(stream, [&]() { return runOnce(frame_inputs, frame_outputs); });
  }
  return runOnce(frame_inputs, frame_outputs);
}

Status InferencePipeline::runOnce(const TensorMap& frame_inputs, TensorMap* frame_outputs) {
  ScopedTimer timer("pipeline_frame");
  TensorMap tensors = frame_inputs;
  frame_outputs->clear();
  for (std::vector<ScheduleStep>::const_iterator it = scheduler_.steps().begin(); it != scheduler_.steps().end(); ++it) {
    InferenceModel* model = findModel(it->model_name);
    if (model == nullptr) {
      return Status::notFound("model not found: " + it->model_name);
    }
    StreamHandle stream;
    Status status = streams_.getStream(it->stream_id, &stream);
    if (!status) {
      return status;
    }
    TensorMap model_outputs;
    status = model->infer(tensors, &model_outputs, stream);
    if (!status) {
      return status;
    }
    tensors.insert(tensors.end(), model_outputs.begin(), model_outputs.end());
    frame_outputs->insert(frame_outputs->end(), model_outputs.begin(), model_outputs.end());
  }
  return streams_.synchronizeAll(runtime_.get());
}

Status InferencePipeline::warmupAndCapture(const TensorMap& frame_inputs) {
  TensorMap outputs;
  for (int i = 0; i < std::max(1, config_.runtime.warmup_runs); ++i) {
    Status status = runOnce(frame_inputs, &outputs);
    if (!status) {
      return status;
    }
  }
  StreamHandle stream;
  Status status = streams_.getStream(config_.models.empty() ? 0 : config_.models.front().stream_id, &stream);
  if (!status) {
    return status;
  }
  status = graph_.capture("full_pipeline", stream, [&]() { return runOnce(frame_inputs, &outputs); });
  if (!status) {
    return status;
  }
  graph_ready_ = graph_.captured();
  return Status::ok();
}

InferenceModel* InferencePipeline::findModel(const std::string& name) const {
  for (std::vector<std::unique_ptr<InferenceModel> >::const_iterator it = models_.begin(); it != models_.end(); ++it) {
    if ((*it)->config().name == name) {
      return it->get();
    }
  }
  return nullptr;
}

Status InferencePipeline::release() {
  for (std::vector<std::unique_ptr<InferenceModel> >::iterator it = models_.begin(); it != models_.end(); ++it) {
    (*it)->release();
  }
  models_.clear();
  graph_.reset();
  streams_.shutdown(runtime_.get());
  memory_pool_.shutdown();
  initialized_ = false;
  graph_ready_ = false;
  return Status::ok();
}

}  // namespace ai
