#include "ai/runtime/cuda_stream.h"

#include <algorithm>
#include <set>

namespace ai {

StreamPool::StreamPool() {}
StreamPool::~StreamPool() {}

Status StreamPool::initialize(DeviceRuntime* runtime, const PipelineConfig& config) {
  if (runtime == nullptr) {
    return Status::invalidArgument("runtime must not be null");
  }
  shutdown(runtime);
  std::set<int> stream_ids;
  for (std::vector<ModelConfig>::const_iterator it = config.models.begin(); it != config.models.end(); ++it) {
    stream_ids.insert(it->stream_id);
  }
  streams_.clear();
  for (std::set<int>::const_iterator it = stream_ids.begin(); it != stream_ids.end(); ++it) {
    StreamHandle stream;
    Status status = runtime->createStream(*it, &stream);
    if (!status) {
      return status;
    }
    streams_.push_back(stream);
  }
  return Status::ok();
}

Status StreamPool::getStream(int stream_id, StreamHandle* stream) const {
  if (stream == nullptr) {
    return Status::invalidArgument("stream must not be null");
  }
  for (std::vector<StreamHandle>::const_iterator it = streams_.begin(); it != streams_.end(); ++it) {
    if (it->id == stream_id) {
      *stream = *it;
      return Status::ok();
    }
  }
  return Status::notFound("stream not found");
}

Status StreamPool::synchronizeAll(DeviceRuntime* runtime) const {
  if (runtime == nullptr) {
    return Status::invalidArgument("runtime must not be null");
  }
  for (std::vector<StreamHandle>::const_iterator it = streams_.begin(); it != streams_.end(); ++it) {
    Status status = runtime->synchronize(*it);
    if (!status) {
      return status;
    }
  }
  return Status::ok();
}

void StreamPool::shutdown(DeviceRuntime* runtime) {
  if (runtime != nullptr) {
    for (std::vector<StreamHandle>::iterator it = streams_.begin(); it != streams_.end(); ++it) {
      runtime->destroyStream(&(*it));
    }
  }
  streams_.clear();
}

}  // namespace ai
