#pragma once

#include <cstddef>
#include <memory>
#include <string>

#include "ai/common/model_config.h"
#include "ai/common/status.h"

namespace ai {

struct StreamHandle {
  StreamHandle() : id(0), native(nullptr) {}
  int id;
  void* native;
};

class DeviceRuntime {
 public:
  virtual ~DeviceRuntime() {}

  virtual Status initialize(const RuntimeConfig& config) = 0;
  virtual Status createStream(int stream_id, StreamHandle* stream) = 0;
  virtual void destroyStream(StreamHandle* stream) = 0;
  virtual Status synchronize(const StreamHandle& stream) = 0;
  virtual Status allocate(std::size_t bytes, void** ptr) = 0;
  virtual void release(void* ptr) = 0;
  virtual std::string name() const = 0;
};

std::unique_ptr<DeviceRuntime> createDefaultRuntime();

}  // namespace ai
