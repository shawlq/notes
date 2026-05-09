#pragma once

#include <cstddef>
#include <map>
#include <memory>
#include <vector>

#include "ai/common/status.h"
#include "ai/runtime/device_runtime.h"

namespace ai {

class MemoryPool {
 public:
  MemoryPool();
  ~MemoryPool();

  Status initialize(DeviceRuntime* runtime, std::size_t bytes);
  Status allocate(std::size_t bytes, void** ptr);
  void reset();
  void shutdown();
  std::size_t capacity() const { return capacity_; }
  std::size_t used() const { return offset_; }

 private:
  DeviceRuntime* runtime_;
  void* base_;
  std::size_t capacity_;
  std::size_t offset_;
};

}  // namespace ai
