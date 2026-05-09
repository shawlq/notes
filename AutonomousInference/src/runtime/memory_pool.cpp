#include "ai/runtime/memory_pool.h"

#include <cstdint>

namespace ai {

MemoryPool::MemoryPool() : runtime_(nullptr), base_(nullptr), capacity_(0U), offset_(0U) {}

MemoryPool::~MemoryPool() {
  shutdown();
}

Status MemoryPool::initialize(DeviceRuntime* runtime, std::size_t bytes) {
  shutdown();
  runtime_ = runtime;
  capacity_ = bytes;
  offset_ = 0U;
  if (bytes == 0U) {
    return Status::ok();
  }
  return runtime_->allocate(bytes, &base_);
}

Status MemoryPool::allocate(std::size_t bytes, void** ptr) {
  if (ptr == nullptr) {
    return Status::invalidArgument("ptr must not be null");
  }
  const std::size_t aligned = (bytes + 255U) & ~static_cast<std::size_t>(255U);
  if (offset_ + aligned > capacity_) {
    return Status::runtimeError("memory pool exhausted");
  }
  *ptr = static_cast<void*>(static_cast<unsigned char*>(base_) + offset_);
  offset_ += aligned;
  return Status::ok();
}

void MemoryPool::reset() {
  offset_ = 0U;
}

void MemoryPool::shutdown() {
  if (runtime_ != nullptr && base_ != nullptr) {
    runtime_->release(base_);
  }
  runtime_ = nullptr;
  base_ = nullptr;
  capacity_ = 0U;
  offset_ = 0U;
}

}  // namespace ai
