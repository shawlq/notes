#include "ai/runtime/device_runtime.h"

#include <cstdlib>
#include <memory>

#include "ai/common/logger.h"

#if defined(AI_ENABLE_CUDA)
#include <cuda_runtime_api.h>
#endif

namespace ai {

namespace {

class CpuRuntime : public DeviceRuntime {
 public:
  Status initialize(const RuntimeConfig& config) override {
    AI_LOG_INFO("runtime") << "CPU stub runtime initialized, device_id=" << config.device_id;
    return Status::ok();
  }

  Status createStream(int stream_id, StreamHandle* stream) override {
    if (stream == nullptr) {
      return Status::invalidArgument("stream must not be null");
    }
    stream->id = stream_id;
    stream->native = nullptr;
    return Status::ok();
  }

  void destroyStream(StreamHandle* stream) override {
    if (stream != nullptr) {
      stream->native = nullptr;
    }
  }

  Status synchronize(const StreamHandle&) override {
    return Status::ok();
  }

  Status allocate(std::size_t bytes, void** ptr) override {
    if (ptr == nullptr) {
      return Status::invalidArgument("ptr must not be null");
    }
    *ptr = std::malloc(bytes == 0U ? 1U : bytes);
    return *ptr == nullptr ? Status::runtimeError("CPU allocation failed") : Status::ok();
  }

  void release(void* ptr) override {
    std::free(ptr);
  }

  std::string name() const override { return "cpu-stub"; }
};

#if defined(AI_ENABLE_CUDA)
class CudaRuntime : public DeviceRuntime {
 public:
  Status initialize(const RuntimeConfig& config) override {
    cudaError_t err = cudaSetDevice(config.device_id);
    if (err != cudaSuccess) {
      return Status::runtimeError(cudaGetErrorString(err));
    }
    AI_LOG_INFO("runtime") << "CUDA runtime initialized, device_id=" << config.device_id;
    return Status::ok();
  }

  Status createStream(int stream_id, StreamHandle* stream) override {
    if (stream == nullptr) {
      return Status::invalidArgument("stream must not be null");
    }
    cudaStream_t cuda_stream = nullptr;
    cudaError_t err = cudaStreamCreateWithFlags(&cuda_stream, cudaStreamNonBlocking);
    if (err != cudaSuccess) {
      return Status::runtimeError(cudaGetErrorString(err));
    }
    stream->id = stream_id;
    stream->native = cuda_stream;
    return Status::ok();
  }

  void destroyStream(StreamHandle* stream) override {
    if (stream != nullptr && stream->native != nullptr) {
      cudaStreamDestroy(reinterpret_cast<cudaStream_t>(stream->native));
      stream->native = nullptr;
    }
  }

  Status synchronize(const StreamHandle& stream) override {
    cudaError_t err = cudaStreamSynchronize(reinterpret_cast<cudaStream_t>(stream.native));
    return err == cudaSuccess ? Status::ok() : Status::runtimeError(cudaGetErrorString(err));
  }

  Status allocate(std::size_t bytes, void** ptr) override {
    cudaError_t err = cudaMalloc(ptr, bytes == 0U ? 1U : bytes);
    return err == cudaSuccess ? Status::ok() : Status::runtimeError(cudaGetErrorString(err));
  }

  void release(void* ptr) override {
    if (ptr != nullptr) {
      cudaFree(ptr);
    }
  }

  std::string name() const override { return "cuda"; }
};
#endif

}  // namespace

std::unique_ptr<DeviceRuntime> createDefaultRuntime() {
#if defined(AI_ENABLE_CUDA)
  return std::unique_ptr<DeviceRuntime>(new CudaRuntime());
#else
  return std::unique_ptr<DeviceRuntime>(new CpuRuntime());
#endif
}

}  // namespace ai
