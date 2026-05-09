#include "ai/runtime/cuda_graph.h"

#include "ai/common/logger.h"

#if defined(AI_ENABLE_CUDA)
#include <cuda_runtime_api.h>
#endif

namespace ai {

CudaGraphExecutor::CudaGraphExecutor()
    : captured_(false)
#if defined(AI_ENABLE_CUDA)
      , graph_(nullptr), graph_exec_(nullptr)
#endif
{
}

Status CudaGraphExecutor::capture(const std::string& name, const StreamHandle& stream, const std::function<Status()>& work) {
  name_ = name;
#if defined(AI_ENABLE_CUDA)
  cudaStream_t cuda_stream = reinterpret_cast<cudaStream_t>(stream.native);
  cudaGraph_t graph = nullptr;
  cudaGraphExec_t graph_exec = nullptr;
  cudaError_t err = cudaStreamBeginCapture(cuda_stream, cudaStreamCaptureModeGlobal);
  if (err != cudaSuccess) {
    return Status::runtimeError(cudaGetErrorString(err));
  }
  Status status = work();
  if (!status) {
    cudaStreamEndCapture(cuda_stream, &graph);
    return status;
  }
  err = cudaStreamEndCapture(cuda_stream, &graph);
  if (err != cudaSuccess) {
    AI_LOG_WARN("cuda_graph") << "capture failed for " << name << ", fallback to normal enqueue: " << cudaGetErrorString(err);
    captured_ = false;
    return Status::ok();
  }
  err = cudaGraphInstantiate(&graph_exec, graph, nullptr, nullptr, 0);
  if (err != cudaSuccess) {
    cudaGraphDestroy(graph);
    return Status::runtimeError(cudaGetErrorString(err));
  }
  graph_ = graph;
  graph_exec_ = graph_exec;
  captured_ = true;
  AI_LOG_INFO("cuda_graph") << "captured graph " << name;
#else
  (void)stream;
  Status status = work();
  if (!status) {
    return status;
  }
  captured_ = true;
  AI_LOG_INFO("cuda_graph") << "CPU graph stub captured " << name;
#endif
  return Status::ok();
}

Status CudaGraphExecutor::launch(const StreamHandle& stream, const std::function<Status()>& fallback_work) {
#if defined(AI_ENABLE_CUDA)
  if (captured_) {
    cudaError_t err = cudaGraphLaunch(reinterpret_cast<cudaGraphExec_t>(graph_exec_), reinterpret_cast<cudaStream_t>(stream.native));
    return err == cudaSuccess ? Status::ok() : Status::runtimeError(cudaGetErrorString(err));
  }
#else
  (void)stream;
#endif
  return fallback_work();
}

void CudaGraphExecutor::reset() {
#if defined(AI_ENABLE_CUDA)
  if (graph_exec_ != nullptr) {
    cudaGraphExecDestroy(reinterpret_cast<cudaGraphExec_t>(graph_exec_));
    graph_exec_ = nullptr;
  }
  if (graph_ != nullptr) {
    cudaGraphDestroy(reinterpret_cast<cudaGraph_t>(graph_));
    graph_ = nullptr;
  }
#endif
  captured_ = false;
}

}  // namespace ai
