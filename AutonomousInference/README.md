# AutonomousInference

`AutonomousInference` 是一个面向自动驾驶多模型部署的 C++11 推理框架样例工程。工程以 BEV、GOD、LaneMark、E2E trajectory 四类模型为目标，提供统一模型接口、异步运行时抽象、CUDA Graph capture 编排、资源预算调度、nuScenes 风格输入适配和 ROS2 通信桥接占位。

## 目录结构

```text
AutonomousInference/
├── CMakeLists.txt
├── requirements.txt
├── config/sample_pipeline.json
├── docs/software_design.md
├── include/ai/
├── src/
│   ├── main.cpp
│   ├── common/
│   ├── framework/
│   ├── models/
│   ├── runtime/
│   └── utils/
├── scripts/run_demo.sh
└── tests/
```

## 架构概览

- `InferenceModel` 统一封装 `loadModel()`、`infer()`、`release()`，新增模型只需实现接口并注册到 `ModelRegistry`。
- `InferencePipeline` 负责配置加载、模型实例化、DAG 调度、预热、CUDA Graph capture 和帧级推理。
- `DeviceRuntime` 屏蔽 CPU stub 与 CUDA 后端差异；默认可在无 GPU 环境编译测试，启用 `AI_ENABLE_CUDA=ON` 后接入 CUDA runtime。
- `StreamPool`、`MemoryPool`、`CudaGraphExecutor` 分别管理 stream、预分配显存和 graph capture/launch。
- `Scheduler` 基于模型依赖 DAG 与 priority 生成执行顺序，支持 BEV fan-out 到 GOD/LaneMark，再进入 E2E trajectory。
- `Ros2Bridge` 提供 ROS2 接口边界；开启 `AI_ENABLE_ROS2=ON` 后由 CMake 发现 `rclcpp` 与消息包。

## 构建与运行

基础构建不依赖 CUDA/TensorRT/ROS2：

```bash
cmake -S AutonomousInference -B AutonomousInference/build -DAI_BUILD_TESTS=ON
cmake --build AutonomousInference/build -j$(nproc)
(cd AutonomousInference/build && ctest --output-on-failure)
./AutonomousInference/build/ai_demo AutonomousInference/config/sample_pipeline.json
```

GPU 环境构建示例：

```bash
cmake -S AutonomousInference -B AutonomousInference/build-gpu \
  -DAI_ENABLE_CUDA=ON \
  -DAI_ENABLE_TENSORRT=ON \
  -DAI_ENABLE_ROS2=ON \
  -DAI_BUILD_TESTS=ON
cmake --build AutonomousInference/build-gpu -j$(nproc)
```

## 生产化扩展点

1. 在 `BaseModel::loadModel()` 中接入 TensorRT engine 反序列化、binding 校验和 execution context 初始化。
2. 在各模型 `infer()` 中替换 CPU stub 输出为 TensorRT `enqueueV3()`，并绑定 `StreamHandle::native`。
3. 对固定 shape 场景执行一次 warmup 后 capture；动态 shape 场景每个 profile/shape 建立独立 execution context 与 graph。
4. 使用 `MemoryPool` 统一分配输入、输出、workspace 与 activation buffer，避免推理阶段频繁 `cudaMalloc/cudaFree`。
5. 用 `Ros2Bridge` 将 nuScenes/传感器输入映射为 ROS2 topics，并发布 objects、lane、trajectory 消息。

## 参考来源

- NVIDIA TensorRT Performance Optimization: https://docs.nvidia.com/deeplearning/tensorrt/latest/performance/optimization.html
- NVIDIA CUDA Graphs Programming Guide: https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#cuda-graphs
- Autoware `autoware_tensorrt_common`: https://autowarefoundation.github.io/autoware_universe/main/perception/autoware_tensorrt_common/
- Autoware BEVFormer / BEVFusion perception modules: https://autowarefoundation.github.io/autoware_universe/main/perception/autoware_tensorrt_bevformer/ ，https://autowarefoundation.github.io/autoware_universe/main/perception/autoware_bevfusion/
- Apollo: https://github.com/ApolloAuto/apollo
- Apollo Vision Net Deployment: https://github.com/ApolloAuto/Apollo-Vision-Net-Deployment
- nuScenes dataset: https://www.nuscenes.org/
