cmake_minimum_required(VERSION 3.16)
project(AutonomousInference VERSION 0.1.0 LANGUAGES CXX)

option(AI_ENABLE_CUDA "Enable CUDA runtime backend" OFF)
option(AI_ENABLE_TENSORRT "Enable TensorRT backend hooks" OFF)
option(AI_ENABLE_ROS2 "Enable ROS2 bridge" OFF)
option(AI_BUILD_TESTS "Build unit and integration tests" ON)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

add_library(autonomous_inference
  src/common/config.cpp
  src/common/logger.cpp
  src/framework/model_registry.cpp
  src/framework/nuscenes_adapter.cpp
  src/framework/pipeline.cpp
  src/framework/ros2_bridge.cpp
  src/framework/scheduler.cpp
  src/models/base_model.cpp
  src/models/bev_model.cpp
  src/models/e2e_trajectory_model.cpp
  src/models/god_model.cpp
  src/models/lane_mark_model.cpp
  src/runtime/cuda_graph.cpp
  src/runtime/cuda_stream.cpp
  src/runtime/device_runtime.cpp
  src/runtime/memory_pool.cpp
  src/runtime/operator_runner.cpp
  src/utils/timer.cpp
)

target_include_directories(autonomous_inference PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
)

target_compile_options(autonomous_inference PRIVATE -Wall -Wextra -Wpedantic)

if(AI_ENABLE_CUDA)
  enable_language(CUDA)
  find_package(CUDAToolkit REQUIRED)
  target_compile_definitions(autonomous_inference PUBLIC AI_ENABLE_CUDA=1)
  target_link_libraries(autonomous_inference PUBLIC CUDA::cudart CUDA::cublas)
endif()

if(AI_ENABLE_TENSORRT)
  find_path(TENSORRT_INCLUDE_DIR NvInfer.h)
  find_library(TENSORRT_LIBRARY nvinfer)
  find_library(TENSORRT_PARSER_LIBRARY nvonnxparser)
  if(NOT TENSORRT_INCLUDE_DIR OR NOT TENSORRT_LIBRARY)
    message(FATAL_ERROR "TensorRT was requested but NvInfer.h or libnvinfer was not found")
  endif()
  target_compile_definitions(autonomous_inference PUBLIC AI_ENABLE_TENSORRT=1)
  target_include_directories(autonomous_inference PUBLIC ${TENSORRT_INCLUDE_DIR})
  target_link_libraries(autonomous_inference PUBLIC ${TENSORRT_LIBRARY} ${TENSORRT_PARSER_LIBRARY})
endif()

if(AI_ENABLE_ROS2)
  find_package(ament_cmake REQUIRED)
  find_package(rclcpp REQUIRED)
  find_package(sensor_msgs REQUIRED)
  find_package(nav_msgs REQUIRED)
  find_package(geometry_msgs REQUIRED)
  target_compile_definitions(autonomous_inference PUBLIC AI_ENABLE_ROS2=1)
  ament_target_dependencies(autonomous_inference rclcpp sensor_msgs nav_msgs geometry_msgs)
endif()

add_executable(ai_demo src/main.cpp)
target_link_libraries(ai_demo PRIVATE autonomous_inference)

if(AI_BUILD_TESTS)
  enable_testing()
  add_executable(ai_tests
    tests/test_config.cpp
    tests/test_pipeline.cpp
    tests/test_main.cpp
  )
  target_link_libraries(ai_tests PRIVATE autonomous_inference)
  add_test(NAME autonomous_inference_tests COMMAND ai_tests)
endif()

install(TARGETS autonomous_inference ai_demo
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION bin
)
install(DIRECTORY include/ DESTINATION include)
