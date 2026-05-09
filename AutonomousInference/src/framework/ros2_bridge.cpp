#include "ai/framework/ros2_bridge.h"

#include "ai/common/logger.h"

namespace ai {

Ros2Bridge::Ros2Bridge() : enabled_(false) {}

Status Ros2Bridge::initialize(const std::string& node_name) {
#if defined(AI_ENABLE_ROS2)
  enabled_ = true;
  AI_LOG_INFO("ros2") << "ROS2 bridge initialized: " << node_name;
#else
  enabled_ = false;
  AI_LOG_WARN("ros2") << "ROS2 bridge requested but AI_ENABLE_ROS2 is OFF: " << node_name;
#endif
  return Status::ok();
}

Status Ros2Bridge::publishTensorSummary(const TensorMap& outputs) {
  AI_LOG_INFO("ros2") << "publishing tensor summary, outputs=" << outputs.size();
  return Status::ok();
}

}  // namespace ai
