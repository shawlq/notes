#pragma once

#include <map>
#include <string>
#include <vector>

#include "ai/common/model_config.h"
#include "ai/common/status.h"

namespace ai {

struct ScheduleStep {
  std::string model_name;
  int priority;
  int stream_id;
};

class Scheduler {
 public:
  Status build(const PipelineConfig& config);
  const std::vector<ScheduleStep>& steps() const { return steps_; }

 private:
  std::vector<ScheduleStep> steps_;
};

}  // namespace ai
