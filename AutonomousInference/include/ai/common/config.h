#pragma once

#include <string>

#include "ai/common/model_config.h"
#include "ai/common/status.h"

namespace ai {

class ConfigLoader {
 public:
  Status loadFromFile(const std::string& path, PipelineConfig* config) const;
  Status loadFromString(const std::string& text, PipelineConfig* config) const;
};

}  // namespace ai
