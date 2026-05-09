#pragma once

#include <functional>
#include <map>
#include <memory>
#include <string>

#include "ai/framework/model.h"

namespace ai {

class ModelRegistry {
 public:
  typedef std::function<std::unique_ptr<InferenceModel>()> Factory;

  static ModelRegistry& instance();

  void registerFactory(const std::string& type, Factory factory);
  std::unique_ptr<InferenceModel> create(const std::string& type) const;

 private:
  std::map<std::string, Factory> factories_;
};

void registerBuiltinModels();

}  // namespace ai
