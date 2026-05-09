#include "ai/framework/model_registry.h"

#include "ai/models/bev_model.h"
#include "ai/models/e2e_trajectory_model.h"
#include "ai/models/god_model.h"
#include "ai/models/lane_mark_model.h"

namespace ai {

ModelRegistry& ModelRegistry::instance() {
  static ModelRegistry registry;
  return registry;
}

void ModelRegistry::registerFactory(const std::string& type, Factory factory) {
  factories_[type] = factory;
}

std::unique_ptr<InferenceModel> ModelRegistry::create(const std::string& type) const {
  std::map<std::string, Factory>::const_iterator it = factories_.find(type);
  if (it == factories_.end()) {
    return std::unique_ptr<InferenceModel>();
  }
  return it->second();
}

void registerBuiltinModels() {
  ModelRegistry::instance().registerFactory("BEV", []() { return std::unique_ptr<InferenceModel>(new BevModel()); });
  ModelRegistry::instance().registerFactory("GOD", []() { return std::unique_ptr<InferenceModel>(new GodModel()); });
  ModelRegistry::instance().registerFactory("LaneMark", []() { return std::unique_ptr<InferenceModel>(new LaneMarkModel()); });
  ModelRegistry::instance().registerFactory("E2E", []() { return std::unique_ptr<InferenceModel>(new E2ETrajectoryModel()); });
}

}  // namespace ai
