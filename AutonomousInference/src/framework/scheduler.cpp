#include "ai/framework/scheduler.h"

#include <algorithm>
#include <map>
#include <set>

namespace ai {

namespace {
const ModelConfig* findModel(const PipelineConfig& config, const std::string& name) {
  for (std::vector<ModelConfig>::const_iterator it = config.models.begin(); it != config.models.end(); ++it) {
    if (it->name == name) {
      return &(*it);
    }
  }
  return nullptr;
}

bool stepLess(const ScheduleStep& lhs, const ScheduleStep& rhs) {
  if (lhs.priority != rhs.priority) {
    return lhs.priority < rhs.priority;
  }
  return lhs.model_name < rhs.model_name;
}
}  // namespace

Status Scheduler::build(const PipelineConfig& config) {
  std::map<std::string, int> indegree;
  std::map<std::string, std::vector<std::string> > graph;
  for (std::vector<ModelConfig>::const_iterator it = config.models.begin(); it != config.models.end(); ++it) {
    indegree[it->name] = 0;
  }
  for (std::vector<PipelineEdge>::const_iterator it = config.edges.begin(); it != config.edges.end(); ++it) {
    if (findModel(config, it->from) == nullptr || findModel(config, it->to) == nullptr) {
      return Status::invalidArgument("edge references unknown model");
    }
    graph[it->from].push_back(it->to);
    indegree[it->to] += 1;
  }

  std::vector<ScheduleStep> ready;
  for (std::map<std::string, int>::const_iterator it = indegree.begin(); it != indegree.end(); ++it) {
    if (it->second == 0) {
      const ModelConfig* model = findModel(config, it->first);
      ready.push_back(ScheduleStep{model->name, model->priority, model->stream_id});
    }
  }
  std::sort(ready.begin(), ready.end(), stepLess);

  steps_.clear();
  while (!ready.empty()) {
    ScheduleStep step = ready.front();
    ready.erase(ready.begin());
    steps_.push_back(step);
    const std::vector<std::string>& next = graph[step.model_name];
    for (std::vector<std::string>::const_iterator it = next.begin(); it != next.end(); ++it) {
      indegree[*it] -= 1;
      if (indegree[*it] == 0) {
        const ModelConfig* model = findModel(config, *it);
        ready.push_back(ScheduleStep{model->name, model->priority, model->stream_id});
        std::sort(ready.begin(), ready.end(), stepLess);
      }
    }
  }

  if (steps_.size() != config.models.size()) {
    return Status::invalidArgument("pipeline graph has a cycle");
  }
  return Status::ok();
}

}  // namespace ai
