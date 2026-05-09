#include "ai/models/e2e_trajectory_model.h"

#include "ai/common/logger.h"

namespace ai {

Status E2ETrajectoryModel::infer(const TensorMap& inputs, TensorMap* outputs, const StreamHandle& stream) {
  Status status = validateInput(inputs);
  if (!status) {
    return status;
  }
  AI_LOG_DEBUG("e2e") << "infer on stream " << stream.id;
  return writeOutput(outputs, 4.0F);
}

}  // namespace ai
