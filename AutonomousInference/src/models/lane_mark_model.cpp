#include "ai/models/lane_mark_model.h"

#include "ai/common/logger.h"

namespace ai {

Status LaneMarkModel::infer(const TensorMap& inputs, TensorMap* outputs, const StreamHandle& stream) {
  Status status = validateInput(inputs);
  if (!status) {
    return status;
  }
  AI_LOG_DEBUG("lane") << "infer on stream " << stream.id;
  return writeOutput(outputs, 3.0F);
}

}  // namespace ai
