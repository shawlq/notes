#include "ai/models/bev_model.h"

#include "ai/common/logger.h"

namespace ai {

Status BevModel::infer(const TensorMap& inputs, TensorMap* outputs, const StreamHandle& stream) {
  Status status = validateInput(inputs);
  if (!status) {
    return status;
  }
  AI_LOG_DEBUG("bev") << "infer on stream " << stream.id;
  return writeOutput(outputs, 1.0F);
}

}  // namespace ai
