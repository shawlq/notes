#include "ai/models/god_model.h"

#include "ai/common/logger.h"

namespace ai {

Status GodModel::infer(const TensorMap& inputs, TensorMap* outputs, const StreamHandle& stream) {
  Status status = validateInput(inputs);
  if (!status) {
    return status;
  }
  AI_LOG_DEBUG("god") << "infer on stream " << stream.id;
  return writeOutput(outputs, 2.0F);
}

}  // namespace ai
