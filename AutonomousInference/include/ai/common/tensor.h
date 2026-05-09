#pragma once

#include <cstddef>
#include <numeric>
#include <string>
#include <vector>

namespace ai {

enum class DataType {
  kFloat32,
  kFloat16,
  kInt8,
  kInt32
};

struct TensorShape {
  std::vector<int> dims;

  std::size_t elementCount() const {
    if (dims.empty()) {
      return 0U;
    }
    std::size_t count = 1U;
    for (std::vector<int>::const_iterator it = dims.begin(); it != dims.end(); ++it) {
      count *= static_cast<std::size_t>(*it);
    }
    return count;
  }

  std::string toString() const;
};

struct Tensor {
  Tensor() : dtype(DataType::kFloat32) {}
  Tensor(const std::string& tensor_name, const TensorShape& tensor_shape)
      : name(tensor_name), shape(tensor_shape), dtype(DataType::kFloat32), data(tensor_shape.elementCount(), 0.0F) {}

  std::string name;
  TensorShape shape;
  DataType dtype;
  std::vector<float> data;
};

using TensorMap = std::vector<Tensor>;

}  // namespace ai
