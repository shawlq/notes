#include "ai/common/config.h"

#include <cctype>
#include <fstream>
#include <map>
#include <sstream>
#include <stdexcept>

#include "ai/common/tensor.h"

namespace ai {

std::string TensorShape::toString() const {
  std::ostringstream oss;
  oss << "[";
  for (std::size_t i = 0; i < dims.size(); ++i) {
    if (i != 0U) {
      oss << ",";
    }
    oss << dims[i];
  }
  oss << "]";
  return oss.str();
}

namespace {

class JsonValue {
 public:
  enum Type { kNull, kBool, kNumber, kString, kObject, kArray };

  JsonValue() : type_(kNull), bool_value_(false), number_value_(0.0) {}
  explicit JsonValue(bool value) : type_(kBool), bool_value_(value), number_value_(0.0) {}
  explicit JsonValue(double value) : type_(kNumber), bool_value_(false), number_value_(value) {}
  explicit JsonValue(const std::string& value) : type_(kString), bool_value_(false), number_value_(0.0), string_value_(value) {}

  static JsonValue object(const std::map<std::string, JsonValue>& value) {
    JsonValue v;
    v.type_ = kObject;
    v.object_value_ = value;
    return v;
  }

  static JsonValue array(const std::vector<JsonValue>& value) {
    JsonValue v;
    v.type_ = kArray;
    v.array_value_ = value;
    return v;
  }

  Type type() const { return type_; }
  bool asBool() const { return bool_value_; }
  double asNumber() const { return number_value_; }
  const std::string& asString() const { return string_value_; }
  const std::map<std::string, JsonValue>& asObject() const { return object_value_; }
  const std::vector<JsonValue>& asArray() const { return array_value_; }

  const JsonValue& at(const std::string& key) const {
    std::map<std::string, JsonValue>::const_iterator it = object_value_.find(key);
    if (it == object_value_.end()) {
      throw std::runtime_error("missing key: " + key);
    }
    return it->second;
  }

 private:
  Type type_;
  bool bool_value_;
  double number_value_;
  std::string string_value_;
  std::map<std::string, JsonValue> object_value_;
  std::vector<JsonValue> array_value_;
};

class JsonParser {
 public:
  explicit JsonParser(const std::string& text) : text_(text), pos_(0U) {}

  JsonValue parse() {
    JsonValue value = parseValue();
    skipSpaces();
    if (pos_ != text_.size()) {
      throw std::runtime_error("trailing characters in JSON");
    }
    return value;
  }

 private:
  JsonValue parseValue() {
    skipSpaces();
    if (pos_ >= text_.size()) {
      throw std::runtime_error("unexpected end of JSON");
    }
    const char c = text_[pos_];
    if (c == '{') {
      return parseObject();
    }
    if (c == '[') {
      return parseArray();
    }
    if (c == '"') {
      return JsonValue(parseString());
    }
    if (c == 't' || c == 'f') {
      return JsonValue(parseBool());
    }
    if (c == 'n') {
      expect("null");
      return JsonValue();
    }
    return JsonValue(parseNumber());
  }

  JsonValue parseObject() {
    consume('{');
    std::map<std::string, JsonValue> object;
    skipSpaces();
    if (peek('}')) {
      consume('}');
      return JsonValue::object(object);
    }
    while (true) {
      std::string key = parseString();
      skipSpaces();
      consume(':');
      object[key] = parseValue();
      skipSpaces();
      if (peek('}')) {
        consume('}');
        break;
      }
      consume(',');
      skipSpaces();
    }
    return JsonValue::object(object);
  }

  JsonValue parseArray() {
    consume('[');
    std::vector<JsonValue> array;
    skipSpaces();
    if (peek(']')) {
      consume(']');
      return JsonValue::array(array);
    }
    while (true) {
      array.push_back(parseValue());
      skipSpaces();
      if (peek(']')) {
        consume(']');
        break;
      }
      consume(',');
    }
    return JsonValue::array(array);
  }

  std::string parseString() {
    consume('"');
    std::ostringstream oss;
    while (pos_ < text_.size()) {
      char c = text_[pos_++];
      if (c == '"') {
        return oss.str();
      }
      if (c == '\\') {
        if (pos_ >= text_.size()) {
          throw std::runtime_error("invalid escape");
        }
        char escaped = text_[pos_++];
        if (escaped == '"' || escaped == '\\' || escaped == '/') {
          oss << escaped;
        } else if (escaped == 'n') {
          oss << '\n';
        } else if (escaped == 't') {
          oss << '\t';
        } else {
          throw std::runtime_error("unsupported escape");
        }
      } else {
        oss << c;
      }
    }
    throw std::runtime_error("unterminated string");
  }

  bool parseBool() {
    if (text_.compare(pos_, 4U, "true") == 0) {
      pos_ += 4U;
      return true;
    }
    if (text_.compare(pos_, 5U, "false") == 0) {
      pos_ += 5U;
      return false;
    }
    throw std::runtime_error("invalid boolean");
  }

  double parseNumber() {
    const std::size_t start = pos_;
    if (text_[pos_] == '-') {
      ++pos_;
    }
    while (pos_ < text_.size() && std::isdigit(static_cast<unsigned char>(text_[pos_]))) {
      ++pos_;
    }
    if (pos_ < text_.size() && text_[pos_] == '.') {
      ++pos_;
      while (pos_ < text_.size() && std::isdigit(static_cast<unsigned char>(text_[pos_]))) {
        ++pos_;
      }
    }
    return std::stod(text_.substr(start, pos_ - start));
  }

  void skipSpaces() {
    while (pos_ < text_.size() && std::isspace(static_cast<unsigned char>(text_[pos_]))) {
      ++pos_;
    }
  }

  bool peek(char expected) const {
    return pos_ < text_.size() && text_[pos_] == expected;
  }

  void consume(char expected) {
    skipSpaces();
    if (pos_ >= text_.size() || text_[pos_] != expected) {
      throw std::runtime_error(std::string("expected '") + expected + "'");
    }
    ++pos_;
  }

  void expect(const char* token) {
    const std::size_t len = std::string(token).size();
    if (text_.compare(pos_, len, token) != 0) {
      throw std::runtime_error(std::string("expected ") + token);
    }
    pos_ += len;
  }

  std::string text_;
  std::size_t pos_;
};

std::vector<int> toIntVector(const JsonValue& value) {
  std::vector<int> dims;
  const std::vector<JsonValue>& array = value.asArray();
  for (std::vector<JsonValue>::const_iterator it = array.begin(); it != array.end(); ++it) {
    dims.push_back(static_cast<int>(it->asNumber()));
  }
  return dims;
}

}  // namespace

Status ConfigLoader::loadFromFile(const std::string& path, PipelineConfig* config) const {
  std::ifstream file(path.c_str());
  if (!file) {
    return Status::notFound("failed to open config file: " + path);
  }
  std::ostringstream buffer;
  buffer << file.rdbuf();
  return loadFromString(buffer.str(), config);
}

Status ConfigLoader::loadFromString(const std::string& text, PipelineConfig* config) const {
  if (config == nullptr) {
    return Status::invalidArgument("config must not be null");
  }
  try {
    JsonValue root = JsonParser(text).parse();
    PipelineConfig parsed;

    const JsonValue& runtime = root.at("runtime");
    parsed.runtime.device_id = static_cast<int>(runtime.at("device_id").asNumber());
    parsed.runtime.enable_cuda_graph = runtime.at("enable_cuda_graph").asBool();
    parsed.runtime.warmup_runs = static_cast<int>(runtime.at("warmup_runs").asNumber());
    parsed.runtime.memory_pool_bytes = static_cast<std::size_t>(runtime.at("memory_pool_bytes").asNumber());
    parsed.runtime.max_concurrent_models = static_cast<int>(runtime.at("max_concurrent_models").asNumber());

    const std::vector<JsonValue>& models = root.at("models").asArray();
    for (std::vector<JsonValue>::const_iterator it = models.begin(); it != models.end(); ++it) {
      ModelConfig model;
      model.name = it->at("name").asString();
      model.type = it->at("type").asString();
      model.engine_path = it->at("engine_path").asString();
      model.input_tensor = it->at("input_tensor").asString();
      model.output_tensor = it->at("output_tensor").asString();
      model.priority = static_cast<int>(it->at("priority").asNumber());
      model.stream_id = static_cast<int>(it->at("stream_id").asNumber());
      model.estimated_workspace_bytes = static_cast<std::size_t>(it->at("estimated_workspace_bytes").asNumber());
      model.input_shape = toIntVector(it->at("input_shape"));
      model.output_shape = toIntVector(it->at("output_shape"));
      parsed.models.push_back(model);
    }

    const std::vector<JsonValue>& edges = root.at("edges").asArray();
    for (std::vector<JsonValue>::const_iterator it = edges.begin(); it != edges.end(); ++it) {
      PipelineEdge edge;
      edge.from = it->at("from").asString();
      edge.to = it->at("to").asString();
      parsed.edges.push_back(edge);
    }

    *config = parsed;
    return Status::ok();
  } catch (const std::exception& e) {
    return Status::invalidArgument(e.what());
  }
}

}  // namespace ai
