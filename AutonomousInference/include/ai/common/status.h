#pragma once

#include <string>

namespace ai {

enum class StatusCode {
  kOk = 0,
  kInvalidArgument,
  kNotFound,
  kRuntimeError,
  kUnavailable
};

class Status {
 public:
  Status() : code_(StatusCode::kOk) {}
  Status(StatusCode code, const std::string& message) : code_(code), message_(message) {}

  static Status ok() { return Status(); }
  static Status invalidArgument(const std::string& message) { return Status(StatusCode::kInvalidArgument, message); }
  static Status notFound(const std::string& message) { return Status(StatusCode::kNotFound, message); }
  static Status runtimeError(const std::string& message) { return Status(StatusCode::kRuntimeError, message); }
  static Status unavailable(const std::string& message) { return Status(StatusCode::kUnavailable, message); }

  bool okStatus() const { return code_ == StatusCode::kOk; }
  explicit operator bool() const { return okStatus(); }
  StatusCode code() const { return code_; }
  const std::string& message() const { return message_; }

 private:
  StatusCode code_;
  std::string message_;
};

}  // namespace ai
