#pragma once

#include <iosfwd>
#include <mutex>
#include <sstream>
#include <string>

namespace ai {

enum class LogLevel {
  kDebug = 0,
  kInfo,
  kWarn,
  kError
};

class Logger {
 public:
  static Logger& instance();

  void setLevel(LogLevel level);
  void setOutput(std::ostream* output);
  void log(LogLevel level, const std::string& component, const std::string& message);

 private:
  Logger();

  LogLevel level_;
  std::ostream* output_;
  std::mutex mutex_;
};

class LogLine {
 public:
  LogLine(LogLevel level, const std::string& component);
  ~LogLine();

  template <typename T>
  LogLine& operator<<(const T& value) {
    stream_ << value;
    return *this;
  }

 private:
  LogLevel level_;
  std::string component_;
  std::ostringstream stream_;
};

#define AI_LOG_DEBUG(component) ::ai::LogLine(::ai::LogLevel::kDebug, component)
#define AI_LOG_INFO(component) ::ai::LogLine(::ai::LogLevel::kInfo, component)
#define AI_LOG_WARN(component) ::ai::LogLine(::ai::LogLevel::kWarn, component)
#define AI_LOG_ERROR(component) ::ai::LogLine(::ai::LogLevel::kError, component)

}  // namespace ai
