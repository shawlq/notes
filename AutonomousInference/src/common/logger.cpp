#include "ai/common/logger.h"

#include <ctime>
#include <iomanip>
#include <iostream>

namespace ai {

namespace {
const char* toString(LogLevel level) {
  switch (level) {
    case LogLevel::kDebug: return "DEBUG";
    case LogLevel::kInfo: return "INFO";
    case LogLevel::kWarn: return "WARN";
    case LogLevel::kError: return "ERROR";
  }
  return "UNKNOWN";
}
}  // namespace

Logger& Logger::instance() {
  static Logger logger;
  return logger;
}

Logger::Logger() : level_(LogLevel::kInfo), output_(&std::cout) {}

void Logger::setLevel(LogLevel level) {
  std::lock_guard<std::mutex> lock(mutex_);
  level_ = level;
}

void Logger::setOutput(std::ostream* output) {
  std::lock_guard<std::mutex> lock(mutex_);
  output_ = output == nullptr ? &std::cout : output;
}

void Logger::log(LogLevel level, const std::string& component, const std::string& message) {
  if (static_cast<int>(level) < static_cast<int>(level_)) {
    return;
  }
  std::lock_guard<std::mutex> lock(mutex_);
  if (output_ == nullptr) {
    return;
  }
  std::time_t now = std::time(nullptr);
  (*output_) << std::put_time(std::localtime(&now), "%F %T") << " [" << toString(level) << "] "
             << component << ": " << message << std::endl;
}

LogLine::LogLine(LogLevel level, const std::string& component) : level_(level), component_(component) {}

LogLine::~LogLine() {
  Logger::instance().log(level_, component_, stream_.str());
}

}  // namespace ai
