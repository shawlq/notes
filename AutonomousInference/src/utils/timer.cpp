#include "ai/utils/timer.h"

#include "ai/common/logger.h"

namespace ai {

Timer::Timer() : start_(std::chrono::steady_clock::now()) {}

void Timer::reset() {
  start_ = std::chrono::steady_clock::now();
}

double Timer::elapsedMs() const {
  const std::chrono::steady_clock::time_point now = std::chrono::steady_clock::now();
  return std::chrono::duration_cast<std::chrono::duration<double, std::milli> >(now - start_).count();
}

ScopedTimer::ScopedTimer(const std::string& name) : name_(name) {}

ScopedTimer::~ScopedTimer() {
  AI_LOG_INFO("timer") << name_ << " took " << timer_.elapsedMs() << " ms";
}

}  // namespace ai
