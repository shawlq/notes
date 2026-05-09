#pragma once

#include <chrono>
#include <string>

namespace ai {

class Timer {
 public:
  Timer();
  void reset();
  double elapsedMs() const;

 private:
  std::chrono::steady_clock::time_point start_;
};

class ScopedTimer {
 public:
  explicit ScopedTimer(const std::string& name);
  ~ScopedTimer();

 private:
  std::string name_;
  Timer timer_;
};

}  // namespace ai
