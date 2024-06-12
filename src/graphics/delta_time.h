#ifndef GRAPHICS_DELTA_TIME_H_
#define GRAPHICS_DELTA_TIME_H_

#include <chrono>

namespace gfx {

class DeltaTime {
  using Clock = std::chrono::steady_clock;
  using Duration = std::chrono::duration<float>;
  using TimePoint = std::chrono::time_point<Clock, Duration>;

public:
  // NOLINTNEXTLINE(google-explicit-constructor, hicpp-explicit-conversions)
  [[nodiscard]] operator Duration::rep() const noexcept { return delta_time_; }

  void Update() noexcept {
    const TimePoint current_time = Clock::now();
    delta_time_ = (current_time - previous_time_).count();
    previous_time_ = current_time;
  }

private:
  TimePoint previous_time_ = Clock::now();
  Duration::rep delta_time_ = 0;
};

}  // namespace gfx

#endif  // GRAPHICS_DELTA_TIME_H_
