module;

#include <chrono>

export module delta_time;

namespace vktf {

export class [[nodiscard]] DeltaTime {
public:
  using FloatSeconds = std::chrono::duration<float>;

  [[nodiscard]] FloatSeconds::rep get() const noexcept { return delta_time_; }

  void Update() noexcept;

private:
  using Clock = std::chrono::steady_clock;
  using TimePoint = std::chrono::time_point<Clock>;

  TimePoint previous_time_ = Clock::now();
  FloatSeconds::rep delta_time_ = 0.0f;
};

}  // namespace vktf

module :private;

namespace vktf {

void DeltaTime::Update() noexcept {
  const auto current_time = Clock::now();
  const auto float_seconds = std::chrono::duration_cast<FloatSeconds>(current_time - previous_time_);
  delta_time_ = float_seconds.count();
  previous_time_ = current_time;
}

}  // namespace vktf
