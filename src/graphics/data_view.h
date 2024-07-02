#ifndef GRAPHICS_DATA_VIEW_H_
#define GRAPHICS_DATA_VIEW_H_

#include <ranges>

namespace gfx {

template <typename R, typename T>
concept DataRange = std::ranges::contiguous_range<R> &&  //
                    std::ranges::sized_range<R> &&       //
                    std::same_as<std::ranges::range_value_t<R>, std::remove_cvref_t<T>>;

template <typename T>
class DataView {
public:
  constexpr DataView(T& data) noexcept : data_{&data}, size_{1} {}  // NOLINT(runtime/explicit)
  constexpr DataView(T* data, const std::size_t size) noexcept : data_{data}, size_{size} {}
  constexpr DataView(DataRange<T> auto&& range) noexcept  // NOLINT(runtime/explicit)
      : data_{std::ranges::data(range)}, size_{std::ranges::size(range)} {}

  [[nodiscard]] constexpr T* data() const noexcept { return data_; }
  [[nodiscard]] constexpr std::size_t size() const noexcept { return size_; }
  [[nodiscard]] constexpr std::size_t size_bytes() const noexcept { return size_ * sizeof(T); }

private:
  T* data_;
  std::size_t size_;
};

}  // namespace gfx

#endif  // GRAPHICS_DATA_VIEW_H_
