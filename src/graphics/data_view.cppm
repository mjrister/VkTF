module;

#include <concepts>
#include <ranges>
#include <type_traits>

export module data_view;

namespace gfx {

export template <typename R, typename T>
concept DataRange = std::ranges::contiguous_range<R> &&  //
                    std::ranges::sized_range<R> &&       //
                    std::same_as<std::ranges::range_value_t<R>, std::remove_cvref_t<T>>;

export template <typename T>
class DataView {
public:
  // NOLINTBEGIN(*-explicit-*): implicit construction follow a similar convention to view types in the standard library
  constexpr DataView(T& data) noexcept : data_{&data}, size_{1} {}
  constexpr DataView(T* data, const std::size_t size) noexcept : data_{data}, size_{size} {}
  constexpr DataView(DataRange<T> auto&& range) : data_{std::ranges::data(range)}, size_{std::ranges::size(range)} {}
  // NOLINTEND(*-explicit-*)

  [[nodiscard]] constexpr T* data() const noexcept { return data_; }
  [[nodiscard]] constexpr std::size_t size() const noexcept { return size_; }
  [[nodiscard]] constexpr std::size_t size_bytes() const noexcept { return sizeof(T) * size_; }

private:
  T* data_;
  std::size_t size_;
};

}  // namespace gfx
