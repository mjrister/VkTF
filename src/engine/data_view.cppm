module;

#include <concepts>
#include <cstddef>
#include <ranges>
#include <type_traits>

export module data_view;

namespace vktf {

export template <typename R, typename T>
concept DataRange = std::ranges::contiguous_range<R> && std::ranges::sized_range<R>
                    && std::same_as<std::ranges::range_value_t<R>, std::remove_const_t<T>>;

export template <typename T>
class [[nodiscard]] DataView {
public:
  constexpr DataView(const T& data) noexcept : data_{&data}, size_{1} {}  // NOLINT(*-explicit-*)
  constexpr DataView(const T* const data, const std::size_t size) noexcept : data_{data}, size_{size} {}
  constexpr DataView(const DataRange<T> auto& data)  // NOLINT(*-explicit-*)
      : data_{std::ranges::data(data)}, size_{std::ranges::size(data)} {}

  [[nodiscard]] constexpr const T* data() const noexcept { return data_; }
  [[nodiscard]] constexpr std::size_t size() const noexcept { return size_; }
  [[nodiscard]] constexpr std::size_t size_bytes() const noexcept { return sizeof(T) * size_; }

private:
  const T* data_;
  std::size_t size_;
};

template <typename T>
DataView(const T&) -> DataView<T>;

template <typename T, std::size_t N>
DataView(const T (&)[N]) -> DataView<T>;  // NOLINT(*-c-arrays)

template <typename R>
  requires DataRange<R, std::ranges::range_value_t<R>>
DataView(const R&) -> DataView<std::ranges::range_value_t<R>>;

}  // namespace vktf
