module;

#include <concepts>
#include <cstddef>
#include <ranges>
#include <type_traits>

export module data_view;

namespace vktf {

/**
 * @brief A concept defining requirements for a contiguous range of homogeneous data.
 * @tparam R The range type.
 * @tparam T The range element type.
 */
export template <typename R, typename T>
concept DataRange = std::ranges::contiguous_range<R> && std::ranges::sized_range<R>
                    && std::same_as<std::ranges::range_value_t<R>, std::remove_const_t<T>>;

/**
 * @brief A non-owning view of contiguous data.
 * @details This class represents a non-owning view of contiguous data that improves upon @c std::span by allowing
 *          implicit construction from a single element. This provides a unified interface in situations that require
 *          working with one or more objects such as copying arbitrary data to a uniform buffer.
 * @tparam T The type for each element in the view.
 * @warning The underlying data for the view must remain valid for the entire lifetime of this abstraction.
 */
export template <typename T>
class [[nodiscard]] DataView {
public:
  /**
   * @brief Creates a @ref DataView.
   * @param data The single element to create a view from.
   */
  constexpr DataView(const T& data) noexcept : data_{&data}, size_{1} {}  // NOLINT(*-explicit-*)

  /**
   * @brief Creates a @ref DataView.
   * @param data A non-owning pointer to the data to create a view from.
   * @param size The number of elements in @p data.
   * @warning The caller is responsible for ensuring @p data is a valid pointer with at least @p size elements.
   */
  constexpr DataView(const T* const data, const std::size_t size) noexcept : data_{data}, size_{size} {}

  /**
   * @brief Creates a @ref DataView.
   * @param data The data range to create a view from.
   */
  constexpr DataView(const DataRange<T> auto& data)  // NOLINT(*-explicit-*)
      : data_{std::ranges::data(data)}, size_{std::ranges::size(data)} {}

  /** @brief Gets a non-owning pointer to the first element in the view. */
  [[nodiscard]] constexpr const T* data() const noexcept { return data_; }

  /** @brief Gets the number of elements in the view. */
  [[nodiscard]] constexpr std::size_t size() const noexcept { return size_; }

  /** @brief Gets the total size in bytes of all elements in the view. */
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
