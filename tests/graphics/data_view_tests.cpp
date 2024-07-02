#include "graphics/data_view.h"

#include <array>
#include <cstdint>
#include <numbers>

#include <gtest/gtest.h>

namespace {

using DataType = float;
constexpr auto kDataValue = std::numbers::pi_v<DataType>;
constexpr std::array kDataValues{kDataValue, kDataValue, kDataValue};

TEST(DataViewTest, SingleValueInitializationHasTheCorrectData) {
  constexpr gfx::DataView kDataView{kDataValue};
  static_assert(kDataView.data() == &kDataValue);
}

TEST(DataViewTest, SingleValueInitializationHasTheCorrectSize) {
  constexpr gfx::DataView kDataView{kDataValue};
  static_assert(kDataView.size() == 1);
}

TEST(DataViewTest, SingleValueInitializationHasTheCorrectSizeInBytes) {
  constexpr gfx::DataView kDataView{kDataValue};
  static_assert(kDataView.size_bytes() == sizeof(DataType));
}

TEST(DataViewTest, PointerInitializationHasTheCorrectData) {
  constexpr auto* kDataValuesPtr = kDataValues.data();
  constexpr auto kDataSize = 2;
  constexpr gfx::DataView kDataView{kDataValuesPtr, kDataSize};
  static_assert(kDataView.data() == kDataValuesPtr);
}

TEST(DataViewTest, PointerInitializationHasTheCorrectSize) {
  constexpr auto* kDataValuesPtr = kDataValues.data();
  constexpr auto kDataViewSize = 2;
  constexpr gfx::DataView kDataView{kDataValuesPtr, kDataViewSize};
  static_assert(kDataView.size() == kDataViewSize);
}

TEST(DataViewTest, PointerInitializationHasTheCorrectSizeinBytes) {
  constexpr auto* kDataValuesPtr = kDataValues.data();
  constexpr auto kDataViewSize = 2;
  constexpr gfx::DataView kDataView{kDataValuesPtr, kDataViewSize};
  static_assert(kDataView.size_bytes() == sizeof(DataType) * kDataViewSize);
}

TEST(DataViewTest, DataRangeInitializationHasTheCorrectData) {
  constexpr gfx::DataView<const DataType> kDataView{kDataValues};
  static_assert(kDataView.data() == kDataValues.data());
}

TEST(DataViewTest, DataRangeInitializationHasTheCorrectSize) {
  constexpr gfx::DataView<const DataType> kDataView{kDataValues};
  static_assert(kDataView.size() == kDataValues.size());
}

TEST(DataViewTest, DataRangeInitializationHasTheCorrectSizeInBytes) {
  constexpr gfx::DataView<const DataType> kDataView{kDataValues};
  static_assert(kDataView.size_bytes() == sizeof(DataType) * kDataValues.size());
}

}  // namespace
