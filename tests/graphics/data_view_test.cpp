#include <array>
#include <numbers>

#include <gtest/gtest.h>

import data_view;

namespace {

using DataType = float;
constexpr auto kDataTypeSize = sizeof(DataType);
constexpr auto kDataValue = std::numbers::e_v<DataType>;
constexpr std::array kDataValues{kDataValue, kDataValue, kDataValue};

TEST(DataViewTest, TestSingleValueInitializationHasTheCorrectData) {
  static constexpr gfx::DataView kDataView{kDataValue};
  static_assert(kDataView.data() == &kDataValue);
}

TEST(DataViewTest, TestSingleValueInitializationHasTheCorrectSize) {
  static constexpr gfx::DataView kDataView{kDataValue};
  static_assert(kDataView.size() == 1);
}

TEST(DataViewTest, TestSingleValueInitializationHasTheCorrectSizeInBytes) {
  static constexpr gfx::DataView kDataView{kDataValue};
  static_assert(kDataView.size_bytes() == sizeof(DataType));
}

TEST(DataViewTest, TestPointerInitializationHasTheCorrectData) {
  static constexpr auto* kDataValuesPtr = kDataValues.data();
  static constexpr auto kDataViewSize = 2;
  static constexpr gfx::DataView kDataView{kDataValuesPtr, kDataViewSize};
  static_assert(kDataView.data() == kDataValuesPtr);
}

TEST(DataViewTest, TestPointerInitializationHasTheCorrectSize) {
  static constexpr auto* kDataValuesPtr = kDataValues.data();
  static constexpr auto kDataViewSize = 2;
  static constexpr gfx::DataView kDataView{kDataValuesPtr, kDataViewSize};
  static_assert(kDataView.size() == kDataViewSize);
}

TEST(DataViewTest, TestPointerInitializationHasTheCorrectSizeinBytes) {
  static constexpr auto* kDataValuesPtr = kDataValues.data();
  static constexpr auto kDataViewSize = 2;
  static constexpr gfx::DataView kDataView{kDataValuesPtr, kDataViewSize};
  static_assert(kDataView.size_bytes() == kDataTypeSize * kDataViewSize);
}

TEST(DataViewTest, TestDataRangeInitializationHasTheCorrectData) {
  static constexpr gfx::DataView<const DataType> kDataView{kDataValues};
  static_assert(kDataView.data() == kDataValues.data());
}

TEST(DataViewTest, DataRangeInitializationHasTheCorrectSize) {
  static constexpr gfx::DataView<const DataType> kDataView{kDataValues};
  static_assert(kDataView.size() == kDataValues.size());
}

TEST(DataViewTest, TestDataRangeInitializationHasTheCorrectSizeInBytes) {
  static constexpr gfx::DataView<const DataType> kDataView{kDataValues};
  static_assert(kDataView.size_bytes() == kDataTypeSize * kDataValues.size());
}

}  // namespace
