#include <array>
#include <numbers>

#include <gtest/gtest.h>

import data_view;

namespace {

using DataType = float;

TEST(DataViewTest, IsConstructibleFromSingleValue) {
  static constexpr auto kData = std::numbers::pi_v<DataType>;
  static constexpr vktf::DataView kDataView{kData};

  static_assert(&kData == kDataView.data());
  static_assert(1 == kDataView.size());
  static_assert(sizeof(DataType) == kDataView.size_bytes());
}

TEST(DataViewTest, IsConstructibleFromPointerAndSize) {
  static constexpr std::array kData{0.0f, 1.0f, 2.0f};
  static constexpr auto kOffset = 1;
  static constexpr auto* kDataViewPtr = &kData[kOffset];
  static constexpr auto kDataViewSize = kData.size() - kOffset;
  static constexpr vktf::DataView kDataView{kDataViewPtr, kDataViewSize};

  static_assert(kDataViewPtr == kDataView.data());
  static_assert(kDataViewSize == kDataView.size());
  static_assert(sizeof(DataType) * kDataViewSize == kDataView.size_bytes());
}

TEST(DataViewTest, IsConstructibleFromCArray) {
  static constexpr DataType kData[] = {1.0f, 2.0f, 3.0f};  // NOLINT(*-c-arrays)
  static constexpr auto kDataViewSize = std::size(kData);
  static constexpr vktf::DataView kDataView{kData};

  static_assert(kData == kDataView.data());
  static_assert(kDataViewSize == kDataView.size());
  static_assert(sizeof(DataType) * kDataViewSize == kDataView.size_bytes());
}

TEST(DataViewTest, IsConstructibleFromDataRange) {
  static constexpr std::array kData{0.0f, 1.0f, 2.0f};
  static constexpr vktf::DataView kDataView{kData};

  static_assert(kData.data() == kDataView.data());
  static_assert(kData.size() == kDataView.size());
  static_assert(sizeof(DataType) * kData.size() == kDataView.size_bytes());
}

}  // namespace
