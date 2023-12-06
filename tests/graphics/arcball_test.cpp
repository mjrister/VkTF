#include "graphics/arcball.cpp"  // NOLINT(build/include)

#include <cmath>

#include <gtest/gtest.h>

namespace {

TEST(ArcballTest, GetNormalizedDeviceCoordinates) {
  constexpr auto kWidth = 300, kHeight = 200;
  constexpr auto kWindowDimensions = std::make_pair(kWidth, kHeight);

  constexpr auto kCursorPositionNdc0 = GetNormalizedDeviceCoordinates(glm::dvec2{0.f, 0.f}, kWindowDimensions);
  static_assert(kCursorPositionNdc0.x == -1.0f);
  static_assert(kCursorPositionNdc0.y == 1.0f);

  constexpr auto kCursorPositionNdc1 = GetNormalizedDeviceCoordinates(glm::dvec2{0.f, kHeight}, kWindowDimensions);
  static_assert(kCursorPositionNdc1.x == -1.0f);
  static_assert(kCursorPositionNdc1.y == -1.0f);

  constexpr auto kCursorPositionNdc2 = GetNormalizedDeviceCoordinates(glm::dvec2{kWidth, kHeight}, kWindowDimensions);
  static_assert(kCursorPositionNdc2.x == 1.0f);
  static_assert(kCursorPositionNdc2.y == -1.0f);

  constexpr auto kCursorPositionNdc3 = GetNormalizedDeviceCoordinates(glm::dvec2{kWidth, 0.f}, kWindowDimensions);
  static_assert(kCursorPositionNdc3.x == 1.0f);
  static_assert(kCursorPositionNdc3.y == 1.0f);

  constexpr auto kCursorPositionNdc4 =
      GetNormalizedDeviceCoordinates(glm::dvec2{kWidth / 2.f, kHeight / 2.f}, kWindowDimensions);
  static_assert(kCursorPositionNdc4.x == 0.f);
  static_assert(kCursorPositionNdc4.y == 0.f);

  constexpr auto kCursorPositionNdc5 = GetNormalizedDeviceCoordinates(glm::dvec2{-1.f, -1.f}, kWindowDimensions);
  static_assert(kCursorPositionNdc5.x == -1.0f);
  static_assert(kCursorPositionNdc5.y == 1.0f);

  constexpr auto kCursorPositionNdc6 =
      GetNormalizedDeviceCoordinates(glm::dvec2{kWidth + 1.f, kHeight + 1.f}, kWindowDimensions);
  static_assert(kCursorPositionNdc6.x == 1.0f);
  static_assert(kCursorPositionNdc6.y == -1.0f);
}

TEST(ArcballTest, GetArcballPositionForCursorInsideUnitSphere) {
  constexpr glm::vec2 kCursorPositionNdc{0.5f, 0.25f};
  const auto arcball_position = GetArcballPosition(kCursorPositionNdc);
  EXPECT_FLOAT_EQ(arcball_position.x, kCursorPositionNdc.x);
  EXPECT_FLOAT_EQ(arcball_position.y, kCursorPositionNdc.y);
  EXPECT_FLOAT_EQ(arcball_position.z, 0.82915622f);
}

TEST(ArcballTest, GetArcballPositionForCursorOutsideUnitSphere) {
  constexpr glm::vec2 kCursorPositionNdc{0.75f, 0.85f};
  const auto arcball_position = GetArcballPosition(kCursorPositionNdc);
  EXPECT_FLOAT_EQ(arcball_position.x, 0.66162163f);
  EXPECT_FLOAT_EQ(arcball_position.y, 0.74983788f);
  EXPECT_FLOAT_EQ(arcball_position.z, 0.0f);
}
}  // namespace
