#include "graphics/arc_camera.cpp"  // NOLINT(build/include)

#include <array>
#include <cstdint>

#include <gtest/gtest.h>

namespace {

constexpr auto kEpsilon = 1.0e-6f;
constexpr auto kHalfPi = glm::half_pi<float>();
constexpr auto kPi = glm::pi<float>();
constexpr auto kRadius = 2.0f;

TEST(CameraTest, ConvertingCartesianCoordinatesOnThePositiveZAxisToSphericalCoordinatesIsCorrect) {
  static constexpr glm::vec3 kCartesianPosition{0.0f, 0.0f, kRadius};
  const auto spherical_position = ToSphericalCoordinates(kCartesianPosition);
  EXPECT_FLOAT_EQ(spherical_position.radius, kRadius);
  EXPECT_FLOAT_EQ(spherical_position.theta, 0.0f);
  EXPECT_FLOAT_EQ(spherical_position.phi, 0.0f);
}

TEST(CameraTest, ConvertingCartesianCoordinatesOnTheNegativeZAxisToSphericalCoordinatesIsCorrect) {
  static constexpr glm::vec3 kCartesianPosition{0.0f, 0.0f, -kRadius};
  const auto spherical_position = ToSphericalCoordinates(kCartesianPosition);
  EXPECT_FLOAT_EQ(spherical_position.radius, kRadius);
  EXPECT_FLOAT_EQ(spherical_position.theta, kPi);
  EXPECT_FLOAT_EQ(spherical_position.phi, 0.0f);
}

TEST(CameraTest, ConvertingCartesianCoordinatesOnThePositiveXAxisToSphericalCoordinatesIsCorrect) {
  static constexpr glm::vec3 kCartesianPosition{kRadius, 0.0f, 0.0f};
  const auto spherical_position = ToSphericalCoordinates(kCartesianPosition);
  EXPECT_FLOAT_EQ(spherical_position.radius, kRadius);
  EXPECT_FLOAT_EQ(spherical_position.theta, kHalfPi);
  EXPECT_FLOAT_EQ(spherical_position.phi, 0.0f);
}

TEST(CameraTest, ConvertingCartesianCoordinatesOnTheNegativeXAxisToSphericalCoordinatesIsCorrect) {
  static constexpr glm::vec3 kPosition{-kRadius, 0.0f, 0.0f};
  const auto spherical_position = ToSphericalCoordinates(kPosition);
  EXPECT_FLOAT_EQ(spherical_position.radius, kRadius);
  EXPECT_FLOAT_EQ(spherical_position.theta, -kHalfPi);
  EXPECT_FLOAT_EQ(spherical_position.phi, 0.0f);
}

TEST(CameraTest, ConvertingCartesianCoordinatesOnThePositiveYAxisToSphericalCoordinatesIsCorrect) {
  static constexpr glm::vec3 kCartesianPosition{0.0f, kRadius, 0.0f};
  const auto spherical_position = ToSphericalCoordinates(kCartesianPosition);
  EXPECT_FLOAT_EQ(spherical_position.radius, kRadius);
  EXPECT_FLOAT_EQ(spherical_position.theta, 0.0f);
  EXPECT_FLOAT_EQ(spherical_position.phi, -kHalfPi);
}

TEST(CameraTest, ConvertingCartesianCoordinatesOnTheNegativeYAxisToSphericalCoordinatesIsCorrect) {
  static constexpr glm::vec3 kPosition{0.0f, -kRadius, 0.0f};
  const auto spherical_position = ToSphericalCoordinates(kPosition);
  EXPECT_FLOAT_EQ(spherical_position.radius, kRadius);
  EXPECT_FLOAT_EQ(spherical_position.theta, 0.0f);
  EXPECT_FLOAT_EQ(spherical_position.phi, kHalfPi);
}

TEST(CameraTest, ConvertingSphericalCoordinatesOnThePositiveZAxisToCartesianCoordinatesIsCorrect) {
  static constexpr gfx::SphericalCoordinates kSphericalCoordinates{.radius = kRadius, .theta = 0.0f, .phi = 0.0f};
  const auto cartesian_position = ToCartesianCoordinates(kSphericalCoordinates);
  EXPECT_NEAR(cartesian_position.x, 0.0f, kEpsilon);
  EXPECT_NEAR(cartesian_position.y, 0.0f, kEpsilon);
  EXPECT_NEAR(cartesian_position.z, kRadius, kEpsilon);
}

TEST(CameraTest, ConvertingSphericalCoordinatesOnTheNegativeZAxisToCartesianCoordinatesIsCorrect) {
  static constexpr gfx::SphericalCoordinates kSphericalCoordinates{.radius = kRadius, .theta = kPi, .phi = 0.0f};
  const auto cartesian_position = ToCartesianCoordinates(kSphericalCoordinates);
  EXPECT_NEAR(cartesian_position.x, 0.0f, kEpsilon);
  EXPECT_NEAR(cartesian_position.y, 0.0f, kEpsilon);
  EXPECT_NEAR(cartesian_position.z, -kRadius, kEpsilon);
}

TEST(CameraTest, ConvertingSphericalCoordinatesOnThePositiveXAxisToCartesianCoordinatesIsCorrect) {
  static constexpr gfx::SphericalCoordinates kSphericalCoordinates{.radius = kRadius, .theta = kHalfPi, .phi = 0.0f};
  const auto cartesian_position = ToCartesianCoordinates(kSphericalCoordinates);
  EXPECT_NEAR(cartesian_position.x, kRadius, kEpsilon);
  EXPECT_NEAR(cartesian_position.y, 0.0f, kEpsilon);
  EXPECT_NEAR(cartesian_position.z, 0.0f, kEpsilon);
}

TEST(CameraTest, ConvertingSphericalCoordinatesOnTheNegativeXAxisToCartesianCoordinatesIsCorrect) {
  static constexpr gfx::SphericalCoordinates kSphericalCoordinates{.radius = kRadius, .theta = -kHalfPi, .phi = 0.0f};
  const auto cartesian_position = ToCartesianCoordinates(kSphericalCoordinates);
  EXPECT_NEAR(cartesian_position.x, -kRadius, kEpsilon);
  EXPECT_NEAR(cartesian_position.y, 0.0f, kEpsilon);
  EXPECT_NEAR(cartesian_position.z, 0.0f, kEpsilon);
}

TEST(CameraTest, ConvertingSphericalCoordinatesOnThePositiveYAxisToCartesianCoordinatesIsCorrect) {
  static constexpr gfx::SphericalCoordinates kSphericalCoordinates{.radius = kRadius, .theta = 0.0f, .phi = -kHalfPi};
  const auto cartesian_position = ToCartesianCoordinates(kSphericalCoordinates);
  EXPECT_NEAR(cartesian_position.x, 0.0f, kEpsilon);
  EXPECT_NEAR(cartesian_position.y, kRadius, kEpsilon);
  EXPECT_NEAR(cartesian_position.z, 0.0f, kEpsilon);
}

TEST(CameraTest, ConvertingSphericalCoordinatesOnTheNegativeYAxisToCartesianCoordinatesIsCorrect) {
  static constexpr gfx::SphericalCoordinates kSphericalCoordinates{.radius = kRadius, .theta = 0.0f, .phi = kHalfPi};
  const auto cartesian_position = ToCartesianCoordinates(kSphericalCoordinates);
  EXPECT_NEAR(cartesian_position.x, 0.0f, kEpsilon);
  EXPECT_NEAR(cartesian_position.y, -kRadius, kEpsilon);
  EXPECT_NEAR(cartesian_position.z, 0.0f, kEpsilon);
}

}  // namespace
