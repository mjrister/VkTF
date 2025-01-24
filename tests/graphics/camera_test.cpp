#include <algorithm>
#include <cassert>
#include <format>
#include <numbers>

#include <gtest/gtest.h>
#include <glm/glm.hpp>

import camera;

namespace {

constexpr auto kPi = std::numbers::pi_v<float>;
constexpr auto kHalfPi = kPi / 2.0f;
constexpr auto kQuarterPi = kPi / 4.0f;

constexpr glm::vec3 kOrigin{0.0f};
constexpr glm::vec3 kZeroVector{0.0f};
constexpr glm::vec3 kXAxis{1.0f, 0.0f, 0.0f};
constexpr glm::vec3 kYAxis{0.0f, 1.0f, 0.0f};
constexpr glm::vec3 kZAxis{0.0f, 0.0f, 1.0f};

constexpr auto kRight = kXAxis;
constexpr auto kLeft = -kXAxis;
constexpr auto kUp = kYAxis;
constexpr auto kDown = -kYAxis;
constexpr auto kForward = -kZAxis;
constexpr auto kBackward = kZAxis;

constexpr auto kDefaultPosition = kOrigin;
constexpr auto kDefaultDirection = kForward;
constexpr vktf::ViewFrustum kDefaultViewFrustum{.field_of_view_y = kQuarterPi,
                                                .aspect_ratio = 16.0f / 9.0f,
                                                .z_near = 0.1f,
                                                .z_far = 1.0e6f};

constexpr auto kEpsilon = 1.0e-5f;
constexpr auto* kInvalidTestNameChar = "?";  // emit a compiler error for unknown values when generating test names

class CameraTest : public testing::Test {
protected:
  vktf::Camera camera_{kDefaultPosition, kDefaultDirection, kDefaultViewFrustum};
};

class CameraTranslateTest : public CameraTest, public testing::WithParamInterface<glm::vec3> {};
class CameraRotateTest : public CameraTest, public testing::WithParamInterface<vktf::EulerAngles> {};

void ExpectNearEqual(const glm::vec3& lhs, const glm::vec3& rhs) {
  EXPECT_NEAR(lhs.x, rhs.x, kEpsilon);
  EXPECT_NEAR(lhs.y, rhs.y, kEpsilon);
  EXPECT_NEAR(lhs.z, rhs.z, kEpsilon);
}

void ExpectNearEqual(const vktf::EulerAngles& lhs, const vktf::EulerAngles& rhs) {
  EXPECT_NEAR(lhs.pitch, rhs.pitch, kEpsilon);
  EXPECT_NEAR(lhs.yaw, rhs.yaw, kEpsilon);
}

std::string GetDirectionName(const glm::vec3& direction) {
  if (direction == kZeroVector) return "ZeroVector";
  if (direction == kRight) return "Right";
  if (direction == kLeft) return "Left";
  if (direction == kUp) return "Up";
  if (direction == kDown) return "Down";
  if (direction == kForward) return "Forward";
  if (direction == kBackward) return "Backward";
  return kInvalidTestNameChar;
}

std::string GetAngleName(const float angle) {
  if (angle == 0.0f) return "Zero";
  if (angle == kQuarterPi) return "QuarterPi";
  if (angle == -kQuarterPi) return "MinusQuarterPi";
  if (angle == kHalfPi) return "HalfPi";
  if (angle == -kHalfPi) return "MinusHalfPi";
  return kInvalidTestNameChar;
}

TEST_P(CameraTranslateTest, TestPositionWhenTranslated) {
  const auto& translation = GetParam();
  const auto position = camera_.GetPosition() + translation;
  camera_.Translate(translation);
  ExpectNearEqual(position, camera_.GetPosition());
}

INSTANTIATE_TEST_SUITE_P(CameraTranslateTestSuite,
                         CameraTranslateTest,
                         testing::Values(kZeroVector, kRight, kLeft, kUp, kDown, kForward, kBackward),
                         [](const auto& test_param_info) { return GetDirectionName(test_param_info.param); });

TEST_P(CameraRotateTest, TestOrientationWhenRotated) {
  static constexpr auto kPitchLimit = glm::radians(89.0f);
  const auto& rotation = GetParam();
  camera_.Rotate(rotation);
  ExpectNearEqual(
      vktf::EulerAngles{.pitch = std::clamp(rotation.pitch, -kPitchLimit, kPitchLimit), .yaw = rotation.yaw},
      camera_.GetOrientation());
}

INSTANTIATE_TEST_SUITE_P(CameraRotateTestSuite,
                         CameraRotateTest,
                         testing::Values(vktf::EulerAngles{.pitch = 0.0f, .yaw = 0.0f},
                                         vktf::EulerAngles{.pitch = kQuarterPi, .yaw = 0.0f},
                                         vktf::EulerAngles{.pitch = -kQuarterPi, .yaw = 0.0f},
                                         vktf::EulerAngles{.pitch = kHalfPi, .yaw = 0.0f},
                                         vktf::EulerAngles{.pitch = -kHalfPi, .yaw = 0.0f},
                                         vktf::EulerAngles{.pitch = 0.0f, .yaw = kHalfPi},
                                         vktf::EulerAngles{.pitch = 0.0f, .yaw = -kHalfPi},
                                         vktf::EulerAngles{.pitch = kQuarterPi, .yaw = kHalfPi},
                                         vktf::EulerAngles{.pitch = kQuarterPi, .yaw = -kHalfPi},
                                         vktf::EulerAngles{.pitch = -kQuarterPi, .yaw = kHalfPi},
                                         vktf::EulerAngles{.pitch = -kQuarterPi, .yaw = -kHalfPi}),
                         [](const auto& test_param_info) {
                           const auto& rotation = test_param_info.param;
                           return std::format("Pitch{}Yaw{}", GetAngleName(rotation.pitch), GetAngleName(rotation.yaw));
                         });

TEST(CameraDeathTest, TestDebugAssertWhenCameraDirectionIsTheZeroVector) {
  EXPECT_DEBUG_DEATH({ (vktf::Camera{kDefaultPosition, kZeroVector, kDefaultViewFrustum}); }, "");
}

TEST(CameraDeathTest, TestDebugAssertWhenCameraDirectionIsCollinearWithUpDirection) {
  EXPECT_DEBUG_DEATH({ (vktf::Camera{kDefaultPosition, kUp, kDefaultViewFrustum}); }, "");
  EXPECT_DEBUG_DEATH({ (vktf::Camera{kDefaultPosition, kDown, kDefaultViewFrustum}); }, "");
}

}  // namespace
