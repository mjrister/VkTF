#include <format>

#include <gtest/gtest.h>
#include <glm/glm.hpp>
#include <glm/gtc/constants.hpp>
#include <glm/gtc/quaternion.hpp>

import camera;

namespace {

// =====================================================================================================================
// Constants
// =====================================================================================================================

constexpr auto kHalfPi = glm::half_pi<float>();
constexpr auto kQuarterPi = glm::quarter_pi<float>();
constexpr auto kEpsilon = 1.0e-6f;
constexpr auto* kInvalidTestNameChar = "?";  // emit a compiler error for unknown values when generating test names

constexpr glm::vec3 kZeroVector{0.0f};
constexpr glm::vec3 kRight{1.0f, 0.0f, 0.0f};
constexpr glm::vec3 kUp{0.0f, 1.0f, 0.0f};
constexpr glm::vec3 kBackward{0.0f, 0.0f, 1.0f};
constexpr auto kLeft = -kRight;
constexpr auto kDown = -kUp;
constexpr auto kForward = -kBackward;

constexpr glm::vec3 kPosition{0.0f, 1.0f, 2.0f};
constexpr glm::vec3 kDirection{1.0f, 0.0f, 0.0f};
constexpr vktf::ViewFrustum kViewFrustum{.field_of_view_y = kHalfPi,
                                         .aspect_ratio = 16.0f / 9.0f,
                                         .z_near = 0.1f,
                                         .z_far = 1.0e6f};

// =====================================================================================================================
// Assertions
// =====================================================================================================================

template <glm::length_t N>
void ExpectNearEqual(const glm::vec<N, float>& lhs, const glm::vec<N, float>& rhs) {
  for (auto i = 0; i < N; ++i) {
    EXPECT_NEAR(lhs[i], rhs[i], kEpsilon) << std::format("Vector elements at {} are not equal", i);
  }
}

void ExpectNearEqual(const glm::quat& lhs, const glm::quat& rhs) {
  for (auto i = 0; i < glm::quat::length(); ++i) {
    EXPECT_NEAR(lhs[i], rhs[i], kEpsilon) << std::format("Quaternion elements at {} are not equal", i);
  }
}

template <glm::length_t N, glm::length_t M>
void ExpectNearEqual(const glm::mat<N, M, float>& lhs, const glm::mat<N, M, float>& rhs) {
  for (auto i = 0; i < N; ++i) {
    for (auto j = 0; j < M; ++j) {
      EXPECT_NEAR(lhs[i][j], rhs[i][j], kEpsilon) << std::format("Matrix elements at ({},{}) are not equal", i, j);
    }
  }
}

// =====================================================================================================================
// CameraTest
// =====================================================================================================================

class CameraTest : public testing::Test {
public:
  struct Result {
    glm::vec3 position;
    glm::quat orientation;
  };

protected:
  vktf::Camera camera_{kPosition, kDirection, kViewFrustum};
};

TEST_F(CameraTest, HasCorrectInitialWorldPositionAndOrientation) {
  const auto orientation = glm::angleAxis(-kHalfPi, kUp);
  ExpectNearEqual(kPosition, camera_.position());
  ExpectNearEqual(orientation, camera_.orientation());
}

TEST_F(CameraTest, HasCorrectInitialViewTransform) {
  const auto view_transform = glm::lookAt(kPosition, kPosition + kDirection, kUp);
  ExpectNearEqual(view_transform, camera_.view_transform());
}

TEST_F(CameraTest, HasCorrectInitialProjectionTransform) {
  const auto& [field_of_view_y, aspect_ratio, z_near, z_far] = kViewFrustum;
  auto projection_transform = glm::perspective(field_of_view_y, aspect_ratio, z_near, z_far);
  projection_transform[1][1] *= -1.0f;  // account for inverted y-axis convention in OpenGL
  ExpectNearEqual(projection_transform, camera_.projection_transform());
}

TEST(CameraDeathTest, AssertsWhenCameraDirectionIsZeroVector) {
  EXPECT_DEBUG_DEATH({ (std::ignore = vktf::Camera{kPosition, kZeroVector, kViewFrustum}); }, "");
}

// =====================================================================================================================
// CameraTranslateTest
// =====================================================================================================================

class CameraTranslateTest : public CameraTest, public testing::WithParamInterface<glm::vec3> {};

CameraTest::Result Translate(vktf::Camera& camera, const glm::vec3& translation) {
  const auto& orientation = camera.orientation();
  const auto position = camera.position() + orientation * translation;
  camera.Translate(translation);
  return CameraTest::Result{.position = position, .orientation = orientation};
}

TEST_P(CameraTranslateTest, HasCorrectWorldPositionAndOrientationWhenTranslated) {
  const auto& translation = GetParam();
  const auto& [position, orientation] = Translate(camera_, translation);
  ExpectNearEqual(position, camera_.position());
  ExpectNearEqual(orientation, camera_.orientation());
}

TEST_P(CameraTranslateTest, HasCorrectViewTransformWhenTranslated) {
  const auto& translation = GetParam();
  const auto& [position, orientation] = Translate(camera_, translation);
  const auto direction = orientation * kForward;
  const auto view_transform = glm::lookAt(position, position + direction, kUp);
  ExpectNearEqual(view_transform, camera_.view_transform());
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

INSTANTIATE_TEST_SUITE_P(CameraTranslateTestSuite,
                         CameraTranslateTest,
                         testing::Values(kZeroVector, kRight, kLeft, kUp, kDown, kForward, kBackward),
                         [](const auto& test_param_info) { return GetDirectionName(test_param_info.param); });

// =====================================================================================================================
// CameraRotateTest
// =====================================================================================================================

struct EulerAngles {
  float pitch = 0.0f;
  float yaw = 0.0f;
};

class CameraRotateTest : public CameraTest, public testing::WithParamInterface<EulerAngles> {};

CameraTest::Result Rotate(vktf::Camera& camera, const EulerAngles& euler_angles) {
  auto& [pitch, yaw] = euler_angles;
  const auto& position = camera.position();
  const auto orientation = glm::angleAxis(yaw, kUp) * camera.orientation() * glm::angleAxis(pitch, kRight);
  camera.Rotate(pitch, yaw);
  return CameraTest::Result{.position = position, .orientation = orientation};
}

TEST_P(CameraRotateTest, HasCorrectWorldPositionAndOrientationWhenRotated) {
  const auto& euler_angles = GetParam();
  const auto& [position, orientation] = Rotate(camera_, euler_angles);
  ExpectNearEqual(position, camera_.position());
  ExpectNearEqual(orientation, camera_.orientation());
}

TEST_P(CameraRotateTest, HasCorrectViewTransformWhenRotated) {
  const auto& euler_angles = GetParam();
  const auto& [position, orientation] = Rotate(camera_, euler_angles);
  const auto direction = orientation * kForward;
  const auto view_transform = glm::lookAt(position, position + direction, kUp);
  ExpectNearEqual(view_transform, camera_.view_transform());
}

std::string GetAngleName(const float angle) {
  if (angle == 0.0f) return "Zero";
  if (angle == kQuarterPi) return "QuarterPi";
  if (angle == -kQuarterPi) return "NegativeQuarterPi";
  if (angle == kHalfPi) return "HalfPi";
  if (angle == -kHalfPi) return "NegativeHalfPi";
  return kInvalidTestNameChar;
}

INSTANTIATE_TEST_SUITE_P(CameraRotateTestSuite,
                         CameraRotateTest,
                         testing::Values(EulerAngles{.pitch = 0.0f, .yaw = 0.0f},
                                         EulerAngles{.pitch = kQuarterPi, .yaw = 0.0f},
                                         EulerAngles{.pitch = -kQuarterPi, .yaw = 0.0f},
                                         EulerAngles{.pitch = 0.0f, .yaw = kQuarterPi},
                                         EulerAngles{.pitch = 0.0f, .yaw = -kQuarterPi},
                                         EulerAngles{.pitch = kQuarterPi, .yaw = kQuarterPi},
                                         EulerAngles{.pitch = kQuarterPi, .yaw = -kQuarterPi},
                                         EulerAngles{.pitch = -kQuarterPi, .yaw = kQuarterPi},
                                         EulerAngles{.pitch = -kQuarterPi, .yaw = -kQuarterPi}),
                         [](const auto& test_param_info) {
                           const auto& rotation = test_param_info.param;
                           return std::format("Pitch{}Yaw{}", GetAngleName(rotation.pitch), GetAngleName(rotation.yaw));
                         });

}  // namespace
