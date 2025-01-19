module;

#include <algorithm>
#include <cassert>
#include <cmath>
#include <limits>

#include <glm/glm.hpp>
#include <glm/gtc/constants.hpp>
#include <glm/gtc/matrix_transform.hpp>

export module camera;

namespace gfx {

export struct ViewFrustum {
  float field_of_view_y = 0.0f;
  float aspect_ratio = 0.0f;
  float z_near = 0.0f;
  float z_far = 0.0f;
};

export class Camera {
public:
  Camera(const glm::vec3& position, const glm::vec3& direction, const ViewFrustum& view_frustum);

  [[nodiscard]] const glm::mat4& view_transform() const noexcept { return view_transform_; }
  [[nodiscard]] const glm::mat4& projection_transform() const noexcept { return projection_transform_; }

  [[nodiscard]] glm::vec3 GetPosition() const;

  void Translate(float dx, float dy, float dz);
  void Rotate(float pitch, float yaw);

private:
  glm::mat4 view_transform_;
  glm::mat4 projection_transform_;
};

}  // namespace gfx

module :private;

namespace {

glm::mat4 GetViewTransform(const glm::vec3& position, const glm::vec3& direction) {
  static constexpr glm::vec3 kUp{0.0f, 1.0f, 0.0f};
  const auto target = position + direction;
  return glm::lookAt(position, target, kUp);
}

glm::mat4 GetProjectionTransform(const gfx::ViewFrustum& view_frustum) {
  const auto& [field_of_view_y, aspect_ratio, z_near, z_far] = view_frustum;
  auto projection_transform = z_far == std::numeric_limits<float>::infinity()
                                  ? glm::infinitePerspective(field_of_view_y, aspect_ratio, z_near)
                                  : glm::perspective(field_of_view_y, aspect_ratio, z_near, z_far);
  projection_transform[1][1] *= -1;  // account for inverted y-axis convention in OpenGL
  return projection_transform;
}

}  // namespace

namespace gfx {

Camera::Camera(const glm::vec3& position, const glm::vec3& direction, const ViewFrustum& view_frustum)
    : view_transform_{GetViewTransform(position, direction)},
      projection_transform_{GetProjectionTransform(view_frustum)} {
  assert(glm::length(direction) > 0.0f);
}

glm::vec3 Camera::GetPosition() const {
  const glm::vec3 translation{view_transform_[3]};
  const glm::mat3 rotation{view_transform_};
  return -translation * rotation;  // invert the view transform translation vector to get the camera world position
}

void Camera::Translate(const float dx, const float dy, const float dz) {
  view_transform_[3] -= glm::vec4{dx, dy, dz, 0.0f};
}

void Camera::Rotate(float pitch, float yaw) {
  // accumulate euler angles derived from the current camera orientation
  static constexpr auto kPitchLimit = glm::half_pi<float>() - std::numeric_limits<float>::epsilon();
  pitch = std::clamp(std::asin(-view_transform_[1][2]) + pitch, -kPitchLimit, kPitchLimit);  // avoid gimbal lock
  const auto cos_pitch = std::cos(pitch);
  const auto sin_pitch = std::sin(pitch);

  yaw += std::atan2(view_transform_[0][2], view_transform_[2][2]);
  const auto cos_yaw = std::cos(yaw);
  const auto sin_yaw = std::sin(yaw);

  // construct a cumulative rotation matrix to represent the next camera orientation
  const glm::mat3 rotation{
      // clang-format off
     cos_yaw, sin_yaw * sin_pitch,  sin_yaw * cos_pitch,
        0.0f,           cos_pitch,           -sin_pitch,
    -sin_yaw, cos_yaw * sin_pitch,  cos_yaw * cos_pitch
      // clang-format on
  };

  const auto translation = -GetPosition();
  view_transform_[0] = glm::vec4{rotation[0], 0.0f};
  view_transform_[1] = glm::vec4{rotation[1], 0.0f};
  view_transform_[2] = glm::vec4{rotation[2], 0.0f};
  view_transform_[3] = glm::vec4{rotation * translation, 1.0f};
}

}  // namespace gfx
