module;

#include <algorithm>
#include <cassert>
#include <cmath>
#include <limits>

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

export module camera;

namespace gfx {

export struct ViewFrustum {
  float field_of_view_y = 0.0f;
  float aspect_ratio = 0.0f;
  float z_near = 0.0f;
  float z_far = 0.0f;
};

export struct EulerAngles {
  float pitch = 0.0f;
  float yaw = 0.0f;
};

export class Camera {
public:
  Camera(const glm::vec3& position, const glm::vec3& direction, const ViewFrustum& view_frustum);

  [[nodiscard]] const glm::mat4& view_transform() const noexcept { return view_transform_; }
  [[nodiscard]] const glm::mat4& projection_transform() const noexcept { return projection_transform_; }

  [[nodiscard]] glm::vec3 GetPosition() const;
  [[nodiscard]] EulerAngles GetOrientation() const;

  void Translate(const glm::vec3& translation) { view_transform_[3] -= glm::vec4{translation, 0.0f}; }
  void Rotate(const EulerAngles& rotation);

private:
  glm::mat4 view_transform_;
  glm::mat4 projection_transform_;
};

}  // namespace gfx

module :private;

namespace {

constexpr glm::vec3 kUp{0.0f, 1.0f, 0.0f};

glm::mat4 GetViewTransform(const glm::vec3& position, const glm::vec3& direction) {
  const auto target = position + direction;
  return glm::lookAt(position, target, kUp);
}

glm::mat4 GetProjectionTransform(const gfx::ViewFrustum& view_frustum) {
  const auto& [field_of_view_y, aspect_ratio, z_near, z_far] = view_frustum;
  auto projection_transform = z_far == std::numeric_limits<float>::infinity()
                                  ? glm::infinitePerspective(field_of_view_y, aspect_ratio, z_near)
                                  : glm::perspective(field_of_view_y, aspect_ratio, z_near, z_far);
  projection_transform[1][1] *= -1.0f;  // account for inverted y-axis convention in OpenGL
  return projection_transform;
}

gfx::EulerAngles operator+(const gfx::EulerAngles& orientation, const gfx::EulerAngles rotation) {
  static constexpr auto kPitchLimit = glm::radians(89.0f);
  return gfx::EulerAngles{.pitch = std::clamp(orientation.pitch + rotation.pitch, -kPitchLimit, kPitchLimit),
                          .yaw = orientation.yaw + rotation.yaw};
}
}  // namespace

namespace gfx {

Camera::Camera(const glm::vec3& position, const glm::vec3& direction, const ViewFrustum& view_frustum)
    : view_transform_{GetViewTransform(position, direction)},
      projection_transform_{GetProjectionTransform(view_frustum)} {
  assert(glm::length(direction) > 0.0f);
  assert(std::abs(glm::dot(direction, kUp)) < 1.0f);
}

glm::vec3 Camera::GetPosition() const {
  const glm::vec3 translation{view_transform_[3]};
  const glm::mat3 rotation{view_transform_};
  return -translation * rotation;  // invert the view-space translation vector to get the world-space position
}

EulerAngles Camera::GetOrientation() const {
  return EulerAngles{.pitch = std::asin(-view_transform_[1][2]),
                     .yaw = std::atan2(view_transform_[0][2], view_transform_[2][2])};
}

void Camera::Rotate(const EulerAngles& rotation) {
  const auto [pitch, yaw] = GetOrientation() + rotation;
  const auto cos_pitch = std::cos(pitch);
  const auto sin_pitch = std::sin(pitch);
  const auto cos_yaw = std::cos(yaw);
  const auto sin_yaw = std::sin(yaw);

  const glm::mat3 euler_rotation{
      // clang-format off
     cos_yaw, sin_yaw * sin_pitch,  sin_yaw * cos_pitch,
        0.0f,           cos_pitch,           -sin_pitch,
    -sin_yaw, cos_yaw * sin_pitch,  cos_yaw * cos_pitch
      // clang-format on
  };

  const auto translation = -GetPosition();
  view_transform_[0] = glm::vec4{euler_rotation[0], 0.0f};
  view_transform_[1] = glm::vec4{euler_rotation[1], 0.0f};
  view_transform_[2] = glm::vec4{euler_rotation[2], 0.0f};
  view_transform_[3] = glm::vec4{euler_rotation * translation, 1.0f};
}

}  // namespace gfx
