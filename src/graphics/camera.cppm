module;

#include <algorithm>
#include <cassert>
#include <cmath>

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

export module camera;

import spherical_coordiantes;

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

  [[nodiscard]] const glm::vec3& position() const noexcept { return position_; }

  [[nodiscard]] glm::mat4 GetViewTransform() const;
  [[nodiscard]] glm::mat4 GetProjectionTransform() const;

  void Translate(float dx, float dy, float dz);
  void Rotate(float theta, float phi);

private:
  glm::vec3 position_;
  SphericalCoordinates orientation_;
  ViewFrustum view_frustum_;
};

}  // namespace gfx

module :private;

namespace gfx {

Camera::Camera(const glm::vec3& position, const glm::vec3& direction, const ViewFrustum& view_frustum)
    : position_{position}, orientation_{ToSphericalCoordinates(-direction)}, view_frustum_{view_frustum} {
  assert(orientation_.radius > 0.0f);
}

glm::mat4 Camera::GetViewTransform() const {
  static constexpr glm::vec3 kUp{0.0f, 1.0f, 0.0f};
  const auto direction = -ToCartesianCoordinates(orientation_);
  const auto target = position_ + direction;
  return glm::lookAt(position_, target, kUp);
}

glm::mat4 Camera::GetProjectionTransform() const {
  const auto& [field_of_view_y, aspect_ratio, z_near, z_far] = view_frustum_;
  auto projection_transform = glm::perspective(field_of_view_y, aspect_ratio, z_near, z_far);
  projection_transform[1][1] *= -1;  // account for inverted y-axis convention in OpenGL
  return projection_transform;
}

void Camera::Translate(const float dx, const float dy, const float dz) {
  const glm::mat3 orientation = GetViewTransform();
  position_ += glm::vec3{dx, dy, dz} * orientation;
}

void Camera::Rotate(const float theta, const float phi) {
  static constexpr auto kPhiMax = glm::two_pi<float>();
  static constexpr auto kThetaMax = glm::radians(89.0f);
  orientation_.theta = std::clamp(orientation_.theta + theta, -kThetaMax, kThetaMax);
  orientation_.phi = std::fmod(orientation_.phi + phi, kPhiMax);
}

}  // namespace gfx
