#include "graphics/camera.h"

#include <algorithm>
#include <cassert>
#include <cmath>

#include <glm/glm.hpp>
#include <glm/gtc/constants.hpp>
#include <glm/gtc/matrix_transform.hpp>

gfx::Camera::Camera(const glm::vec3& position, const glm::vec3& direction, const ViewFrustum& view_frustum)
    : position_{position}, view_frustum_{view_frustum} {
  assert(glm::length(direction) > 0.0f);
  orientation_ = ToSphericalCoordinates(glm::normalize(-direction));
}

glm::mat4 gfx::Camera::GetViewTransform() const {
  static constexpr glm::vec3 kUp{0.0f, 1.0f, 0.0f};
  const auto direction = -gfx::ToCartesianCoordinates(orientation_);
  const auto target = position_ + direction;
  return glm::lookAt(position_, target, kUp);
}

glm::mat4 gfx::Camera::GetProjectionTransform() const {
  const auto [field_of_view_y, aspect_ratio, z_near, z_far] = view_frustum_;
  auto projection_transform = glm::perspective(field_of_view_y, aspect_ratio, z_near, z_far);
  projection_transform[1][1] *= -1;  // account for inverted y-axis convention in OpenGL
  return projection_transform;
}

void gfx::Camera::Translate(const float dx, const float dy, const float dz) {
  const glm::mat3 view_transform = GetViewTransform();
  position_ += glm::vec3{dx, dy, dz} * view_transform;
}

void gfx::Camera::Rotate(const float theta, const float phi) {
  static constexpr auto kThetaMax = glm::two_pi<float>();
  static constexpr auto kPhiMax = glm::radians(89.0f);
  orientation_.theta = std::fmodf(orientation_.theta + theta, kThetaMax);
  orientation_.phi = std::clamp(orientation_.phi + phi, -kPhiMax, kPhiMax);
}
