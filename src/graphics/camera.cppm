module;

#include <algorithm>
#include <cassert>
#include <cmath>

#include <glm/glm.hpp>
#include <glm/gtc/constants.hpp>
#include <glm/gtc/matrix_transform.hpp>

export module camera;

namespace gfx {

export struct ViewFrustum {
  float field_of_view_y{};
  float aspect_ratio{};
  float z_near{};
  float z_far{};
};

struct SphericalCoordinates {
  float radius{};
  float theta{};
  float phi{};
};

export class Camera {
public:
  Camera(const glm::vec3& position, const glm::vec3& direction, const ViewFrustum& view_frustum);

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

namespace {

gfx::SphericalCoordinates ToSphericalCoordinates(const glm::vec3& cartesian_coordinates) {
  const auto radius = glm::length(cartesian_coordinates);
  return radius == 0.0f
             ? gfx::SphericalCoordinates{.radius = 0.0f, .theta = 0.0f, .phi = 0.0f}
             : gfx::SphericalCoordinates{.radius = radius,
                                         .theta = std::atan2(cartesian_coordinates.x, cartesian_coordinates.z),
                                         .phi = std::asin(-cartesian_coordinates.y / radius)};
}

glm::vec3 ToCartesianCoordinates(const gfx::SphericalCoordinates& spherical_coordinates) {
  const auto [radius, theta, phi] = spherical_coordinates;
  const auto cos_phi = std::cos(phi);
  const auto x = radius * std::sin(theta) * cos_phi;
  const auto y = radius * std::sin(-phi);
  const auto z = radius * std::cos(theta) * cos_phi;
  return glm::vec3{x, y, z};
}

}  // namespace

namespace gfx {

Camera::Camera(const glm::vec3& position, const glm::vec3& direction, const ViewFrustum& view_frustum)
    : position_{position}, orientation_{ToSphericalCoordinates(-direction)}, view_frustum_{view_frustum} {
  assert(glm::length(direction) > 0.0f);
}

glm::mat4 Camera::GetViewTransform() const {
  static constexpr glm::vec3 kUp{0.0f, 1.0f, 0.0f};
  const auto direction = -ToCartesianCoordinates(orientation_);
  const auto target = position_ + direction;
  return glm::lookAt(position_, target, kUp);
}

glm::mat4 Camera::GetProjectionTransform() const {
  const auto [field_of_view_y, aspect_ratio, z_near, z_far] = view_frustum_;
  auto projection_transform = glm::perspective(field_of_view_y, aspect_ratio, z_near, z_far);
  projection_transform[1][1] *= -1;  // account for inverted y-axis convention in OpenGL
  return projection_transform;
}

void Camera::Translate(const float dx, const float dy, const float dz) {
  const glm::mat3 view_transform = GetViewTransform();
  position_ += glm::vec3{dx, dy, dz} * view_transform;
}

void Camera::Rotate(const float theta, const float phi) {
  static constexpr auto kThetaMax = glm::two_pi<float>();
  static constexpr auto kPhiMax = glm::radians(89.0f);
  orientation_.theta = std::fmod(orientation_.theta + theta, kThetaMax);
  orientation_.phi = std::clamp(orientation_.phi + phi, -kPhiMax, kPhiMax);
}

}  // namespace gfx
