module;

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/quaternion.hpp>

export module camera;

namespace vktf {

export struct [[nodiscard]] ViewFrustum {
  float field_of_view_y = 0.0f;
  float aspect_ratio = 0.0f;
  float z_near = 0.0f;
  float z_far = 0.0f;
};

export class [[nodiscard]] Camera {
public:
  Camera(const glm::vec3& world_position, const glm::vec3& direction, const ViewFrustum& view_frustum);

  [[nodiscard]] const glm::vec3& world_position() const noexcept { return world_position_; }

  [[nodiscard]] glm::mat4 GetViewTransform() const;
  [[nodiscard]] glm::mat4 GetProjectionTransform() const;

private:
  glm::vec3 world_position_;
  glm::quat orientation_;
  ViewFrustum view_frustum_;
};

}  // namespace vktf

module :private;

namespace vktf {

namespace {

constexpr glm::vec3 kWorldUp{0.0f, 1.0f, 0.0f};

}

Camera::Camera(const glm::vec3& world_position, const glm::vec3& direction, const ViewFrustum& view_frustum)
    : world_position_{world_position},
      orientation_{glm::quatLookAt(glm::normalize(direction), kWorldUp)},
      view_frustum_{view_frustum} {
  assert(glm::length(direction) > 0.0f);
}

glm::mat4 Camera::GetViewTransform() const {
  const auto rotation = glm::mat3_cast(glm::conjugate(orientation_));
  const auto translation = rotation * -world_position_;
  return glm::mat4{glm::vec4{rotation[0], 0.0},
                   glm::vec4{rotation[1], 0.0f},
                   glm::vec4{rotation[2], 0.0f},
                   glm::vec4{translation, 1.0f}};
}

glm::mat4 Camera::GetProjectionTransform() const {
  const auto& [field_of_view_y, aspect_ratio, z_near, z_far] = view_frustum_;
  auto projection_transform = glm::perspective(field_of_view_y, aspect_ratio, z_near, z_far);
  projection_transform[1][1] *= -1.0f;  // account for inverted y-axis convention in OpenGL
  return projection_transform;
}

}  // namespace vktf
