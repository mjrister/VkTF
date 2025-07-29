module;

#include <cassert>
#include <optional>

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
  Camera(const glm::vec3& position, const glm::vec3& direction, const ViewFrustum& view_frustum);

  [[nodiscard]] const glm::vec3& position() const noexcept { return position_; }
  [[nodiscard]] const glm::quat& orientation() const noexcept { return orientation_; }
  [[nodiscard]] const glm::mat4& view_transform() const;
  [[nodiscard]] const glm::mat4& projection_transform() const;

  void Translate(const glm::vec3& translation);
  void Rotate(float pitch, float yaw);

private:
  glm::vec3 position_;
  glm::quat orientation_;
  ViewFrustum view_frustum_;
  mutable std::optional<glm::mat4> view_transform_;
  mutable std::optional<glm::mat4> projection_transform_;
};

}  // namespace vktf

module :private;

namespace vktf {

namespace {

constexpr glm::vec3 kWorldUp{0.0f, 1.0f, 0.0f};

glm::mat4 GetViewTransform(const glm::vec3& position, const glm::quat& orientation) {
  const auto view_rotation = glm::mat3_cast(glm::conjugate(orientation));
  const auto view_translation = view_rotation * -position;
  return glm::mat4{glm::vec4{view_rotation[0], 0.0f},
                   glm::vec4{view_rotation[1], 0.0f},
                   glm::vec4{view_rotation[2], 0.0f},
                   glm::vec4{view_translation, 1.0f}};
}

glm::mat4 GetProjectionTransform(const ViewFrustum& view_frustum) {
  const auto& [field_of_view_y, aspect_ratio, z_near, z_far] = view_frustum;
  auto projection_transform = glm::perspective(field_of_view_y, aspect_ratio, z_near, z_far);
  projection_transform[1][1] *= -1.0f;  // account for inverted y-axis convention in OpenGL
  return projection_transform;
}

}  // namespace

Camera::Camera(const glm::vec3& position, const glm::vec3& direction, const ViewFrustum& view_frustum)
    : position_{position},
      orientation_{glm::quatLookAt(glm::normalize(direction), kWorldUp)},
      view_frustum_{view_frustum} {
  assert(glm::length(direction) > 0.0f);
}

void Camera::Translate(const glm::vec3& translation) {
  position_ += orientation_ * translation;
  view_transform_ = std::nullopt;
}

void Camera::Rotate(const float pitch, const float yaw) {
  static constexpr glm::vec3 kLocalRight{1.0f, 0.0f, 0.0f};
  const auto pitch_rotation = glm::angleAxis(pitch, kLocalRight);
  const auto yaw_rotation = glm::angleAxis(yaw, kWorldUp);
  const auto orientation = yaw_rotation * orientation_ * pitch_rotation;
  orientation_ = glm::normalize(orientation);
  view_transform_ = std::nullopt;
}

const glm::mat4& Camera::view_transform() const {
  if (!view_transform_.has_value()) {
    view_transform_ = GetViewTransform(position_, orientation_);
  }
  return *view_transform_;
}

const glm::mat4& Camera::projection_transform() const {
  if (!projection_transform_.has_value()) {
    projection_transform_ = GetProjectionTransform(view_frustum_);
  }
  return *projection_transform_;
}

}  // namespace vktf
