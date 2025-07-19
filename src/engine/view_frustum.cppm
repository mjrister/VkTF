module;

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

export module view_frustum;

import aabb;

namespace vktf {

export class [[nodiscard]] ViewFrustum {
public:
  struct [[nodiscard]] Properties {
    float field_of_view_y = 0.0f;
    float aspect_ratio = 0.0f;
    float z_near = 0.0f;
    float z_far = 0.0f;
  };

  explicit ViewFrustum(const glm::mat4& view_projection_transform);

  [[nodiscard]] bool Intersects(const Aabb& aabb) const;

private:
  struct Planes {
    glm::vec4 left{0.0f};
    glm::vec4 right{0.0f};
    glm::vec4 bottom{0.0f};
    glm::vec4 top{0.0f};
    glm::vec4 near{0.0f};
    glm::vec4 far{0.0f};
  };

  Planes planes_;
};

}  // namespace vktf

module :private;

namespace vktf {

namespace {

glm::vec4 Normalize(const glm::vec4& plane) {
  const auto length = glm::length(glm::vec3{plane});
  return plane / length;
}

}  // namespace

ViewFrustum::ViewFrustum(const glm::mat4& view_projection_transform)
    : planes_{Planes{.left = Normalize(view_projection_transform[3] + view_projection_transform[0]),
                     .right = Normalize(view_projection_transform[3] - view_projection_transform[0]),
                     .bottom = Normalize(view_projection_transform[3] + view_projection_transform[1]),
                     .top = Normalize(view_projection_transform[3] - view_projection_transform[1]),
                     .near = Normalize(view_projection_transform[2]),
                     .far = Normalize(view_projection_transform[3] - view_projection_transform[2])}} {}

bool ViewFrustum::Intersects(const Aabb& aabb) const {
  return true;  // TODO
}

}  // namespace vktf
