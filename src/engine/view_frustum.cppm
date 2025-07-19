module;

#include <algorithm>
#include <array>

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
  std::array<glm::vec4, 6> planes_;
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
    : planes_{Normalize(view_projection_transform[3] + view_projection_transform[0]),  // left plane
              Normalize(view_projection_transform[3] - view_projection_transform[0]),  // right plane
              Normalize(view_projection_transform[3] + view_projection_transform[1]),  // bottom plane
              Normalize(view_projection_transform[3] - view_projection_transform[1]),  // top plane
              Normalize(view_projection_transform[2]),                                 // near plane
              Normalize(view_projection_transform[3] - view_projection_transform[2])}  // far plane
{}

bool ViewFrustum::Intersects(const Aabb& aabb) const {
  return std::ranges::all_of(planes_, [&aabb](const auto& plane) {
    const glm::vec3 positive_vertex{plane.x >= 0.0f ? aabb.max.x : aabb.min.x,
                                    plane.y >= 0.0f ? aabb.max.y : aabb.min.y,
                                    plane.z >= 0.0f ? aabb.max.z : aabb.min.z};
    return glm::dot(glm::vec3{plane}, positive_vertex) + plane.w >= 0.0f;
  });
}

}  // namespace vktf
