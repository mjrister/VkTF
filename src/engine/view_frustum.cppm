module;

#include <algorithm>
#include <array>

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

export module view_frustum;

import bounding_box;

namespace vktf {

/**
 * @brief A view frustum for use in culling.
 * @details This class enables view frustum culling, a performance optimization that prevents rendering a mesh when not
 *          visible in the camera's view.
 */
export class [[nodiscard]] ViewFrustum {
public:
  /**
   * @brief Creates a @ref ViewFrustum.
   * @param view_projection_transform The view-projection matrix used to derive the view frustum planes in world space.
   */
  explicit ViewFrustum(const glm::mat4& view_projection_transform);

  /**
   * @brief Tests if a bounding box intersects with the view frustum.
   * @param bounding_box The bounding box to test for intersection with the view frustum.
   * @return @c true if the bounding box intersects with the view frustum, @c false otherwise.
   */
  [[nodiscard]] bool Intersects(const BoundingBox& bounding_box) const;

private:
  std::array<glm::vec4, 6> planes_;
};

}  // namespace vktf

module :private;

namespace vktf {

namespace {

glm::vec4 Normalize(const glm::vec4& plane) {
  const glm::vec3 normal{plane.x, plane.y, plane.z};
  const auto length = glm::length(normal);
  return plane / length;
}

std::array<glm::vec4, 6> GetPlanes(const glm::mat4& view_projection_transform) {
  return std::array{Normalize(view_projection_transform[3] + view_projection_transform[0]),   // left plane
                    Normalize(view_projection_transform[3] - view_projection_transform[0]),   // right plane
                    Normalize(view_projection_transform[3] + view_projection_transform[1]),   // top plane
                    Normalize(view_projection_transform[3] - view_projection_transform[1]),   // bottom plane
                    Normalize(view_projection_transform[2]),                                  // near plane
                    Normalize(view_projection_transform[3] - view_projection_transform[2])};  // far plane
}

}  // namespace

ViewFrustum::ViewFrustum(const glm::mat4& view_projection_transform)
    : planes_{GetPlanes(glm::transpose(view_projection_transform))}  // plane equations assume row-major order
{}

bool ViewFrustum::Intersects(const BoundingBox& bounding_box) const {
  return std::ranges::all_of(planes_, [&bounding_box](const auto& plane) {
    const glm::vec3 normal{plane.x, plane.y, plane.z};
    const glm::vec4 positive_vertex{normal.x >= 0.0f ? bounding_box.max.x : bounding_box.min.x,
                                    normal.y >= 0.0f ? bounding_box.max.y : bounding_box.min.y,
                                    normal.z >= 0.0f ? bounding_box.max.z : bounding_box.min.z,
                                    1.0f};
    return glm::dot(plane, positive_vertex) >= 0.0f;
  });
}

}  // namespace vktf
