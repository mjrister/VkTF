module;

#include <algorithm>
#include <array>
#include <limits>

#include <glm/glm.hpp>

export module bounding_box;

namespace vktf {

/**
 * @brief An axis-aligned bounding box.
 * @details An axis-aligned bounding box (AABB) represents the minimum extent needed to enclose a set of vertices for
 *          intersection tests. This can be efficiently represented using two points for the minimum and maximum corners
 *          of the bounding box from which all other properties of the bounding box can be derived.
 */
export struct [[nodiscard]] BoundingBox {
  /** @brief The minimum point of the axis-aligned bounding box. */
  glm::vec3 min{0.0f};

  /** @brief The maximum point of the axis-aligned bounding box. */
  glm::vec3 max{0.0f};
};

/**
 * @brief Transforms an axis-aligned bounding box to a new coordinate space.
 * @param bounding_box The bounding box to transform.
 * @param transform The transform matrix defining the basis vectors for the new coordinate space.
 * @return A new axis-aligned bounding box representing @p bounding_box transformed by the matrix @p transform.
 */
export [[nodiscard]] BoundingBox Transform(const BoundingBox& bounding_box, const glm::mat4& transform);

}  // namespace vktf

module :private;

namespace vktf {

BoundingBox Transform(const BoundingBox& bounding_box, const glm::mat4& transform) {
  const auto& [min_vertex, max_vertex] = bounding_box;
  const std::array local_vertices{glm::vec4{min_vertex.x, min_vertex.y, min_vertex.z, 1.0f},
                                  glm::vec4{max_vertex.x, min_vertex.y, min_vertex.z, 1.0f},
                                  glm::vec4{min_vertex.x, max_vertex.y, min_vertex.z, 1.0f},
                                  glm::vec4{max_vertex.x, max_vertex.y, min_vertex.z, 1.0f},
                                  glm::vec4{min_vertex.x, min_vertex.y, max_vertex.z, 1.0f},
                                  glm::vec4{max_vertex.x, min_vertex.y, max_vertex.z, 1.0f},
                                  glm::vec4{min_vertex.x, max_vertex.y, max_vertex.z, 1.0f},
                                  glm::vec4{max_vertex.x, max_vertex.y, max_vertex.z, 1.0f}};

  return std::ranges::fold_left(local_vertices,
                                BoundingBox{.min = glm::vec3{std::numeric_limits<float>::max()},
                                            .max = glm::vec3{std::numeric_limits<float>::min()}},
                                [&transform](const auto& transform_bounding_box, const auto& local_vertex) {
                                  const auto& [transform_min_vertex, transform_max_vertex] = transform_bounding_box;
                                  const glm::vec3 transform_vertex = transform * local_vertex;
                                  return BoundingBox{.min = glm::min(transform_min_vertex, transform_vertex),
                                                     .max = glm::max(transform_max_vertex, transform_vertex)};
                                });
}

}  // namespace vktf
