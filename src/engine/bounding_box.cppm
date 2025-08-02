module;

#include <algorithm>
#include <array>
#include <limits>

#include <glm/glm.hpp>

export module bounding_box;

namespace vktf {

export struct [[nodiscard]] BoundingBox {
  glm::vec3 min{0.0f};
  glm::vec3 max{0.0f};
};

export [[nodiscard]] BoundingBox Transform(const BoundingBox& local_bounding_box, const glm::mat4& world_transform);

}  // namespace vktf

module :private;

namespace vktf {

BoundingBox Transform(const BoundingBox& local_bounding_box, const glm::mat4& world_transform) {
  const auto& [local_min_vertex, local_max_vertex] = local_bounding_box;
  const std::array local_vertices{glm::vec4{local_min_vertex.x, local_min_vertex.y, local_min_vertex.z, 1.0f},
                                  glm::vec4{local_max_vertex.x, local_min_vertex.y, local_min_vertex.z, 1.0f},
                                  glm::vec4{local_min_vertex.x, local_max_vertex.y, local_min_vertex.z, 1.0f},
                                  glm::vec4{local_max_vertex.x, local_max_vertex.y, local_min_vertex.z, 1.0f},
                                  glm::vec4{local_min_vertex.x, local_min_vertex.y, local_max_vertex.z, 1.0f},
                                  glm::vec4{local_max_vertex.x, local_min_vertex.y, local_max_vertex.z, 1.0f},
                                  glm::vec4{local_min_vertex.x, local_max_vertex.y, local_max_vertex.z, 1.0f},
                                  glm::vec4{local_max_vertex.x, local_max_vertex.y, local_max_vertex.z, 1.0f}};

  return std::ranges::fold_left(local_vertices,
                                BoundingBox{.min = glm::vec3{std::numeric_limits<float>::max()},
                                            .max = glm::vec3{std::numeric_limits<float>::min()}},
                                [&world_transform](const auto& world_bounding_box, const auto& local_vertex) {
                                  const auto& [world_min_vertex, world_max_vertex] = world_bounding_box;
                                  const glm::vec3 world_vertex = world_transform * local_vertex;
                                  return BoundingBox{.min = glm::min(world_min_vertex, world_vertex),
                                                     .max = glm::max(world_max_vertex, world_vertex)};
                                });
}

}  // namespace vktf
