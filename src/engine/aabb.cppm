module;

#include <glm/glm.hpp>

export module aabb;

namespace vktf {

export struct Aabb {
  glm::vec3 min{0.0f};
  glm::vec3 max{0.0f};
};

}  // namespace vktf
