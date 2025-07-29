module;

#include <glm/glm.hpp>

export module bounding_box;

namespace vktf {

export struct BoundingBox {
  glm::vec3 min{0.0f};
  glm::vec3 max{0.0f};
};

}  // namespace vktf
