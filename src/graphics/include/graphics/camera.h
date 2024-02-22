#ifndef SRC_GRAPHICS_INCLUDE_GRAPHICS_CAMERA_H_
#define SRC_GRAPHICS_INCLUDE_GRAPHICS_CAMERA_H_

#include <glm/mat4x4.hpp>
#include <glm/vec3.hpp>

#include "math/spherical_coordinates.h"

namespace gfx {

struct ViewFrustum {
  float field_of_view_y{};
  float aspect_ratio{};
  float z_near{};
  float z_far{};
};

class Camera {
public:
  Camera(const glm::vec3& position, const glm::vec3& direction, const ViewFrustum& view_frustum);

  [[nodiscard]] glm::mat4 GetViewTransform() const;
  [[nodiscard]] glm::mat4 GetProjectionTransform() const;

  void Translate(float dx, float dy, float dz);
  void Rotate(float theta, float phi);

private:
  glm::vec3 position_{0.0f};
  SphericalCoordinates orientation_{};
  ViewFrustum view_frustum_{};
};

}  // namespace gfx

#endif  // SRC_GRAPHICS_INCLUDE_GRAPHICS_CAMERA_H_
