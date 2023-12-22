#ifndef SRC_GRAPHICS_ARC_CAMERA_H_
#define SRC_GRAPHICS_ARC_CAMERA_H_

#include <glm/mat4x4.hpp>
#include <glm/vec3.hpp>

namespace gfx {

struct ViewFrustum {
  float field_of_view_y{};
  float aspect_ratio{};
  float z_near{};
  float z_far{};
};

struct SphericalCoordinates {
  float radius{};
  float theta{};
  float phi{};
};

class ArcCamera {
public:
  ArcCamera(const glm::vec3& target, const glm::vec3& position, const ViewFrustum& view_frustum);

  [[nodiscard]] const glm::mat4& view_transform() const noexcept { return view_transform_; }
  [[nodiscard]] const glm::mat4& projection_transform() const noexcept { return projection_transform_; }

  void Rotate(float theta, float phi);

private:
  glm::vec3 target_;
  SphericalCoordinates position_;
  glm::mat4 view_transform_;
  glm::mat4 projection_transform_;
};

}  // namespace gfx

#endif  // SRC_GRAPHICS_ARC_CAMERA_H_
