#ifndef SRC_MATH_INCLUDE_MATH_SPHERICAL_COORDINATES_H_
#define SRC_MATH_INCLUDE_MATH_SPHERICAL_COORDINATES_H_

#include <glm/fwd.hpp>

namespace gfx {

struct SphericalCoordinates {
  float radius = 0.0f;
  float theta = 0.0f;
  float phi = 0.0f;
};

SphericalCoordinates ToSphericalCoordinates(const glm::vec3& cartesian_coordinates);
glm::vec3 ToCartesianCoordinates(const SphericalCoordinates& spherical_coordinates);

}  // namespace gfx

#endif  // SRC_MATH_INCLUDE_MATH_SPHERICAL_COORDINATES_H_
