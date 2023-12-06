#ifndef SRC_GRAPHICS_ARCBALL_H_
#define SRC_GRAPHICS_ARCBALL_H_

#include <optional>
#include <utility>

#include <glm/fwd.hpp>

namespace qem::arcball {

/**
 * \brief Gets the axis and angle to rotate a mesh using changes in cursor position.
 * \param cursor_position_start The starting cursor position.
 * \param cursor_position_end The ending cursor position.
 * \param window_dimensions The window width and height.
 * \return The axis (in camera space) and angle to rotate the mesh if the angle between the arcball positions of
 *         \p cursor_position_start and \p cursor_position_end is nonzero, otherwise \c std::nullopt.
 * \see https://en.wikibooks.org/wiki/OpenGL_Programming/Modern_OpenGL_Tutorial_Arcball
 */
std::optional<std::pair<glm::vec3, float>> GetRotation(const glm::dvec2& cursor_position_start,
                                                       const glm::dvec2& cursor_position_end,
                                                       const std::pair<int, int>& window_dimensions);

}  // namespace qem::arcball

#endif  // SRC_GRAPHICS_ARCBALL_H_
