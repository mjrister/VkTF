#ifndef SRC_GRAPHICS_INCLUDE_GRAPHICS_MESH_H_
#define SRC_GRAPHICS_INCLUDE_GRAPHICS_MESH_H_

#include <cstdint>
#include <utility>

#include <glm/vec2.hpp>
#include <glm/vec3.hpp>
#include <vulkan/vulkan.hpp>

#include "graphics/buffer.h"

namespace gfx {

struct Vertex {
  glm::vec3 position{0.0f};
  glm::vec2 texture_coordinates{0.0f};
  glm::vec3 normal{0.0f};
};

class Mesh {
public:
  Mesh(Buffer&& vertex_buffer, Buffer&& index_buffer, const std::uint32_t index_count) noexcept
      : vertex_buffer_{std::move(vertex_buffer)}, index_buffer_{std::move(index_buffer)}, index_count_{index_count} {}

  void Render(const vk::CommandBuffer command_buffer) const {
    command_buffer.bindVertexBuffers(0, *vertex_buffer_, static_cast<vk::DeviceSize>(0));
    command_buffer.bindIndexBuffer(*index_buffer_, 0, vk::IndexType::eUint32);
    command_buffer.drawIndexed(index_count_, 1, 0, 0, 0);
  }

private:
  Buffer vertex_buffer_;
  Buffer index_buffer_;
  std::uint32_t index_count_;
};

}  // namespace gfx

#endif  // SRC_GRAPHICS_INCLUDE_GRAPHICS_MESH_H_
