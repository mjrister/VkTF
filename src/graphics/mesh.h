#ifndef SRC_GRAPHICS_MESH_H_
#define SRC_GRAPHICS_MESH_H_

#include <vector>

#include <glm/vec3.hpp>
#include <vulkan/vulkan.hpp>

#include "graphics/buffer.h"
#include "graphics/device.h"

namespace gfx {

class Mesh {
public:
  struct Vertex {
    glm::vec3 position{0.0f};
    glm::vec3 normal{0.0f};
    glm::vec2 texture_coordinates{0.0f};
  };

  struct PushConstants {
    glm::mat4 model_transform{1.0f};
  };

  Mesh(const Device& device, const std::vector<Vertex>& vertices, const std::vector<std::uint32_t>& indices)
      : vertex_buffer_{CreateDeviceLocalBuffer<Vertex>(device, vk::BufferUsageFlagBits::eVertexBuffer, vertices)},
        index_buffer_{CreateDeviceLocalBuffer<std::uint32_t>(device, vk::BufferUsageFlagBits::eIndexBuffer, indices)},
        index_count_{static_cast<std::uint32_t>(indices.size())} {}

  void Render(const vk::CommandBuffer& command_buffer,
              const vk::PipelineLayout& pipeline_layout,
              const glm::mat4& transform) const {
    command_buffer.pushConstants<PushConstants>(pipeline_layout,
                                                vk::ShaderStageFlagBits::eVertex,
                                                0,
                                                PushConstants{.model_transform = transform});
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

#endif  // SRC_GRAPHICS_MESH_H_
