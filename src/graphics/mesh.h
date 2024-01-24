#ifndef SRC_GRAPHICS_MESH_H_
#define SRC_GRAPHICS_MESH_H_

#include <span>

#include <glm/vec2.hpp>
#include <glm/vec3.hpp>
#include <vulkan/vulkan.hpp>

#include "graphics/buffer.h"
#include "graphics/device.h"

namespace gfx {

class Mesh {
public:
  struct Vertex {
    glm::vec3 position{0.0f};
    glm::vec2 texture_coordinates{0.0f};
    glm::vec3 normal{0.0f};
  };

  Mesh(const Device& device,
       const std::span<const Vertex> vertices,
       const std::span<const std::uint32_t> indices,
       const vk::DescriptorSet descriptor_set)
      : vertex_buffer_{CreateDeviceLocalBuffer<Vertex>(device, vk::BufferUsageFlagBits::eVertexBuffer, vertices)},
        index_buffer_{CreateDeviceLocalBuffer<std::uint32_t>(device, vk::BufferUsageFlagBits::eIndexBuffer, indices)},
        index_count_{static_cast<std::uint32_t>(indices.size())},
        descriptor_set_{descriptor_set} {}

  void Render(const vk::CommandBuffer command_buffer, const vk::PipelineLayout pipeline_layout) const {
    command_buffer.bindDescriptorSets(vk::PipelineBindPoint::eGraphics, pipeline_layout, 1, descriptor_set_, nullptr);
    command_buffer.bindVertexBuffers(0, *vertex_buffer_, static_cast<vk::DeviceSize>(0));
    command_buffer.bindIndexBuffer(*index_buffer_, 0, vk::IndexType::eUint32);
    command_buffer.drawIndexed(index_count_, 1, 0, 0, 0);
  }

private:
  Buffer vertex_buffer_;
  Buffer index_buffer_;
  std::uint32_t index_count_;
  vk::DescriptorSet descriptor_set_;
};

}  // namespace gfx

#endif  // SRC_GRAPHICS_MESH_H_
