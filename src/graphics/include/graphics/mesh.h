#ifndef SRC_GRAPHICS_INCLUDE_GRAPHICS_MESH_H_
#define SRC_GRAPHICS_INCLUDE_GRAPHICS_MESH_H_

#include <cstdint>
#include <utility>

#include <vulkan/vulkan.hpp>

#include "graphics/buffer.h"

namespace gfx {

struct IndexBuffer {
  std::uint32_t index_count = 0;
  vk::IndexType index_type = vk::IndexType::eUint16;
  Buffer buffer;
};

class Mesh {
public:
  Mesh(Buffer&& vertex_buffer, IndexBuffer&& index_buffer, const vk::DescriptorSet descriptor_set) noexcept
      : vertex_buffer_{std::move(vertex_buffer)},
        index_buffer_{std::move(index_buffer)},
        descriptor_set_{descriptor_set} {}

  void Render(const vk::PipelineLayout pipeline_layout, const vk::CommandBuffer command_buffer) const {
    command_buffer.bindDescriptorSets(vk::PipelineBindPoint::eGraphics, pipeline_layout, 0, descriptor_set_, nullptr);
    command_buffer.bindVertexBuffers(0, *vertex_buffer_, static_cast<vk::DeviceSize>(0));
    command_buffer.bindIndexBuffer(*index_buffer_.buffer, 0, index_buffer_.index_type);
    command_buffer.drawIndexed(index_buffer_.index_count, 1, 0, 0, 0);
  }

private:
  Buffer vertex_buffer_;
  IndexBuffer index_buffer_;
  vk::DescriptorSet descriptor_set_;
};

}  // namespace gfx

#endif  // SRC_GRAPHICS_INCLUDE_GRAPHICS_MESH_H_
