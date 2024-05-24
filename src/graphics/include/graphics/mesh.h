#ifndef SRC_GRAPHICS_INCLUDE_GRAPHICS_MESH_H_
#define SRC_GRAPHICS_INCLUDE_GRAPHICS_MESH_H_

#include <cstdint>
#include <utility>

#include <vulkan/vulkan.hpp>

#include "graphics/buffer.h"

namespace gfx {

class Mesh {
public:
  Mesh(Buffer&& vertex_buffer,
       Buffer&& index_buffer,
       const std::uint32_t index_count,
       const vk::IndexType index_type,
       const vk::DescriptorSet descriptor_set) noexcept
      : vertex_buffer_{std::move(vertex_buffer)},
        index_buffer_{std::move(index_buffer)},
        index_count_{index_count},
        index_type_{index_type},
        descriptor_set_{descriptor_set} {}

  void Render(const vk::PipelineLayout pipeline_layout, const vk::CommandBuffer command_buffer) const {
    command_buffer.bindDescriptorSets(vk::PipelineBindPoint::eGraphics, pipeline_layout, 0, descriptor_set_, nullptr);
    command_buffer.bindVertexBuffers(0, *vertex_buffer_, static_cast<vk::DeviceSize>(0));
    command_buffer.bindIndexBuffer(*index_buffer_, 0, index_type_);
    command_buffer.drawIndexed(index_count_, 1, 0, 0, 0);
  }

private:
  Buffer vertex_buffer_;
  Buffer index_buffer_;
  std::uint32_t index_count_;
  vk::IndexType index_type_;
  vk::DescriptorSet descriptor_set_;
};

}  // namespace gfx

#endif  // SRC_GRAPHICS_INCLUDE_GRAPHICS_MESH_H_
