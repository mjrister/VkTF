module;

#include <cstdint>
#include <vector>

#include <vulkan/vulkan.hpp>

export module mesh;

import buffer;

namespace gfx {

export struct IndexBuffer {
  std::uint32_t index_count = 0;
  vk::IndexType index_type = vk::IndexType::eUint16;
  Buffer buffer;
};

export struct Primitive {
  Buffer vertex_buffer;
  IndexBuffer index_buffer;
  vk::DescriptorSet descriptor_set;
};

export class Mesh {
public:
  explicit Mesh(std::vector<Primitive> primitives) noexcept : primitives_{std::move(primitives)} {}

  void Render(const vk::PipelineLayout graphics_pipeline_layout, const vk::CommandBuffer command_buffer) const {
    for (const auto& [vertex_buffer, index_buffer, descriptor_set] : primitives_) {
      using enum vk::PipelineBindPoint;
      command_buffer.bindDescriptorSets(eGraphics, graphics_pipeline_layout, 1, descriptor_set, nullptr);
      command_buffer.bindVertexBuffers(0, *vertex_buffer, static_cast<vk::DeviceSize>(0));
      command_buffer.bindIndexBuffer(*index_buffer.buffer, 0, index_buffer.index_type);
      command_buffer.drawIndexed(index_buffer.index_count, 1, 0, 0, 0);
    }
  }

private:
  std::vector<Primitive> primitives_;
};

}  // namespace gfx
