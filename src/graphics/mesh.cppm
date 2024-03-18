module;

#include <cstdint>
#include <utility>

#include <vulkan/vulkan.hpp>

export module mesh;

import buffer;

namespace gfx {

export class Mesh {
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
