#include "graphics/mesh.h"

#include <cstdint>

#include "graphics/device.h"

namespace {

template <typename T>
[[nodiscard]] gfx::Buffer CreateBuffer(const gfx::Device& device,
                                       const VmaAllocator allocator,
                                       const vk::BufferUsageFlags buffer_usage_flags,
                                       const vk::ArrayProxy<const T> data) {
  const auto size_bytes = sizeof(T) * data.size();

  gfx::Buffer host_visible_buffer{size_bytes,
                                  vk::BufferUsageFlagBits::eTransferSrc,
                                  allocator,
                                  VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT};
  host_visible_buffer.Copy(data);

  gfx::Buffer device_local_buffer{size_bytes, buffer_usage_flags | vk::BufferUsageFlagBits::eTransferDst, allocator};
  device_local_buffer.Copy(device, host_visible_buffer);

  return device_local_buffer;
}

}  // namespace

gfx::Mesh::Mesh(const Device& device,
                const VmaAllocator allocator,
                const vk::ArrayProxy<const Vertex> vertices,
                const vk::ArrayProxy<const std::uint32_t> indices)
    : vertex_buffer_{CreateBuffer<Vertex>(device, allocator, vk::BufferUsageFlagBits::eVertexBuffer, vertices)},
      index_buffer_{CreateBuffer<std::uint32_t>(device, allocator, vk::BufferUsageFlagBits::eIndexBuffer, indices)},
      index_count_{indices.size()} {}
