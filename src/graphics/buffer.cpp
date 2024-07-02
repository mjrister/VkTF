#include "graphics/buffer.h"

namespace gfx {

Buffer::Buffer(const vk::DeviceSize size_bytes,
               const vk::BufferUsageFlags usage_flags,
               const VmaAllocator allocator,
               const VmaAllocationCreateInfo& allocation_create_info)
    : size_bytes_{size_bytes}, allocator_{allocator} {
  const VkBufferCreateInfo buffer_create_info{.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
                                              .size = size_bytes,
                                              .usage = static_cast<VkBufferUsageFlags>(usage_flags)};

  VkBuffer buffer = nullptr;
  const auto result =
      vmaCreateBuffer(allocator_, &buffer_create_info, &allocation_create_info, &buffer, &allocation_, nullptr);
  vk::resultCheck(static_cast<vk::Result>(result), "Buffer creation failed");
  buffer_ = vk::Buffer{buffer};
}

Buffer& Buffer::operator=(Buffer&& buffer) noexcept {
  if (this != &buffer) {
    UnmapMemory();
    buffer_ = std::exchange(buffer.buffer_, nullptr);
    size_bytes_ = std::exchange(buffer.size_bytes_, 0);
    mapped_memory_ = std::exchange(buffer.mapped_memory_, nullptr);
    allocator_ = std::exchange(buffer.allocator_, nullptr);
    allocation_ = std::exchange(buffer.allocation_, nullptr);
  }
  return *this;
}

Buffer::~Buffer() noexcept {
  if (allocator_ != nullptr) {
    UnmapMemory();
    vmaDestroyBuffer(allocator_, buffer_, allocation_);
  }
}

void* Buffer::MapMemory() {
  if (mapped_memory_ == nullptr) {
    const auto result = vmaMapMemory(allocator_, allocation_, &mapped_memory_);
    vk::resultCheck(static_cast<vk::Result>(result), "Map memory failed");
  }
  return mapped_memory_;
}

void Buffer::UnmapMemory() noexcept {
  if (mapped_memory_ != nullptr) {
    vmaUnmapMemory(allocator_, allocation_);
    mapped_memory_ = nullptr;
  }
}

}  // namespace gfx
