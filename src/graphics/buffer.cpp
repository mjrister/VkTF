#include "graphics/buffer.h"

gfx::Buffer::Buffer(const vk::DeviceSize size,
                    const vk::BufferUsageFlags buffer_usage_flags,
                    const VmaAllocator allocator,
                    const VmaAllocationCreateInfo& allocation_create_info)
    : allocator_{allocator}, size_{size} {
  const VkBufferCreateInfo buffer_create_info{.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
                                              .size = size,
                                              .usage = static_cast<VkBufferUsageFlags>(buffer_usage_flags)};

  VkBuffer buffer{};
  const auto result =
      vmaCreateBuffer(allocator_, &buffer_create_info, &allocation_create_info, &buffer, &allocation_, nullptr);
  vk::resultCheck(static_cast<vk::Result>(result), "Buffer creation failed");

  buffer_ = vk::Buffer{buffer};
}

gfx::Buffer& gfx::Buffer::operator=(Buffer&& buffer) noexcept {
  if (this != &buffer) {
    UnmapMemory();
    allocator_ = std::exchange(buffer.allocator_, {});
    allocation_ = std::exchange(buffer.allocation_, {});
    buffer_ = std::exchange(buffer.buffer_, {});
    size_ = std::exchange(buffer.size_, {});
    mapped_memory_ = std::exchange(buffer.mapped_memory_, {});
  }
  return *this;
}

gfx::Buffer::~Buffer() {
  if (allocator_ != nullptr) {
    UnmapMemory();
    vmaDestroyBuffer(allocator_, buffer_, allocation_);
  }
}

void* gfx::Buffer::MapMemory() {
  if (mapped_memory_ == nullptr) {
    const auto result = vmaMapMemory(allocator_, allocation_, &mapped_memory_);
    vk::resultCheck(static_cast<vk::Result>(result), "Map memory failed");
  }
  return mapped_memory_;
}

void gfx::Buffer::UnmapMemory() {
  if (mapped_memory_ != nullptr) {
    vmaUnmapMemory(allocator_, allocation_);
    mapped_memory_ = nullptr;
  }
}
