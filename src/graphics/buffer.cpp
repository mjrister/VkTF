#include "graphics/buffer.h"

#include "graphics/device.h"

gfx::Buffer::Buffer(const vk::DeviceSize size,
                    const vk::BufferUsageFlags buffer_usage_flags,
                    const VmaAllocator allocator,
                    const VmaAllocationCreateFlags allocation_create_flags)
    : allocator_{allocator}, size_{size} {
  VkBufferCreateInfo buffer_create_info{};
  buffer_create_info.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
  buffer_create_info.size = size;
  buffer_create_info.usage = static_cast<VkBufferUsageFlags>(buffer_usage_flags);

  VmaAllocationCreateInfo allocation_create_info{};
  allocation_create_info.usage = VMA_MEMORY_USAGE_AUTO;
  allocation_create_info.flags = allocation_create_flags;

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

void gfx::Buffer::Copy(const Device& device, const Buffer& buffer) {
  device.SubmitOneTimeTransferCommandBuffer([&src_buffer = buffer, &dst_buffer = *this](const auto command_buffer) {
    assert(src_buffer.size_ <= dst_buffer.size_);
    command_buffer.copyBuffer(*src_buffer, *dst_buffer, vk::BufferCopy{.size = src_buffer.size_});
  });
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
