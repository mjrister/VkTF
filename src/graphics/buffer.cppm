module;

#include <cassert>
#include <cstring>
#include <utility>
#include <vector>

#include <vk_mem_alloc.h>
#include <vulkan/vulkan.hpp>

export module buffer;

import allocator;
import data_view;

namespace vktf {

export class Buffer {
public:
  Buffer(const vk::DeviceSize size_bytes,
         const vk::BufferUsageFlags usage_flags,
         const VmaAllocator allocator,
         const VmaAllocationCreateInfo& allocation_create_info = kDefaultAllocationCreateInfo);

  Buffer(const Buffer&) = delete;
  Buffer(Buffer&& buffer) noexcept { *this = std::move(buffer); }

  Buffer& operator=(const Buffer&) = delete;
  Buffer& operator=(Buffer&& buffer) noexcept;

  ~Buffer() noexcept;

  [[nodiscard]] vk::Buffer operator*() const noexcept { return buffer_; }

  template <typename T>
  void Copy(const DataView<const T> data_view) const {
    assert(mapped_memory_ != nullptr);
    assert(data_view.size_bytes() <= size_bytes_);
    memcpy(mapped_memory_, data_view.data(), data_view.size_bytes());
    const auto result = vmaFlushAllocation(allocator_, allocation_, 0, vk::WholeSize);
    vk::detail::resultCheck(static_cast<vk::Result>(result), "Flush allocation failed");
  }

  void MapMemory();
  void UnmapMemory() noexcept;

private:
  vk::Buffer buffer_;
  vk::DeviceSize size_bytes_ = 0;
  void* mapped_memory_ = nullptr;
  VmaAllocator allocator_ = nullptr;
  VmaAllocation allocation_ = nullptr;
};

export template <typename T>
[[nodiscard]] const Buffer& EmplaceStagingBuffer(const DataView<const T> data_view,
                                                 const VmaAllocator allocator,
                                                 std::vector<Buffer>& staging_buffers) {
  auto& staging_buffer = staging_buffers.emplace_back(data_view.size_bytes(),
                                                      vk::BufferUsageFlagBits::eTransferSrc,
                                                      allocator,
                                                      kHostVisibleAllocationCreateInfo);
  staging_buffer.MapMemory();
  staging_buffer.Copy(data_view);
  staging_buffer.UnmapMemory();  // staging buffers are copied once so they can be unmapped immediately

  return staging_buffer;
}

export template <typename T>
[[nodiscard]] Buffer CreateBuffer(const DataView<const T> data_view,
                                  const vk::BufferUsageFlags usage_flags,
                                  const vk::CommandBuffer command_buffer,
                                  const VmaAllocator allocator,
                                  std::vector<Buffer>& staging_buffers) {
  const auto& staging_buffer = EmplaceStagingBuffer(data_view, allocator, staging_buffers);
  Buffer buffer{data_view.size_bytes(), usage_flags | vk::BufferUsageFlagBits::eTransferDst, allocator};
  command_buffer.copyBuffer(*staging_buffer, *buffer, vk::BufferCopy{.size = data_view.size_bytes()});
  return buffer;
}

}  // namespace vktf

module :private;

namespace vktf {

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
  vk::detail::resultCheck(static_cast<vk::Result>(result), "Buffer creation failed");
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

void Buffer::MapMemory() {
  if (mapped_memory_ == nullptr) {
    const auto result = vmaMapMemory(allocator_, allocation_, &mapped_memory_);
    vk::detail::resultCheck(static_cast<vk::Result>(result), "Map memory failed");
  }
}

void Buffer::UnmapMemory() noexcept {
  if (mapped_memory_ != nullptr) {
    vmaUnmapMemory(allocator_, allocation_);
    mapped_memory_ = nullptr;
  }
}

}  // namespace vktf
