module;

#include <cassert>
#include <cstring>
#include <utility>

#include <vk_mem_alloc.h>
#include <vulkan/vulkan.hpp>

export module buffer;

import allocator;
import data_view;

namespace vktf {

export class Buffer {
public:
  Buffer(const std::size_t size_bytes,
         const vk::BufferUsageFlags buffer_usage_flags,
         const VmaAllocator allocator,
         const VmaAllocationCreateInfo& allocation_create_info = kDefaultAllocationCreateInfo);

  Buffer(const Buffer&) = delete;
  Buffer(Buffer&& buffer) noexcept { *this = std::move(buffer); }

  Buffer& operator=(const Buffer&) = delete;
  Buffer& operator=(Buffer&& buffer) noexcept;

  virtual ~Buffer() noexcept;

  [[nodiscard]] vk::Buffer operator*() const noexcept { return buffer_; }

protected:
  Buffer() noexcept = default;

  vk::Buffer buffer_;
  VmaAllocator allocator_ = nullptr;
  VmaAllocation allocation_ = nullptr;
};

export class HostVisibleBuffer final : public Buffer {
public:
  HostVisibleBuffer(const std::size_t size_bytes,
                    const vk::BufferUsageFlags buffer_usage_flags,
                    const VmaAllocator allocator)
      : Buffer{size_bytes, buffer_usage_flags, allocator, kHostVisibleAllocationCreateInfo}, size_bytes_{size_bytes} {}

  HostVisibleBuffer(const HostVisibleBuffer&) = delete;
  HostVisibleBuffer(HostVisibleBuffer&& host_visible_buffer) noexcept { *this = std::move(host_visible_buffer); }

  HostVisibleBuffer& operator=(const HostVisibleBuffer&) = delete;
  HostVisibleBuffer& operator=(HostVisibleBuffer&& host_visible_buffer) noexcept;

  ~HostVisibleBuffer() noexcept override { UnmapMemory(); }

  [[nodiscard]] auto size_bytes() const noexcept { return size_bytes_; }

  void MapMemory();
  void UnmapMemory() noexcept;

  template <typename T>
  void Copy(const DataView<const T> data_view) const {
    assert(mapped_memory_ != nullptr);
    assert(data_view.size_bytes() <= size_bytes_);
    memcpy(mapped_memory_, data_view.data(), data_view.size_bytes());
    const auto result = vmaFlushAllocation(allocator_, allocation_, 0, vk::WholeSize);
    vk::detail::resultCheck(static_cast<vk::Result>(result), "Flush allocation failed");
  }

private:
  static constexpr VmaAllocationCreateInfo kHostVisibleAllocationCreateInfo{
      .flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT,
      .usage = VMA_MEMORY_USAGE_AUTO};

  std::size_t size_bytes_ = 0;
  void* mapped_memory_ = nullptr;
};

export template <typename T>
[[nodiscard]] HostVisibleBuffer CreateStagingBuffer(const DataView<const T> data_view, const VmaAllocator allocator) {
  HostVisibleBuffer staging_buffer{data_view.size_bytes(), vk::BufferUsageFlagBits::eTransferSrc, allocator};
  staging_buffer.MapMemory();
  staging_buffer.Copy(data_view);
  staging_buffer.UnmapMemory();  // staging buffers are copied once so they can be unmapped immediately
  return staging_buffer;
}

export [[nodiscard]] Buffer CreateDeviceLocalBuffer(const HostVisibleBuffer& host_visible_buffer,
                                                    const vk::BufferUsageFlags buffer_usage_flags,
                                                    const vk::CommandBuffer command_buffer,
                                                    const VmaAllocator allocator) {
  static constexpr VmaAllocationCreateInfo kAllocationCreateInfo{.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE};
  Buffer device_local_buffer{host_visible_buffer.size_bytes(),
                             buffer_usage_flags | vk::BufferUsageFlagBits::eTransferDst,
                             allocator,
                             kAllocationCreateInfo};
  command_buffer.copyBuffer(*host_visible_buffer,
                            *device_local_buffer,
                            vk::BufferCopy{.size = host_visible_buffer.size_bytes()});
  return device_local_buffer;
}

}  // namespace vktf

module :private;

namespace vktf {

Buffer::Buffer(const std::size_t size_bytes,
               const vk::BufferUsageFlags buffer_usage_flags,
               const VmaAllocator allocator,
               const VmaAllocationCreateInfo& allocation_create_info)
    : allocator_{allocator} {
  const VkBufferCreateInfo buffer_create_info{.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
                                              .size = static_cast<VkDeviceSize>(size_bytes),
                                              .usage = static_cast<VkBufferUsageFlags>(buffer_usage_flags)};

  VkBuffer buffer = nullptr;
  const auto result =
      vmaCreateBuffer(allocator_, &buffer_create_info, &allocation_create_info, &buffer, &allocation_, nullptr);
  vk::detail::resultCheck(static_cast<vk::Result>(result), "Buffer creation failed");

  buffer_ = vk::Buffer{buffer};
}

Buffer& Buffer::operator=(Buffer&& buffer) noexcept {
  if (this != &buffer) {
    buffer_ = std::exchange(buffer.buffer_, nullptr);
    allocator_ = std::exchange(buffer.allocator_, nullptr);
    allocation_ = std::exchange(buffer.allocation_, nullptr);
  }
  return *this;
}

Buffer::~Buffer() noexcept {
  if (allocator_ != nullptr) {
    vmaDestroyBuffer(allocator_, buffer_, allocation_);
  }
}

HostVisibleBuffer& HostVisibleBuffer::operator=(HostVisibleBuffer&& host_visible_buffer) noexcept {
  if (this != &host_visible_buffer) {
    UnmapMemory();
    size_bytes_ = std::exchange(host_visible_buffer.size_bytes_, 0);
    mapped_memory_ = std::exchange(host_visible_buffer.mapped_memory_, nullptr);
    Buffer::operator=(std::move(host_visible_buffer));
  }
  return *this;
}

void HostVisibleBuffer::MapMemory() {
  if (mapped_memory_ == nullptr) {
    const auto result = vmaMapMemory(allocator_, allocation_, &mapped_memory_);
    vk::detail::resultCheck(static_cast<vk::Result>(result), "Map memory failed");
  }
}

void HostVisibleBuffer::UnmapMemory() noexcept {
  if (mapped_memory_ != nullptr) {
    vmaUnmapMemory(allocator_, allocation_);
    mapped_memory_ = nullptr;
  }
}

}  // namespace vktf
