module;

#include <cassert>
#include <cstring>
#include <utility>

#include <vk_mem_alloc.h>
#include <vulkan/vulkan.hpp>

export module buffer;

import data_view;
import vma_allocator;

namespace vktf {

export class [[nodiscard]] Buffer {
public:
  struct [[nodiscard]] CreateInfo {
    vk::DeviceSize size_bytes = 0;
    vk::BufferUsageFlags usage_flags;
    const VmaAllocationCreateInfo& allocation_create_info;
  };

  Buffer(const vma::Allocator& allocator, const CreateInfo& create_info);

  Buffer(const Buffer&) = delete;
  Buffer(Buffer&& buffer) noexcept { *this = std::move(buffer); }

  Buffer& operator=(const Buffer&) = delete;
  Buffer& operator=(Buffer&& buffer) noexcept;

  virtual ~Buffer() noexcept;

  [[nodiscard]] vk::Buffer operator*() const noexcept { return buffer_; }

  [[nodiscard]] vk::DeviceSize size_bytes() const noexcept { return size_bytes_; }

  void Copy(const Buffer& src_buffer, const vk::CommandBuffer command_buffer);

protected:
  Buffer() noexcept = default;

  VmaAllocator allocator_ = nullptr;
  VmaAllocation allocation_ = nullptr;
  vk::Buffer buffer_;
  vk::DeviceSize size_bytes_ = 0;
  vk::BufferUsageFlags usage_flags_;
};

export class [[nodiscard]] HostVisibleBuffer final : public Buffer {
public:
  struct [[nodiscard]] CreateInfo {
    const vk::DeviceSize size_bytes = 0;
    const vk::BufferUsageFlags usage_flags;
  };

  HostVisibleBuffer(const vma::Allocator& allocator, const CreateInfo& create_info);

  HostVisibleBuffer(const HostVisibleBuffer&) = delete;
  HostVisibleBuffer(HostVisibleBuffer&& host_visible_buffer) noexcept { *this = std::move(host_visible_buffer); }

  HostVisibleBuffer& operator=(const HostVisibleBuffer&) = delete;
  HostVisibleBuffer& operator=(HostVisibleBuffer&& host_visible_buffer) noexcept;

  ~HostVisibleBuffer() noexcept override { UnmapMemory(); }

  void MapMemory();
  void UnmapMemory() noexcept;

  template <typename T>
  void Copy(const DataView<T> data_view) {
    assert(mapped_memory_ != nullptr);
    assert(data_view.size_bytes() <= size_bytes_);
    memcpy(mapped_memory_, data_view.data(), data_view.size_bytes());
    const auto result = vmaFlushAllocation(allocator_, allocation_, 0, vk::WholeSize);
    vk::detail::resultCheck(static_cast<vk::Result>(result), "Flush allocation failed");
  }

private:
  void* mapped_memory_ = nullptr;
};

export template <typename T>
[[nodiscard]] HostVisibleBuffer CreateStagingBuffer(const vma::Allocator& allocator, const DataView<T> data_view) {
  HostVisibleBuffer staging_buffer{allocator,
                                   HostVisibleBuffer::CreateInfo{.size_bytes = data_view.size_bytes(),
                                                                 .usage_flags = vk::BufferUsageFlagBits::eTransferSrc}};
  staging_buffer.MapMemory();
  staging_buffer.Copy(data_view);
  staging_buffer.UnmapMemory();  // staging buffers are copied once so they can be unmapped immediately
  return staging_buffer;
}

export [[nodiscard]] Buffer CreateDeviceLocalBuffer(const vma::Allocator& allocator,
                                                    const vk::CommandBuffer command_buffer,
                                                    const HostVisibleBuffer& host_visible_buffer,
                                                    const vk::BufferUsageFlags usage_flags) {
  Buffer device_local_buffer{allocator,
                             Buffer::CreateInfo{.size_bytes = host_visible_buffer.size_bytes(),
                                                .usage_flags = usage_flags | vk::BufferUsageFlagBits::eTransferDst,
                                                .allocation_create_info = vma::kDeviceLocalAllocationCreateInfo}};
  device_local_buffer.Copy(host_visible_buffer, command_buffer);
  return device_local_buffer;
}

}  // namespace vktf

module :private;

namespace vktf {

namespace {

std::pair<VmaAllocation, vk::Buffer> CreateBuffer(const VmaAllocator allocator, const Buffer::CreateInfo& create_info) {
  const auto& [size_bytes, usage_flags, allocation_create_info] = create_info;
  const VkBufferCreateInfo buffer_create_info{.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
                                              .size = size_bytes,
                                              .usage = static_cast<VkBufferUsageFlags>(usage_flags)};

  VkBuffer buffer = nullptr;
  VmaAllocation allocation = nullptr;
  const auto result =
      vmaCreateBuffer(allocator, &buffer_create_info, &allocation_create_info, &buffer, &allocation, nullptr);
  vk::detail::resultCheck(static_cast<vk::Result>(result), "Buffer creation failed");

  return std::pair{allocation, buffer};
}

void DestroyBuffer(const VmaAllocator allocator, const vk::Buffer buffer, const VmaAllocation allocation) noexcept {
  if (allocator != nullptr) {
    vmaDestroyBuffer(allocator, buffer, allocation);
  }
}

}  // namespace

Buffer::Buffer(const vma::Allocator& allocator, const CreateInfo& create_info)
    : allocator_{*allocator}, size_bytes_{create_info.size_bytes}, usage_flags_{create_info.usage_flags} {
  std::tie(allocation_, buffer_) = CreateBuffer(*allocator, create_info);
}

Buffer& Buffer::operator=(Buffer&& buffer) noexcept {
  if (this != &buffer) {
    DestroyBuffer(allocator_, buffer_, allocation_);
    buffer_ = std::exchange(buffer.buffer_, nullptr);
    usage_flags_ = std::exchange(buffer.usage_flags_, {});
    size_bytes_ = std::exchange(buffer.size_bytes_, 0);
    allocator_ = std::exchange(buffer.allocator_, nullptr);
    allocation_ = std::exchange(buffer.allocation_, nullptr);
  }
  return *this;
}

Buffer::~Buffer() noexcept { DestroyBuffer(allocator_, buffer_, allocation_); }

void Buffer::Copy(const Buffer& src_buffer, const vk::CommandBuffer command_buffer) {
  const auto& dst_buffer = *this;
  assert(src_buffer.size_bytes_ <= dst_buffer.size_bytes_);
  assert(src_buffer.usage_flags_ & vk::BufferUsageFlagBits::eTransferSrc);
  assert(dst_buffer.usage_flags_ & vk::BufferUsageFlagBits::eTransferDst);
  command_buffer.copyBuffer(*src_buffer, *dst_buffer, vk::BufferCopy{.size = src_buffer.size_bytes_});
}

HostVisibleBuffer::HostVisibleBuffer(const vma::Allocator& allocator, const CreateInfo& create_info)
    : Buffer{allocator,
             Buffer::CreateInfo{.size_bytes = create_info.size_bytes,
                                .usage_flags = create_info.usage_flags,
                                .allocation_create_info = vma::kHostVisibleAllocationCreateInfo}} {}

HostVisibleBuffer& HostVisibleBuffer::operator=(HostVisibleBuffer&& host_visible_buffer) noexcept {
  if (this != &host_visible_buffer) {
    UnmapMemory();
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
