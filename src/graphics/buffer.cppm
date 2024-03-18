module;

#include <cassert>
#include <cstring>
#include <utility>

#include <vk_mem_alloc.h>
#include <vulkan/vulkan.hpp>

export module buffer;

namespace gfx {

export class Buffer {
public:
  Buffer(vk::DeviceSize size,
         vk::BufferUsageFlags buffer_usage_flags,
         VmaAllocator allocator,
         const VmaAllocationCreateInfo& allocation_create_info);

  Buffer(const Buffer&) = delete;
  Buffer(Buffer&& buffer) noexcept { *this = std::move(buffer); }

  Buffer& operator=(const Buffer&) = delete;
  Buffer& operator=(Buffer&& buffer) noexcept;

  ~Buffer() noexcept;

  [[nodiscard]] vk::Buffer operator*() const noexcept { return buffer_; }

  template <typename T>
  void Copy(const vk::ArrayProxy<const T> data) {
    assert(sizeof(T) * data.size() <= size_);
    auto* mapped_memory = MapMemory();
    memcpy(mapped_memory, data.data(), size_);
    const auto result = vmaFlushAllocation(allocator_, allocation_, 0, vk::WholeSize);
    vk::resultCheck(static_cast<vk::Result>(result), "Flush allocation failed");
  }

  template <typename T>
  void CopyOnce(const vk::ArrayProxy<const T> data) {
    Copy(data);
    UnmapMemory();
  }

private:
  void* MapMemory();
  void UnmapMemory() noexcept;

  vk::Buffer buffer_;
  vk::DeviceSize size_{};
  void* mapped_memory_{};
  VmaAllocator allocator_{};
  VmaAllocation allocation_{};
};

}  // namespace gfx

module :private;

namespace gfx {

Buffer::Buffer(const vk::DeviceSize size,
               const vk::BufferUsageFlags buffer_usage_flags,
               const VmaAllocator allocator,
               const VmaAllocationCreateInfo& allocation_create_info)
    : size_{size}, allocator_{allocator} {
  const VkBufferCreateInfo buffer_create_info{.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
                                              .size = size,
                                              .usage = static_cast<VkBufferUsageFlags>(buffer_usage_flags)};

  VkBuffer buffer{};
  const auto result =
      vmaCreateBuffer(allocator_, &buffer_create_info, &allocation_create_info, &buffer, &allocation_, nullptr);
  vk::resultCheck(static_cast<vk::Result>(result), "Buffer creation failed");
  buffer_ = buffer;
}

Buffer& Buffer::operator=(Buffer&& buffer) noexcept {
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
