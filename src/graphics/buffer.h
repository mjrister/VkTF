#ifndef GRAPHICS_BUFFER_H_
#define GRAPHICS_BUFFER_H_

#include <cassert>
#include <cstring>
#include <utility>

#include <vk_mem_alloc.h>
#include <vulkan/vulkan.hpp>

namespace gfx {

class Buffer {
public:
  Buffer(vk::DeviceSize buffer_size_bytes,
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
  void Copy(const vk::ArrayProxy<const T> buffer_data) {
    assert(sizeof(T) * buffer_data.size() <= buffer_size_bytes_);
    auto* mapped_memory = MapMemory();
    memcpy(mapped_memory, buffer_data.data(), buffer_size_bytes_);
    const auto result = vmaFlushAllocation(allocator_, allocation_, 0, vk::WholeSize);
    vk::resultCheck(static_cast<vk::Result>(result), "Flush allocation failed");
  }

  template <typename T>
  void CopyOnce(const vk::ArrayProxy<const T> buffer_data) {
    Copy(buffer_data);
    UnmapMemory();
  }

private:
  void* MapMemory();
  void UnmapMemory() noexcept;

  vk::Buffer buffer_;
  vk::DeviceSize buffer_size_bytes_ = 0;
  void* mapped_memory_ = nullptr;
  VmaAllocator allocator_ = nullptr;
  VmaAllocation allocation_ = nullptr;
};

}  // namespace gfx

#endif  // GRAPHICS_BUFFER_H_
