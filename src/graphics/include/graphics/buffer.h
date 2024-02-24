#ifndef SRC_GRAPHICS_INCLUDE_GRAPHICS_BUFFER_H_
#define SRC_GRAPHICS_INCLUDE_GRAPHICS_BUFFER_H_

#include <cassert>
#include <cstring>
#include <utility>

#include <vk_mem_alloc.h>
#include <vulkan/vulkan.hpp>

namespace gfx {

class Buffer {
public:
  Buffer(vk::DeviceSize size,
         vk::BufferUsageFlags buffer_usage_flags,
         VmaAllocator allocator,
         const VmaAllocationCreateInfo& allocation_create_info);

  Buffer(const Buffer&) = delete;
  Buffer(Buffer&& buffer) noexcept { *this = std::move(buffer); }

  Buffer& operator=(const Buffer&) = delete;
  Buffer& operator=(Buffer&& buffer) noexcept;

  ~Buffer();

  [[nodiscard]] vk::Buffer operator*() const noexcept { return buffer_; }

  template <typename T>
  void Copy(const vk::ArrayProxy<const T> data) {
    // TODO(matthew-rister): switch to vmaCopyMemoryToAllocation when version 3.1.0 is released
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
  void UnmapMemory();

  VmaAllocator allocator_{};
  VmaAllocation allocation_{};
  vk::Buffer buffer_{};
  vk::DeviceSize size_{};
  void* mapped_memory_{};
};

}  // namespace gfx

#endif  // SRC_GRAPHICS_INCLUDE_GRAPHICS_BUFFER_H_
