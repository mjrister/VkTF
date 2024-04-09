#ifndef SRC_GRAPHICS_INCLUDE_GRAPHICS_ALLOCATOR_H_
#define SRC_GRAPHICS_INCLUDE_GRAPHICS_ALLOCATOR_H_

#include <utility>

#include <vk_mem_alloc.h>
#include <vulkan/vulkan.hpp>

namespace gfx {

class Allocator {
public:
  Allocator(const vk::Instance instance, const vk::PhysicalDevice physical_device, const vk::Device device);

  Allocator(const Allocator&) = delete;
  Allocator(Allocator&& allocator) noexcept { *this = std::move(allocator); }

  Allocator& operator=(const Allocator&) = delete;
  Allocator& operator=(Allocator&& allocator) noexcept;

  ~Allocator() noexcept { vmaDestroyAllocator(allocator_); }

  [[nodiscard]] VmaAllocator operator*() const noexcept { return allocator_; }

private:
  VmaAllocator allocator_ = nullptr;
};

}  // namespace gfx

#endif  // SRC_GRAPHICS_INCLUDE_GRAPHICS_ALLOCATOR_H_
