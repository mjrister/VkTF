#ifndef GRAPHICS_ALLOCATOR_H_
#define GRAPHICS_ALLOCATOR_H_

#include <utility>

#include <vk_mem_alloc.h>
#include <vulkan/vulkan.hpp>

namespace gfx {

class Allocator {
public:
  Allocator(vk::Instance instance, vk::PhysicalDevice physical_device, vk::Device device);

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

#endif  // GRAPHICS_ALLOCATOR_H_
