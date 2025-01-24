module;

#include <utility>

#include <vk_mem_alloc.h>
#include <vulkan/vulkan.hpp>

export module allocator;

import instance;

namespace vktf {

export class Allocator {
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

export constexpr VmaAllocationCreateInfo kDefaultAllocationCreateInfo{.usage = VMA_MEMORY_USAGE_AUTO};

}  // namespace vktf

module :private;

namespace {

VmaVulkanFunctions GetVulkanFunctions() {
  return VmaVulkanFunctions{
#if VULKAN_HPP_DISPATCH_LOADER_DYNAMIC == 1
      .vkGetInstanceProcAddr = VULKAN_HPP_DEFAULT_DISPATCHER.vkGetInstanceProcAddr,
      .vkGetDeviceProcAddr = VULKAN_HPP_DEFAULT_DISPATCHER.vkGetDeviceProcAddr,
      .vkGetPhysicalDeviceProperties = VULKAN_HPP_DEFAULT_DISPATCHER.vkGetPhysicalDeviceProperties,
      .vkGetPhysicalDeviceMemoryProperties = VULKAN_HPP_DEFAULT_DISPATCHER.vkGetPhysicalDeviceMemoryProperties,
      .vkAllocateMemory = VULKAN_HPP_DEFAULT_DISPATCHER.vkAllocateMemory,
      .vkFreeMemory = VULKAN_HPP_DEFAULT_DISPATCHER.vkFreeMemory,
      .vkMapMemory = VULKAN_HPP_DEFAULT_DISPATCHER.vkMapMemory,
      .vkUnmapMemory = VULKAN_HPP_DEFAULT_DISPATCHER.vkUnmapMemory,
      .vkFlushMappedMemoryRanges = VULKAN_HPP_DEFAULT_DISPATCHER.vkFlushMappedMemoryRanges,
      .vkInvalidateMappedMemoryRanges = VULKAN_HPP_DEFAULT_DISPATCHER.vkInvalidateMappedMemoryRanges,
      .vkBindBufferMemory = VULKAN_HPP_DEFAULT_DISPATCHER.vkBindBufferMemory,
      .vkBindImageMemory = VULKAN_HPP_DEFAULT_DISPATCHER.vkBindImageMemory,
      .vkGetBufferMemoryRequirements = VULKAN_HPP_DEFAULT_DISPATCHER.vkGetBufferMemoryRequirements,
      .vkGetImageMemoryRequirements = VULKAN_HPP_DEFAULT_DISPATCHER.vkGetImageMemoryRequirements,
      .vkCreateBuffer = VULKAN_HPP_DEFAULT_DISPATCHER.vkCreateBuffer,
      .vkDestroyBuffer = VULKAN_HPP_DEFAULT_DISPATCHER.vkDestroyBuffer,
      .vkCreateImage = VULKAN_HPP_DEFAULT_DISPATCHER.vkCreateImage,
      .vkDestroyImage = VULKAN_HPP_DEFAULT_DISPATCHER.vkDestroyImage,
      .vkCmdCopyBuffer = VULKAN_HPP_DEFAULT_DISPATCHER.vkCmdCopyBuffer,
      .vkGetBufferMemoryRequirements2KHR = VULKAN_HPP_DEFAULT_DISPATCHER.vkGetBufferMemoryRequirements2,
      .vkGetImageMemoryRequirements2KHR = VULKAN_HPP_DEFAULT_DISPATCHER.vkGetImageMemoryRequirements2,
      .vkBindBufferMemory2KHR = VULKAN_HPP_DEFAULT_DISPATCHER.vkBindBufferMemory2,
      .vkBindImageMemory2KHR = VULKAN_HPP_DEFAULT_DISPATCHER.vkBindImageMemory2,
      .vkGetPhysicalDeviceMemoryProperties2KHR = VULKAN_HPP_DEFAULT_DISPATCHER.vkGetPhysicalDeviceMemoryProperties2,
      .vkGetDeviceBufferMemoryRequirements = VULKAN_HPP_DEFAULT_DISPATCHER.vkGetDeviceBufferMemoryRequirements,
      .vkGetDeviceImageMemoryRequirements = VULKAN_HPP_DEFAULT_DISPATCHER.vkGetDeviceImageMemoryRequirements
#endif
  };
}

}  // namespace

namespace vktf {

Allocator::Allocator(const vk::Instance instance, const vk::PhysicalDevice physical_device, const vk::Device device) {
  const auto vulkan_functions = GetVulkanFunctions();
  const VmaAllocatorCreateInfo allocator_create_info{.physicalDevice = physical_device,
                                                     .device = device,
                                                     .pVulkanFunctions = &vulkan_functions,
                                                     .instance = instance,
                                                     .vulkanApiVersion = Instance::kApiVersion};
  const auto result = vmaCreateAllocator(&allocator_create_info, &allocator_);
  vk::detail::resultCheck(static_cast<vk::Result>(result), "Allocator creation failed");
}

Allocator& Allocator::operator=(Allocator&& allocator) noexcept {
  if (this != &allocator) {
    allocator_ = std::exchange(allocator.allocator_, nullptr);
  }
  return *this;
}

}  // namespace vktf
