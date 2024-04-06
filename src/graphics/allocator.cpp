#include "graphics/allocator.h"

#include "graphics/instance.h"

namespace {

VmaVulkanFunctions GetVulkanFunctions() {
  return VmaVulkanFunctions {
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

VmaAllocator CreateAllocator(const vk::Instance instance,
                             const vk::PhysicalDevice physical_device,
                             const vk::Device device) {
  const auto vulkan_functions = GetVulkanFunctions();
  const VmaAllocatorCreateInfo allocator_create_info{.flags = VMA_ALLOCATOR_CREATE_EXT_MEMORY_PRIORITY_BIT,
                                                     .physicalDevice = physical_device,
                                                     .device = device,
                                                     .pVulkanFunctions = &vulkan_functions,
                                                     .instance = instance,
                                                     .vulkanApiVersion = gfx::Instance::kApiVersion};

  VmaAllocator allocator{};
  const auto result = vmaCreateAllocator(&allocator_create_info, &allocator);
  vk::resultCheck(static_cast<vk::Result>(result), "Allocator creation failed");

  return allocator;
}

}  // namespace

namespace gfx {

Allocator::Allocator(const vk::Instance instance, const vk::PhysicalDevice physical_device, const vk::Device device)
    : allocator_{CreateAllocator(instance, physical_device, device)} {}

Allocator& Allocator::operator=(Allocator&& allocator) noexcept {
  if (this != &allocator) {
    allocator_ = std::exchange(allocator.allocator_, {});
  }
  return *this;
}

}  // namespace gfx
