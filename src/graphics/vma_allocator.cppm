module;

#include <cstdint>
#include <memory>

#include <vk_mem_alloc.h>
#include <vulkan/vulkan.hpp>

export module vma_allocator;

namespace vktf::vma {

using UniqueAllocator = std::unique_ptr<VmaAllocator_T, decltype(&vmaDestroyAllocator)>;

export class [[nodiscard]] Allocator {
public:
  struct [[nodiscard]] CreateInfo {
    vk::Instance instance;
    vk::PhysicalDevice physical_device;
    std::uint32_t vulkan_api_version = 0;
  };

  Allocator(vk::Device device, const CreateInfo& create_info);

  [[nodiscard]] VmaAllocator operator*() const noexcept { return allocator_.get(); }

  [[nodiscard]] vk::Device device() const noexcept { return device_; }

private:
  vk::Device device_;
  UniqueAllocator allocator_;
};

export constexpr VmaAllocationCreateInfo kDedicatedMemoryAllocationCreateInfo{
    .flags = VMA_ALLOCATION_CREATE_DEDICATED_MEMORY_BIT,
    .usage = VMA_MEMORY_USAGE_AUTO,
    .priority = 1.0f};

export constexpr VmaAllocationCreateInfo kHostVisibleAllocationCreateInfo{
    .flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT,
    .usage = VMA_MEMORY_USAGE_AUTO_PREFER_HOST};

export constexpr VmaAllocationCreateInfo kDeviceLocalAllocationCreateInfo{.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE};

}  // namespace vktf::vma

module :private;

namespace vktf::vma {

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

UniqueAllocator CreateAllocator(const vk::Device device, const Allocator::CreateInfo& create_info) {
  const auto& [instance, physical_device, vulkan_api_version] = create_info;
  const auto vulkan_functions = GetVulkanFunctions();
  const VmaAllocatorCreateInfo allocator_create_info{.physicalDevice = physical_device,
                                                     .device = device,
                                                     .pVulkanFunctions = &vulkan_functions,
                                                     .instance = instance,
                                                     .vulkanApiVersion = vulkan_api_version};

  VmaAllocator allocator = nullptr;
  const auto result = vmaCreateAllocator(&allocator_create_info, &allocator);
  vk::detail::resultCheck(static_cast<vk::Result>(result), "Allocator creation failed");
  return UniqueAllocator{allocator, vmaDestroyAllocator};
}

}  // namespace

Allocator::Allocator(const vk::Device device, const CreateInfo& create_info)
    : device_{device}, allocator_{CreateAllocator(device_, create_info)} {}

}  // namespace vktf::vma
