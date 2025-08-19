module;

#include <cstdint>
#include <memory>

#include <vk_mem_alloc.h>
#include <vulkan/vulkan.hpp>

export module vma_allocator;

namespace vktf::vma {

using UniqueAllocator = std::unique_ptr<VmaAllocator_T, decltype(&vmaDestroyAllocator)>;

/**
 * @brief An abstraction for a Vulkan Memory Allocator (VMA) allocator.
 * @details This class handles the initial setup and lifetime management of a VMA allocator.
 * @warning This class retains a non-owning handle to a Vulkan device. The caller is response for ensuring it remains
 *          valid for the entire lifetime of this abstraction.
 * @see https://gpuopen.com/vulkan-memory-allocator/ Vulkan Memory Allocator
 */
export class [[nodiscard]] Allocator {
public:
  /** @brief The parameters for creating an @ref Allocator. */
  struct [[nodiscard]] CreateInfo {
    /** @brief The Vulkan instance. */
    vk::Instance instance;

    /** @brief The Vulkan physical device. */
    vk::PhysicalDevice physical_device;

    /** @brief The Vulkan API version enabled for the application. */
    std::uint32_t vulkan_api_version = 0;
  };

  /**
   * @brief Creates an @ref Allocator.
   * @param device The device for allocating memory.
   * @param create_info @copybrief Allocator::CreateInfo
   */
  Allocator(vk::Device device, const CreateInfo& create_info);

  /** @brief Gets the underlying VMA allocator handle. */
  [[nodiscard]] VmaAllocator operator*() const noexcept { return allocator_.get(); }

  /** @brief Gets the Vulkan device. */
  [[nodiscard]] vk::Device device() const noexcept { return device_; }

private:
  vk::Device device_;
  UniqueAllocator allocator_;
};

/** @brief A VmaAllocationCreateInfo that specifies an allocation should have its own memory block. */
export constexpr VmaAllocationCreateInfo kDedicatedMemoryAllocationCreateInfo{
    .flags = VMA_ALLOCATION_CREATE_DEDICATED_MEMORY_BIT,
    .usage = VMA_MEMORY_USAGE_AUTO,
    .priority = 1.0f};

/** @brief A VmaAllocationCreateInfo that specifies an allocation should prefer sequential write host-visible memory. */
export constexpr VmaAllocationCreateInfo kHostVisibleAllocationCreateInfo{
    .flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT,
    .usage = VMA_MEMORY_USAGE_AUTO_PREFER_HOST};

/** @brief A VmaAllocationCreateInfo that specifies an allocation should prefer device-local memory. */
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
