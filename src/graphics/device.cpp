#include "graphics/device.h"

#include <algorithm>
#include <array>
#include <ranges>
#include <unordered_set>
#include <vector>

#include "graphics/physical_device.h"

namespace gfx {

Device::Device(const PhysicalDevice& physical_device) {
  const auto [graphics_index, present_index] = physical_device.queue_family_indices();
  const auto device_queue_create_info =
      std::unordered_set{graphics_index, present_index}  //
      | std::views::transform([](const auto queue_family_index) {
          static constexpr auto kHighestNormalizedQueuePriority = 1.0f;
          return vk::DeviceQueueCreateInfo{.queueFamilyIndex = queue_family_index,
                                           .queueCount = 1,
                                           .pQueuePriorities = &kHighestNormalizedQueuePriority};
        })
      | std::ranges::to<std::vector>();

  static constexpr std::array kDeviceExtensions{VK_KHR_SWAPCHAIN_EXTENSION_NAME};

  const auto& physical_device_features = physical_device.features();
  const vk::PhysicalDeviceFeatures enabled_features{
      .samplerAnisotropy = physical_device_features.samplerAnisotropy,
      .textureCompressionETC2 = physical_device_features.textureCompressionETC2,
      .textureCompressionASTC_LDR = physical_device_features.textureCompressionASTC_LDR,
      .textureCompressionBC = physical_device_features.textureCompressionBC};

  device_ = physical_device->createDeviceUnique(
      vk::DeviceCreateInfo{.queueCreateInfoCount = static_cast<std::uint32_t>(device_queue_create_info.size()),
                           .pQueueCreateInfos = device_queue_create_info.data(),
                           .enabledExtensionCount = static_cast<std::uint32_t>(kDeviceExtensions.size()),
                           .ppEnabledExtensionNames = kDeviceExtensions.data(),
                           .pEnabledFeatures = &enabled_features});

#if VULKAN_HPP_DISPATCH_LOADER_DYNAMIC == 1
  VULKAN_HPP_DEFAULT_DISPATCHER.init(*device_);
#endif
}

}  // namespace gfx
