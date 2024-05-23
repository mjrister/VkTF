#include "graphics/device.h"

#include <algorithm>
#include <array>
#include <concepts>
#include <optional>
#include <ranges>
#include <stdexcept>
#include <unordered_set>
#include <vector>

namespace {

std::optional<std::uint32_t> FindQueueFamilyIndex(
    const std::vector<vk::QueueFamilyProperties>& all_queue_family_properties,
    std::predicate<const vk::QueueFamilyProperties&> auto&& find_fn) {
  if (const auto iterator = std::ranges::find_if(all_queue_family_properties, find_fn);
      iterator != all_queue_family_properties.cend()) {
    const auto index = std::ranges::distance(all_queue_family_properties.cbegin(), iterator);
    return static_cast<std::uint32_t>(index);
  }
  return std::nullopt;
}

gfx::QueueFamilyIndices FindQueueFamilyIndices(const vk::PhysicalDevice physical_device, const vk::SurfaceKHR surface) {
  const auto all_queue_family_properties = physical_device.getQueueFamilyProperties();

  const auto maybe_graphics_index =
      FindQueueFamilyIndex(all_queue_family_properties, [](const auto& queue_family_properties) {
        return static_cast<bool>(queue_family_properties.queueFlags & vk::QueueFlagBits::eGraphics);
      });

  const auto maybe_present_index =
      FindQueueFamilyIndex(all_queue_family_properties,
                           [physical_device, surface, index = 0u](const auto& /*queue_family_properties*/) mutable {
                             return physical_device.getSurfaceSupportKHR(index++, surface) == vk::True;
                           });

  const auto maybe_transfer_index =
      FindQueueFamilyIndex(all_queue_family_properties, [](const auto& queue_family_properties) {
        // prefer a dedicated transfer queue family to enable asynchronous transfers to device memory
        return static_cast<bool>(queue_family_properties.queueFlags & vk::QueueFlagBits::eTransfer)
               && !static_cast<bool>(queue_family_properties.queueFlags & vk::QueueFlagBits::eGraphics);
      });

  if (maybe_graphics_index.has_value() && maybe_present_index.has_value()) {
    return gfx::QueueFamilyIndices{.graphics_index = *maybe_graphics_index,
                                   .present_index = *maybe_present_index,
                                   // graphics queue family always implicitly accept transfer commands
                                   .transfer_index = maybe_transfer_index.value_or(*maybe_graphics_index)};
  }

  throw std::runtime_error{"Physical device does not support required queue families"};
}

vk::UniqueDevice CreateDevice(const gfx::PhysicalDevice& physical_device,
                              const gfx::QueueFamilyIndices& queue_family_indices) {
  static constexpr auto kHighestNormalizedQueuePriority = 1.0f;
  const auto [graphics_index, present_index, transfer_index] = queue_family_indices;
  const auto device_queue_create_info =
      std::unordered_set{graphics_index, present_index, transfer_index}
      | std::views::transform([](const auto queue_family_index) {
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

  auto device = physical_device->createDeviceUnique(
      vk::DeviceCreateInfo{.queueCreateInfoCount = static_cast<std::uint32_t>(device_queue_create_info.size()),
                           .pQueueCreateInfos = device_queue_create_info.data(),
                           .enabledExtensionCount = static_cast<std::uint32_t>(kDeviceExtensions.size()),
                           .ppEnabledExtensionNames = kDeviceExtensions.data(),
                           .pEnabledFeatures = &enabled_features});

#if VULKAN_HPP_DISPATCH_LOADER_DYNAMIC == 1
  VULKAN_HPP_DEFAULT_DISPATCHER.init(*device);
#endif

  return device;
}

}  // namespace

namespace gfx {

Device::Device(const vk::Instance instance, const vk::SurfaceKHR surface)
    : physical_device_{instance},
      queue_family_indices_{FindQueueFamilyIndices(*physical_device_, surface)},
      device_{CreateDevice(physical_device_, queue_family_indices_)},
      graphics_queue_{device_->getQueue(queue_family_indices_.graphics_index, 0)},
      present_queue_{device_->getQueue(queue_family_indices_.present_index, 0)},
      transfer_queue_{device_->getQueue(queue_family_indices_.transfer_index, 0)} {}

}  // namespace gfx
