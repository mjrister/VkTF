#include "graphics/physical_device.h"

#include <algorithm>
#include <cassert>
#include <optional>
#include <ranges>
#include <stdexcept>
#include <vector>

namespace {

using QueueFamilyIndices = gfx::PhysicalDevice::QueueFamilyIndices;

struct RankedPhysicalDevice {
  static constexpr auto kInvalidRank = -1;
  vk::PhysicalDevice physical_device;
  vk::PhysicalDeviceLimits physical_device_limits;
  QueueFamilyIndices queue_family_indices;
  int rank = kInvalidRank;
};

std::optional<std::uint32_t> FindQueueFamily(const std::vector<vk::QueueFamilyProperties>& queue_family_properties,
                                             std::predicate<const vk::QueueFamilyProperties&> auto&& find_fn) {
  if (const auto iterator = std::ranges::find_if(queue_family_properties, find_fn);
      iterator != queue_family_properties.cend()) {
    assert(iterator->queueCount > 0);
    const auto index = std::ranges::distance(queue_family_properties.cbegin(), iterator);
    return static_cast<std::uint32_t>(index);
  }
  return std::nullopt;
}

std::optional<QueueFamilyIndices> FindQueueFamilyIndices(const vk::PhysicalDevice physical_device,
                                                         const vk::SurfaceKHR surface) {
  const auto all_queue_family_properties = physical_device.getQueueFamilyProperties();

  const auto graphics_index = FindQueueFamily(all_queue_family_properties, [](const auto& queue_family_properties) {
    return static_cast<bool>(queue_family_properties.queueFlags & vk::QueueFlagBits::eGraphics);
  });

  const auto present_index =
      FindQueueFamily(all_queue_family_properties, [physical_device, surface, index = 0u](const auto&) mutable {
        return physical_device.getSurfaceSupportKHR(index++, surface) == vk::True;
      });

  const auto transfer_index = FindQueueFamily(all_queue_family_properties, [](const auto& queue_family_properties) {
    // prefer a dedicated transfer queue family to enable asynchronous transfers to device memory
    return static_cast<bool>(queue_family_properties.queueFlags & vk::QueueFlagBits::eTransfer)
           && !static_cast<bool>(queue_family_properties.queueFlags & vk::QueueFlagBits::eGraphics);
  });

  if (graphics_index.has_value() && present_index.has_value()) {
    return QueueFamilyIndices{.graphics_index = *graphics_index,
                              .present_index = *present_index,
                              // graphics queue family always implicitly accept transfer commands
                              .transfer_index = transfer_index.value_or(*graphics_index)};
  }

  return std::nullopt;
}

RankedPhysicalDevice GetRankedPhysicalDevice(const vk::PhysicalDevice physical_device, const vk::SurfaceKHR surface) {
  return FindQueueFamilyIndices(physical_device, surface)
      .transform([physical_device](const auto queue_family_indices) {
        const auto physical_device_properties = physical_device.getProperties();
        return RankedPhysicalDevice{
            .physical_device = physical_device,
            .physical_device_limits = physical_device_properties.limits,
            .queue_family_indices = queue_family_indices,
            .rank = physical_device_properties.deviceType == vk::PhysicalDeviceType::eDiscreteGpu};
      })
      .value_or(RankedPhysicalDevice{.rank = RankedPhysicalDevice::kInvalidRank});
}

RankedPhysicalDevice SelectPhysicalDevice(const vk::Instance instance, const vk::SurfaceKHR surface) {
  const auto ranked_physical_devices = instance.enumeratePhysicalDevices()
                                       | std::views::transform([surface](const auto physical_device) {
                                           return GetRankedPhysicalDevice(physical_device, surface);
                                         })
                                       | std::views::filter([](const auto& ranked_physical_device) {
                                           return ranked_physical_device.rank != RankedPhysicalDevice::kInvalidRank;
                                         })
                                       | std::ranges::to<std::vector>();

  return ranked_physical_devices.empty()
             ? throw std::runtime_error{"Unsupported physical device"}
             : *std::ranges::max_element(ranked_physical_devices, {}, &RankedPhysicalDevice::rank);
}

}  // namespace

gfx::PhysicalDevice::PhysicalDevice(const vk::Instance instance, const vk::SurfaceKHR surface) {
  const auto [physical_device, limits, queue_family_indices, _] = SelectPhysicalDevice(instance, surface);
  physical_device_ = physical_device;
  limits_ = limits;
  mem_properties_ = physical_device.getMemoryProperties();
  queue_family_indices_ = queue_family_indices;
}
