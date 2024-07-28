#include "graphics/physical_device.h"

#include <algorithm>
#include <optional>
#include <ranges>
#include <stdexcept>

namespace {

std::optional<gfx::QueueFamilyIndices> FindQueueFamilyIndices(const vk::PhysicalDevice physical_device,
                                                              const vk::SurfaceKHR surface) {
  std::optional<std::uint32_t> maybe_graphics_index;
  std::optional<std::uint32_t> maybe_present_index;

  for (std::uint32_t index = 0; const auto& queue_family_properties : physical_device.getQueueFamilyProperties()) {
    if (!maybe_graphics_index.has_value() && queue_family_properties.queueFlags & vk::QueueFlagBits::eGraphics) {
      maybe_graphics_index = index;
    }
    if (!maybe_present_index.has_value() && physical_device.getSurfaceSupportKHR(index, surface) == vk::True) {
      maybe_present_index = index;
    }
    if (maybe_graphics_index.has_value() && maybe_present_index.has_value()) {
      return gfx::QueueFamilyIndices{.graphics_index = *maybe_graphics_index, .present_index = *maybe_present_index};
    }
    ++index;
  }

  return std::nullopt;
}

}  // namespace

namespace gfx {

struct PhysicalDevice::RankedPhysicalDevice {
  static constexpr auto kInvalidRank = -1;
  vk::PhysicalDevice physical_device;
  vk::PhysicalDeviceLimits physical_device_limits;
  QueueFamilyIndices queue_family_indices;
  int rank = kInvalidRank;
};

PhysicalDevice::PhysicalDevice(const vk::Instance instance, const vk::SurfaceKHR surface)
    : PhysicalDevice{SelectPhysicalDevice(instance, surface)} {}

PhysicalDevice::PhysicalDevice(const RankedPhysicalDevice& ranked_physical_device)
    : physical_device_{ranked_physical_device.physical_device},
      physical_device_limits_{ranked_physical_device.physical_device_limits},
      physical_device_features_{physical_device_.getFeatures()},
      queue_family_indices_{ranked_physical_device.queue_family_indices} {}

PhysicalDevice::RankedPhysicalDevice PhysicalDevice::SelectPhysicalDevice(const vk::Instance instance,
                                                                          const vk::SurfaceKHR surface) {
  const auto ranked_physical_devices = instance.enumeratePhysicalDevices()
                                       | std::views::transform([surface](const auto physical_device) {
                                           return GetRankedPhysicalDevice(physical_device, surface);
                                         })
                                       | std::views::filter([](const auto& ranked_physical_device) {
                                           return ranked_physical_device.rank != RankedPhysicalDevice::kInvalidRank;
                                         })
                                       | std::ranges::to<std::vector>();

  return ranked_physical_devices.empty()
             ? throw std::runtime_error{"No supported physical device could be found"}
             : *std::ranges::max_element(ranked_physical_devices, {}, &RankedPhysicalDevice::rank);
}

// TODO(matthew-rister): implement a better ranking system based on required device features, limits, and format support
PhysicalDevice::RankedPhysicalDevice PhysicalDevice::GetRankedPhysicalDevice(const vk::PhysicalDevice physical_device,
                                                                             const vk::SurfaceKHR surface) {
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

}  // namespace gfx
