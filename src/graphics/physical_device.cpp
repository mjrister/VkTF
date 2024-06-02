#include "graphics/physical_device.h"

#include <algorithm>
#include <ranges>
#include <stdexcept>

struct RankedPhysicalDevice {
  static constexpr auto kInvalidRank = -1;
  vk::PhysicalDevice physical_device;
  vk::PhysicalDeviceLimits physical_device_limits;
  int rank = kInvalidRank;
};

// TODO(matthew-rister): implement a better ranking system based on physical device features, limits, and format support
RankedPhysicalDevice GetMaxRankPhysicalDevice(const vk::Instance instance) {
  const auto ranked_physical_devices =
      instance.enumeratePhysicalDevices() | std::views::transform([](const auto physical_device) {
        const auto physical_device_properties = physical_device.getProperties();
        return RankedPhysicalDevice{
            .physical_device = physical_device,
            .physical_device_limits = physical_device_properties.limits,
            .rank = physical_device_properties.deviceType == vk::PhysicalDeviceType::eDiscreteGpu};
      });

  return std::ranges::empty(ranked_physical_devices)
             ? throw std::runtime_error{"No physical device could be found"}
             : *std::ranges::max_element(ranked_physical_devices, {}, &RankedPhysicalDevice::rank);
}

gfx::PhysicalDevice::PhysicalDevice(const vk::Instance instance) {
  const auto& [physical_device, physical_device_limits, _] = GetMaxRankPhysicalDevice(instance);
  // NOLINTBEGIN(cppcoreguidelines-prefer-member-initializer)
  physical_device_ = physical_device;
  physical_device_limits_ = physical_device_limits;
  physical_device_features_ = physical_device.getFeatures();
  // NOLINTEND(cppcoreguidelines-prefer-member-initializer)
}
