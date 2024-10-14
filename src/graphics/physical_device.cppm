module;

#include <cstdint>
#include <optional>
#include <ranges>
#include <stdexcept>

#include <vulkan/vulkan.hpp>

export module physical_device;

namespace gfx {

export struct QueueFamilyIndices {
  std::uint32_t graphics_index = 0;
  std::uint32_t present_index = 0;
};

struct RankedPhysicalDevice {
  vk::PhysicalDevice physical_device;
  vk::PhysicalDeviceLimits physical_device_limits;
  QueueFamilyIndices queue_family_indices;
  int rank = 0;
};

export class PhysicalDevice {
public:
  PhysicalDevice(vk::Instance instance, vk::SurfaceKHR surface);

  [[nodiscard]] vk::PhysicalDevice operator*() const noexcept { return physical_device_; }
  [[nodiscard]] const vk::PhysicalDevice* operator->() const noexcept { return &physical_device_; }

  [[nodiscard]] const vk::PhysicalDeviceLimits& limits() const noexcept { return physical_device_limits_; }
  [[nodiscard]] const vk::PhysicalDeviceFeatures& features() const noexcept { return physical_device_features_; }
  [[nodiscard]] const QueueFamilyIndices& queue_family_indices() const noexcept { return queue_family_indices_; }

private:
  explicit PhysicalDevice(const RankedPhysicalDevice& ranked_physical_device);

  vk::PhysicalDevice physical_device_;
  vk::PhysicalDeviceLimits physical_device_limits_;
  vk::PhysicalDeviceFeatures physical_device_features_;
  QueueFamilyIndices queue_family_indices_;
};

}  // namespace gfx

module :private;

namespace {

constexpr auto kInvalidRank = -1;

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

// TODO(matthew-rister): implement a more robust ranking system based on device features, limits, and format support
gfx::RankedPhysicalDevice GetRankedPhysicalDevice(const vk::PhysicalDevice physical_device,
                                                  const vk::SurfaceKHR surface) {
  return FindQueueFamilyIndices(physical_device, surface)
      .transform([physical_device](const auto queue_family_indices) {
        const auto physical_device_properties = physical_device.getProperties();
        return gfx::RankedPhysicalDevice{
            .physical_device = physical_device,
            .physical_device_limits = physical_device_properties.limits,
            .queue_family_indices = queue_family_indices,
            .rank = physical_device_properties.deviceType == vk::PhysicalDeviceType::eDiscreteGpu};
      })
      .value_or(gfx::RankedPhysicalDevice{.rank = kInvalidRank});
}

gfx::RankedPhysicalDevice SelectPhysicalDevice(const vk::Instance instance, const vk::SurfaceKHR surface) {
  const auto ranked_physical_devices = instance.enumeratePhysicalDevices()
                                       | std::views::transform([surface](const auto physical_device) {
                                           return GetRankedPhysicalDevice(physical_device, surface);
                                         })
                                       | std::views::filter([](const auto& ranked_physical_device) {
                                           return ranked_physical_device.rank != kInvalidRank;
                                         })
                                       | std::ranges::to<std::vector>();

  return ranked_physical_devices.empty()
             ? throw std::runtime_error{"No supported physical device could be found"}
             : *std::ranges::max_element(ranked_physical_devices, {}, &gfx::RankedPhysicalDevice::rank);
}

}  // namespace

namespace gfx {

PhysicalDevice::PhysicalDevice(const vk::Instance instance, const vk::SurfaceKHR surface)
    : PhysicalDevice{SelectPhysicalDevice(instance, surface)} {}

PhysicalDevice::PhysicalDevice(const RankedPhysicalDevice& ranked_physical_device)
    : physical_device_{ranked_physical_device.physical_device},
      physical_device_limits_{ranked_physical_device.physical_device_limits},
      physical_device_features_{physical_device_.getFeatures()},
      queue_family_indices_{ranked_physical_device.queue_family_indices} {}

}  // namespace gfx
