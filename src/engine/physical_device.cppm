module;

#include <algorithm>
#include <cstdint>
#include <limits>
#include <optional>
#include <ranges>
#include <stdexcept>
#include <unordered_set>
#include <vector>

#include <vulkan/vulkan.hpp>

export module physical_device;

import queue;

namespace vktf {
struct RankedPhysicalDevice;

/**
 * @brief An abstraction for a Vulkan physical device.
 * @details This class handles enumerating over all Vulkan physical devices and selecting the one that provides the best
 *          performance characteristics while supporting the minimum application requirements.
 * @see https://registry.khronos.org/vulkan/specs/latest/man/html/VkPhysicalDevice.html VkPhysicalDevice
 */
export class [[nodiscard]] PhysicalDevice {
public:
  /** @brief The parameters for creating a @ref PhysicalDevice. */
  struct [[nodiscard]] CreateInfo {
    /** @brief The surface to present images to. */
    vk::SurfaceKHR surface;

    /** @brief The device extensions required by the application. */
    const std::vector<const char*>& required_extensions;
  };

  /**
   * @brief Creates a @ref PhysicalDevice.
   * @param instance The instance for enumerating available physical devices.
   * @param create_info @copybrief PhysicalDevice::CreateInfo
   * @throws std::runtime_error Thrown if no supported physical device is found.
   */
  PhysicalDevice(vk::Instance instance, const CreateInfo& create_info);

  /** @brief Gets the underlying Vulkan physical device handle. */
  [[nodiscard]] vk::PhysicalDevice operator*() const noexcept { return physical_device_; }

  /** @brief Gets the available features for this physical device. */
  [[nodiscard]] const vk::PhysicalDeviceFeatures& features() const noexcept { return physical_device_features_; }

  /** @brief Gets the hardware limits for this physical device. */
  [[nodiscard]] const vk::PhysicalDeviceLimits& limits() const noexcept { return physical_device_limits_; }

  /** @brief Gets the selected queue families for this physical device. */
  [[nodiscard]] const QueueFamilies& queue_families() const noexcept { return queue_families_; }

private:
  explicit PhysicalDevice(const RankedPhysicalDevice& ranked_physical_device);

  vk::PhysicalDevice physical_device_;
  vk::PhysicalDeviceFeatures physical_device_features_;
  vk::PhysicalDeviceLimits physical_device_limits_;
  QueueFamilies queue_families_;
};

}  // namespace vktf

module :private;

namespace vktf {

struct RankedPhysicalDevice {
  static constexpr auto kInvalidRank = std::numeric_limits<std::int32_t>::min();
  vk::PhysicalDevice physical_device;
  vk::PhysicalDeviceLimits physical_device_limits;
  QueueFamilies queue_families;
  std::int32_t rank = kInvalidRank;
};

namespace {

bool HasRequiredExtensions(const vk::PhysicalDevice physical_device,
                           const std::vector<const char*>& required_extensions) {
  const auto device_extension_properties = physical_device.enumerateDeviceExtensionProperties();
  const auto device_extensions = device_extension_properties
                                 | std::views::transform([](const auto& extension_properties) -> std::string_view {
                                     return extension_properties.extensionName;
                                   })
                                 | std::ranges::to<std::unordered_set>();

  return std::ranges::all_of(required_extensions, [&device_extensions](const std::string_view required_extension) {
    return device_extensions.contains(required_extension);
  });
}

std::optional<QueueFamilies> FindQueueFamilies(const vk::PhysicalDevice physical_device, const vk::SurfaceKHR surface) {
  std::optional<QueueFamily> graphics_queue_family;
  std::optional<QueueFamily> present_queue_family;

  for (std::uint32_t index = 0; const auto& queue_family_properties : physical_device.getQueueFamilyProperties()) {
    assert(queue_family_properties.queueCount > 0);  // required by the Vulkan specification
    const QueueFamily queue_family{.index = index++, .queue_count = queue_family_properties.queueCount};
    const auto has_graphics_support = queue_family_properties.queueFlags & vk::QueueFlagBits::eGraphics;
    const auto has_present_support = physical_device.getSurfaceSupportKHR(queue_family.index, surface) == vk::True;

    if (has_graphics_support && has_present_support) {  // prefer combined graphics and present queue family
      return QueueFamilies{.graphics_family = queue_family, .present_family = queue_family};
    }
    if (!graphics_queue_family.has_value() && has_graphics_support) {
      graphics_queue_family = queue_family;
    }
    if (!present_queue_family.has_value() && has_present_support) {
      present_queue_family = queue_family;
    }
  }

  if (!graphics_queue_family.has_value() || !present_queue_family.has_value()) {
    return std::nullopt;
  }

  return QueueFamilies{.graphics_family = *graphics_queue_family, .present_family = *present_queue_family};
}

RankedPhysicalDevice GetRankedPhysicalDevice(const vk::PhysicalDevice physical_device,
                                             const vk::SurfaceKHR surface,
                                             const std::vector<const char*>& required_extensions) {
  static const RankedPhysicalDevice kInvalidPhysicalDevice{.rank = RankedPhysicalDevice::kInvalidRank};
  if (!HasRequiredExtensions(physical_device, required_extensions)) return kInvalidPhysicalDevice;

  const auto& queue_families = FindQueueFamilies(physical_device, surface);
  if (!queue_families.has_value()) return kInvalidPhysicalDevice;

  const auto physical_device_properties = physical_device.getProperties();
  return RankedPhysicalDevice{
      .physical_device = physical_device,
      .physical_device_limits = physical_device_properties.limits,
      .queue_families = *queue_families,
      // TODO: consider a more robust ranking system based on device features, limits, and format support
      .rank = physical_device_properties.deviceType == vk::PhysicalDeviceType::eDiscreteGpu};
}

RankedPhysicalDevice SelectPhysicalDevice(const vk::Instance instance,
                                          const vk::SurfaceKHR surface,
                                          const std::vector<const char*>& required_extensions) {
  const auto ranked_physical_devices =
      instance.enumeratePhysicalDevices()
      | std::views::transform([surface, required_extensions](const auto physical_device) {
          return GetRankedPhysicalDevice(physical_device, surface, required_extensions);
        })
      | std::views::filter([](const auto& ranked_physical_device) {
          return ranked_physical_device.rank != RankedPhysicalDevice::kInvalidRank;
        })
      | std::ranges::to<std::vector>();

  return ranked_physical_devices.empty()
             ? throw std::runtime_error{"No supported physical device could be found"}
             : *std::ranges::max_element(ranked_physical_devices, {}, &RankedPhysicalDevice::rank);
}

}  // namespace

PhysicalDevice::PhysicalDevice(const vk::Instance instance, const CreateInfo& create_info)
    : PhysicalDevice{SelectPhysicalDevice(instance, create_info.surface, create_info.required_extensions)} {}

PhysicalDevice::PhysicalDevice(const RankedPhysicalDevice& ranked_physical_device)
    : physical_device_{ranked_physical_device.physical_device},
      physical_device_features_{physical_device_.getFeatures()},
      physical_device_limits_{ranked_physical_device.physical_device_limits},
      queue_families_{ranked_physical_device.queue_families} {}

}  // namespace vktf
