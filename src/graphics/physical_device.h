#ifndef GRAPHICS_PHYSICAL_DEVICE_H_
#define GRAPHICS_PHYSICAL_DEVICE_H_

#include <cstdint>
#include <limits>

#include <vulkan/vulkan.hpp>

namespace gfx {

struct QueueFamilyIndices {
  static constexpr auto kInvalidIndex = std::numeric_limits<std::uint32_t>::max();
  std::uint32_t graphics_index = kInvalidIndex;
  std::uint32_t present_index = kInvalidIndex;
};

class PhysicalDevice {
public:
  PhysicalDevice(vk::Instance instance, vk::SurfaceKHR surface);

  [[nodiscard]] vk::PhysicalDevice operator*() const noexcept { return physical_device_; }
  [[nodiscard]] const vk::PhysicalDevice* operator->() const noexcept { return &physical_device_; }

  [[nodiscard]] const vk::PhysicalDeviceLimits& limits() const noexcept { return physical_device_limits_; }
  [[nodiscard]] const vk::PhysicalDeviceFeatures& features() const noexcept { return physical_device_features_; }
  [[nodiscard]] const QueueFamilyIndices& queue_family_indices() const noexcept { return queue_family_indices_; }

private:
  struct RankedPhysicalDevice;

  explicit PhysicalDevice(const RankedPhysicalDevice& ranked_physical_device);

  static RankedPhysicalDevice SelectPhysicalDevice(vk::Instance instance, vk::SurfaceKHR surface);
  static RankedPhysicalDevice GetRankedPhysicalDevice(vk::PhysicalDevice physical_device, vk::SurfaceKHR surface);

  vk::PhysicalDevice physical_device_;
  vk::PhysicalDeviceLimits physical_device_limits_;
  vk::PhysicalDeviceFeatures physical_device_features_;
  QueueFamilyIndices queue_family_indices_;
};

}  // namespace gfx

#endif  // GRAPHICS_PHYSICAL_DEVICE_H_
