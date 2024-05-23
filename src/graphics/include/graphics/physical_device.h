#ifndef SRC_GRAPHICS_INCLUDE_GRAPHICS_PHYSICAL_DEVICE_H_
#define SRC_GRAPHICS_INCLUDE_GRAPHICS_PHYSICAL_DEVICE_H_

#include <vulkan/vulkan.hpp>

namespace gfx {

class PhysicalDevice {
public:
  explicit PhysicalDevice(vk::Instance instance);

  [[nodiscard]] vk::PhysicalDevice operator*() const noexcept { return physical_device_; }
  [[nodiscard]] const vk::PhysicalDevice* operator->() const noexcept { return &physical_device_; }

  [[nodiscard]] const vk::PhysicalDeviceLimits& limits() const noexcept { return physical_device_limits_; }
  [[nodiscard]] const vk::PhysicalDeviceFeatures& features() const noexcept { return physical_device_features_; }

private:
  vk::PhysicalDevice physical_device_;
  vk::PhysicalDeviceLimits physical_device_limits_;
  vk::PhysicalDeviceFeatures physical_device_features_;
};

}  // namespace gfx

#endif  // SRC_GRAPHICS_INCLUDE_GRAPHICS_PHYSICAL_DEVICE_H_
