#ifndef GRAPHICS_DEVICE_H_
#define GRAPHICS_DEVICE_H_

#include <vulkan/vulkan.hpp>

namespace gfx {
class PhysicalDevice;

class Device {
public:
  explicit Device(const PhysicalDevice& physical_device);

  [[nodiscard]] vk::Device operator*() const noexcept { return *device_; }
  [[nodiscard]] const vk::Device* operator->() const noexcept { return &(*device_); }

private:
  vk::UniqueDevice device_;
};

}  // namespace gfx

#endif  // GRAPHICS_DEVICE_H_
