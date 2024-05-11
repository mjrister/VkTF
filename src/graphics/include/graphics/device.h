#ifndef SRC_GRAPHICS_INCLUDE_GRAPHICS_DEVICE_H_
#define SRC_GRAPHICS_INCLUDE_GRAPHICS_DEVICE_H_

#include <cstdint>
#include <limits>

#include <vulkan/vulkan.hpp>

#include "graphics/physical_device.h"

namespace gfx {

struct QueueFamilyIndices {
  static constexpr auto kInvalidIndex = std::numeric_limits<std::uint32_t>::max();
  std::uint32_t graphics_index = kInvalidIndex;
  std::uint32_t present_index = kInvalidIndex;
  std::uint32_t transfer_index = kInvalidIndex;
};

class Device {
public:
  Device(const vk::Instance instance, const vk::SurfaceKHR surface);

  [[nodiscard]] vk::Device operator*() const noexcept { return *device_; }
  [[nodiscard]] const vk::Device* operator->() const noexcept { return &(*device_); }

  [[nodiscard]] const PhysicalDevice& physical_device() const noexcept { return physical_device_; }
  [[nodiscard]] const QueueFamilyIndices& queue_family_indices() const noexcept { return queue_family_indices_; }
  [[nodiscard]] vk::Queue graphics_queue() const noexcept { return graphics_queue_; }
  [[nodiscard]] vk::Queue present_queue() const noexcept { return present_queue_; }
  [[nodiscard]] vk::Queue transfer_queue() const noexcept { return transfer_queue_; }

private:
  PhysicalDevice physical_device_;
  QueueFamilyIndices queue_family_indices_;
  vk::UniqueDevice device_;
  vk::Queue graphics_queue_, present_queue_, transfer_queue_;
};

}  // namespace gfx

#endif  // SRC_GRAPHICS_INCLUDE_GRAPHICS_DEVICE_H_
