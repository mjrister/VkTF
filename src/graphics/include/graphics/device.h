#ifndef SRC_GRAPHICS_INCLUDE_GRAPHICS_DEVICE_H_
#define SRC_GRAPHICS_INCLUDE_GRAPHICS_DEVICE_H_

#include <cstdint>

#include <vulkan/vulkan.hpp>

namespace gfx {

struct QueueFamilyIndices {
  std::uint32_t graphics_index{};
  std::uint32_t present_index{};
  std::uint32_t transfer_index{};
};

class Device {
public:
  Device(vk::Instance instance, vk::SurfaceKHR surface);

  [[nodiscard]] vk::Device operator*() const noexcept { return *device_; }
  [[nodiscard]] const vk::Device* operator->() const noexcept { return &(*device_); }

  [[nodiscard]] const vk::PhysicalDevice& physical_device() const noexcept { return physical_device_; }
  [[nodiscard]] const QueueFamilyIndices& queue_family_indices() const noexcept { return queue_family_indices_; }
  [[nodiscard]] vk::Queue graphics_queue() const noexcept { return graphics_queue_; }
  [[nodiscard]] vk::Queue present_queue() const noexcept { return present_queue_; }
  [[nodiscard]] vk::Queue transfer_queue() const noexcept { return transfer_queue_; }

private:
  vk::PhysicalDevice physical_device_;
  QueueFamilyIndices queue_family_indices_;
  vk::UniqueDevice device_;
  vk::Queue graphics_queue_, present_queue_, transfer_queue_;
};

}  // namespace gfx

#endif  // SRC_GRAPHICS_INCLUDE_GRAPHICS_DEVICE_H_
