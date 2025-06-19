module;

#include <cstdint>
#include <ranges>
#include <unordered_set>
#include <vector>

#include <vulkan/vulkan.hpp>

export module device;

import queue;

namespace vktf {

export class [[nodiscard]] Device {
public:
  struct [[nodiscard]] CreateInfo {
    const QueueFamilies& queue_families;
    const std::vector<const char*>& enabled_extensions;
    const vk::PhysicalDeviceFeatures& enabled_features;
  };

  Device(vk::PhysicalDevice physical_device, const CreateInfo& create_info);

  [[nodiscard]] vk::Device operator*() const noexcept { return *device_; }
  [[nodiscard]] const vk::Device* operator->() const noexcept { return device_.operator->(); }

private:
  vk::UniqueDevice device_;
};

}  // namespace vktf

module :private;

namespace vktf {

namespace {

std::vector<vk::DeviceQueueCreateInfo> GetDeviceQueueCreateInfos(const QueueFamilies& queue_families) {
  const auto& [graphics_queue_family, present_queue_family] = queue_families;

  return std::unordered_set{graphics_queue_family.index, present_queue_family.index}
         | std::views::transform([](const auto queue_family_index) {
             static constexpr auto kDefaultQueuePriority = 0.5f;
             return vk::DeviceQueueCreateInfo{
                 .queueFamilyIndex = queue_family_index,
                 // TODO: add support for multiple queues to enable multithreaded command buffer submission
                 .queueCount = 1,
                 .pQueuePriorities = &kDefaultQueuePriority};
           })
         | std::ranges::to<std::vector>();
}

vk::UniqueDevice CreateDevice(const vk::PhysicalDevice& physical_device, const Device::CreateInfo& create_info) {
  const auto& [queue_families, enabled_extensions, enabled_features] = create_info;
  const auto device_queue_create_infos = GetDeviceQueueCreateInfos(queue_families);

  auto device = physical_device.createDeviceUnique(
      vk::DeviceCreateInfo{.queueCreateInfoCount = static_cast<std::uint32_t>(device_queue_create_infos.size()),
                           .pQueueCreateInfos = device_queue_create_infos.data(),
                           .enabledExtensionCount = static_cast<std::uint32_t>(enabled_extensions.size()),
                           .ppEnabledExtensionNames = enabled_extensions.data(),
                           .pEnabledFeatures = &enabled_features});

#if VULKAN_HPP_DISPATCH_LOADER_DYNAMIC == 1
  VULKAN_HPP_DEFAULT_DISPATCHER.init(*device);
#endif

  return device;
}

}  // namespace

Device::Device(const vk::PhysicalDevice physical_device, const CreateInfo& create_info)
    : device_{CreateDevice(physical_device, create_info)} {}

}  // namespace vktf
