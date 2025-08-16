module;

#include <cstdint>
#include <ranges>
#include <unordered_set>
#include <vector>

#include <vulkan/vulkan.hpp>

export module device;

import queue;

namespace vktf {

/** \brief An abstraction for a Vulkan device. */
export class [[nodiscard]] Device {
public:
  /** \brief Parameters used to create a \ref Device. */
  struct [[nodiscard]] CreateInfo {
    /** \brief The queue families the device will submit work to. */
    const QueueFamilies& queue_families;

    /**
     * \brief The device extensions to enable for the application.
     * \warning These extensions are assumed to be validated by \ref PhysicalDevice::CreateInfo::required_extensions.
     */
    const std::vector<const char*>& enabled_extensions;

    /**
     * \brief The device features to enable for the application.
     * \warning These features are assumed to be a valid subset of \ref PhysicalDevice::features.
     */
    const vk::PhysicalDeviceFeatures& enabled_features;
  };

  /**
   * \brief Constructs a \ref Device.
   * \param physical_device The physical device to create the logical device from.
   * \param create_info \copybrief Device::CreateInfo
   */
  Device(vk::PhysicalDevice physical_device, const CreateInfo& create_info);

  /** \brief Gets the underlying Vulkan device handle. */
  [[nodiscard]] vk::Device operator*() const noexcept { return *device_; }

  /** \brief Gets a pointer to the underlying Vulkan device handle. */
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
