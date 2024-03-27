module;

#include <algorithm>
#include <array>
#include <concepts>
#include <cstdint>
#include <optional>
#include <ranges>
#include <stdexcept>
#include <unordered_set>
#include <vector>

#include <vulkan/vulkan.hpp>

export module device;

namespace gfx {

export struct QueueFamilyIndices {
  std::uint32_t graphics_index{};
  std::uint32_t present_index{};
  std::uint32_t transfer_index{};
};

export class Device {
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

module :private;

namespace {

vk::PhysicalDevice SelectPhysicalDevice(const vk::Instance instance) {
  const auto physical_devices = instance.enumeratePhysicalDevices();
  if (physical_devices.empty()) {
    throw std::runtime_error{"No supported physical device could be found"};
  }
  static constexpr auto kIsDiscreteGpu = [](const auto physical_device) {
    const auto device_type = physical_device.getProperties().deviceType;
    return device_type == vk::PhysicalDeviceType::eDiscreteGpu;
  };
  if (const auto iterator = std::ranges::find_if(physical_devices, kIsDiscreteGpu);
      iterator != std::ranges::cend(physical_devices)) {
    return *iterator;
  }
  return physical_devices.front();
}

std::optional<std::uint32_t> FindQueueFamilyIndex(
    const std::vector<vk::QueueFamilyProperties>& all_queue_family_properties,
    std::predicate<const vk::QueueFamilyProperties&> auto&& find_fn) {
  if (const auto iterator = std::ranges::find_if(all_queue_family_properties, find_fn);
      iterator != all_queue_family_properties.cend()) {
    const auto index = std::ranges::distance(all_queue_family_properties.cbegin(), iterator);
    return static_cast<std::uint32_t>(index);
  }
  return std::nullopt;
}

gfx::QueueFamilyIndices FindQueueFamilyIndices(const vk::PhysicalDevice physical_device, const vk::SurfaceKHR surface) {
  const auto all_queue_family_properties = physical_device.getQueueFamilyProperties();

  const auto graphics_index =
      FindQueueFamilyIndex(all_queue_family_properties, [](const auto& queue_family_properties) {
        return static_cast<bool>(queue_family_properties.queueFlags & vk::QueueFlagBits::eGraphics);
      });

  const auto present_index =
      FindQueueFamilyIndex(all_queue_family_properties,
                           [physical_device, surface, index = 0u](const auto& /*queue_family_properties*/) mutable {
                             return physical_device.getSurfaceSupportKHR(index++, surface) == vk::True;
                           });

  const auto transfer_index =
      FindQueueFamilyIndex(all_queue_family_properties, [](const auto& queue_family_properties) {
        // prefer a dedicated transfer queue family to enable asynchronous transfers to device memory
        return static_cast<bool>(queue_family_properties.queueFlags & vk::QueueFlagBits::eTransfer)
               && !static_cast<bool>(queue_family_properties.queueFlags & vk::QueueFlagBits::eGraphics);
      });

  if (graphics_index.has_value() && present_index.has_value()) {
    return gfx::QueueFamilyIndices{.graphics_index = *graphics_index,
                                   .present_index = *present_index,
                                   // graphics queue family always implicitly accept transfer commands
                                   .transfer_index = transfer_index.value_or(*graphics_index)};
  }

  throw std::runtime_error{"Physical device does not support required queue families"};
}

vk::UniqueDevice CreateDevice(const vk::PhysicalDevice physical_device,
                              const gfx::QueueFamilyIndices& queue_family_indices) {
  static constexpr auto kHighestNormalizedQueuePriority = 1.0f;
  const auto [graphics_index, present_index, transfer_index] = queue_family_indices;

  const auto device_queue_create_info =
      std::unordered_set{graphics_index, present_index, transfer_index}
      | std::views::transform([](const auto queue_family_index) {
          return vk::DeviceQueueCreateInfo{.queueFamilyIndex = queue_family_index,
                                           .queueCount = 1,
                                           .pQueuePriorities = &kHighestNormalizedQueuePriority};
        })
      | std::ranges::to<std::vector>();

  static constexpr std::array kDeviceExtensions{VK_KHR_SWAPCHAIN_EXTENSION_NAME};

  auto device = physical_device.createDeviceUnique(
      vk::DeviceCreateInfo{.queueCreateInfoCount = static_cast<std::uint32_t>(device_queue_create_info.size()),
                           .pQueueCreateInfos = device_queue_create_info.data(),
                           .enabledExtensionCount = static_cast<std::uint32_t>(kDeviceExtensions.size()),
                           .ppEnabledExtensionNames = kDeviceExtensions.data()});

#if VULKAN_HPP_DISPATCH_LOADER_DYNAMIC == 1
  VULKAN_HPP_DEFAULT_DISPATCHER.init(*device);
#endif

  return device;
}

}  // namespace

namespace gfx {

Device::Device(const vk::Instance instance, const vk::SurfaceKHR surface)
    : physical_device_{SelectPhysicalDevice(instance)},
      queue_family_indices_{FindQueueFamilyIndices(physical_device_, surface)},
      device_{CreateDevice(physical_device_, queue_family_indices_)},
      graphics_queue_{device_->getQueue(queue_family_indices_.graphics_index, 0)},
      present_queue_{device_->getQueue(queue_family_indices_.present_index, 0)},
      transfer_queue_{device_->getQueue(queue_family_indices_.transfer_index, 0)} {}

}  // namespace gfx
