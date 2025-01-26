module;

#include <cstdint>
#include <initializer_list>

#include <vulkan/vulkan.hpp>

export module instance;

import window;

namespace vktf {

export class Instance {
public:
  static constexpr auto kApiVersion = vk::ApiVersion13;

  Instance();

  [[nodiscard]] vk::Instance operator*() const noexcept { return *instance_; }

private:
  vk::UniqueInstance instance_;
};

}  // namespace vktf

module :private;

namespace vktf {

Instance::Instance() {
#if VULKAN_HPP_DISPATCH_LOADER_DYNAMIC == 1
  VULKAN_HPP_DEFAULT_DISPATCHER.init();
#endif

  static constexpr vk::ApplicationInfo kApplicationInfo{.apiVersion = kApiVersion};
  static constexpr std::initializer_list<const char* const> kInstanceLayers = {
#ifndef NDEBUG
      "VK_LAYER_KHRONOS_validation"
#endif
  };
  const auto instance_extensions = Window::GetInstanceExtensions();

  instance_ = vk::createInstanceUnique(
      vk::InstanceCreateInfo{.pApplicationInfo = &kApplicationInfo,
                             .enabledLayerCount = static_cast<std::uint32_t>(kInstanceLayers.size()),
                             .ppEnabledLayerNames = std::data(kInstanceLayers),
                             .enabledExtensionCount = static_cast<std::uint32_t>(instance_extensions.size()),
                             .ppEnabledExtensionNames = instance_extensions.data()});

#if VULKAN_HPP_DISPATCH_LOADER_DYNAMIC == 1
  VULKAN_HPP_DEFAULT_DISPATCHER.init(*instance_);
#endif
}

}  // namespace vktf
