module;

#include <cstdint>
#include <format>
#include <ranges>
#include <stdexcept>
#include <string_view>
#include <unordered_set>
#include <vector>

#include <vulkan/vulkan.hpp>

export module instance;

namespace vktf {

export class [[nodiscard]] Instance {
public:
  struct [[nodiscard]] CreateInfo {
    const vk::ApplicationInfo& application_info;
    const std::vector<const char*>& required_layers;
    const std::vector<const char*>& required_extensions;
  };

  explicit Instance(const CreateInfo& create_info);

  [[nodiscard]] vk::Instance operator*() const noexcept { return *instance_; }

private:
  vk::UniqueInstance instance_;
};

}  // namespace vktf

module :private;

namespace vktf {

namespace {

void ValidateInstanceLayers(const std::vector<const char*>& required_layers) {
  const auto instance_layer_properties = vk::enumerateInstanceLayerProperties();
  const auto instance_layers = instance_layer_properties
                               | std::views::transform([](const auto& layer_properties) -> std::string_view {
                                   return layer_properties.layerName;
                                 })
                               | std::ranges::to<std::unordered_set>();

  for (const std::string_view required_layer : required_layers) {
    if (!instance_layers.contains(required_layer)) {
      throw std::runtime_error{std::format("Missing required instance layer {}", required_layer)};
    }
  }
}

void ValidateInstanceExtensions(const std::vector<const char*>& required_extensions) {
  const auto instance_extension_properties = vk::enumerateInstanceExtensionProperties();
  const auto instance_extensions = instance_extension_properties
                                   | std::views::transform([](const auto& extension_properties) -> std::string_view {
                                       return extension_properties.extensionName;
                                     })
                                   | std::ranges::to<std::unordered_set>();

  for (const std::string_view required_extension : required_extensions) {
    if (!instance_extensions.contains(required_extension)) {
      throw std::runtime_error{std::format("Missing required instance extension {}", required_extension)};
    }
  }
}

vk::UniqueInstance CreateInstance(const Instance::CreateInfo& create_info) {
#if VULKAN_HPP_DISPATCH_LOADER_DYNAMIC == 1
  VULKAN_HPP_DEFAULT_DISPATCHER.init();
#endif

  const auto& [application_info, required_layers, required_extensions] = create_info;
  ValidateInstanceLayers(required_layers);
  ValidateInstanceExtensions(required_extensions);

  auto instance = vk::createInstanceUnique(
      vk::InstanceCreateInfo{.pApplicationInfo = &application_info,
                             .enabledLayerCount = static_cast<std::uint32_t>(required_layers.size()),
                             .ppEnabledLayerNames = required_layers.data(),
                             .enabledExtensionCount = static_cast<std::uint32_t>(required_extensions.size()),
                             .ppEnabledExtensionNames = required_extensions.data()});

#if VULKAN_HPP_DISPATCH_LOADER_DYNAMIC == 1
  VULKAN_HPP_DEFAULT_DISPATCHER.init(*instance);
#endif

  return instance;
}

}  // namespace

Instance::Instance(const CreateInfo& create_info) : instance_{CreateInstance(create_info)} {}

}  // namespace vktf
