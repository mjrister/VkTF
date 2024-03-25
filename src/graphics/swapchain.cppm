module;

#include <algorithm>
#include <array>
#include <cstdint>
#include <limits>
#include <ranges>
#include <tuple>
#include <vector>

#include <vulkan/vulkan.hpp>

export module swapchain;

import device;
import window;

namespace gfx {

export class Swapchain {
public:
  Swapchain(const Window& window, vk::SurfaceKHR surface, const Device& device);

  [[nodiscard]] vk::SwapchainKHR operator*() const noexcept { return *swapchain_; }
  [[nodiscard]] const vk::SwapchainKHR* operator->() const noexcept { return &(*swapchain_); }

  [[nodiscard]] vk::Format image_format() const noexcept { return image_format_; }
  [[nodiscard]] vk::Extent2D image_extent() const noexcept { return image_extent_; }

  [[nodiscard]] std::ranges::view auto image_views() const {
    return image_views_ | std::views::transform([](const auto& image_view) { return *image_view; });
  }

private:
  vk::UniqueSwapchainKHR swapchain_;
  vk::Format image_format_ = vk::Format::eUndefined;
  vk::Extent2D image_extent_;
  std::vector<vk::UniqueImageView> image_views_;
};

}  // namespace gfx

module :private;

namespace {

vk::SurfaceFormatKHR GetSwapchainSurfaceFormat(const vk::PhysicalDevice physical_device, const vk::SurfaceKHR surface) {
  const auto surface_formats = physical_device.getSurfaceFormatsKHR(surface);
  if (static constexpr vk::SurfaceFormatKHR kTargetSurfaceFormat{vk::Format::eB8G8R8A8Srgb,
                                                                 vk::ColorSpaceKHR::eSrgbNonlinear};
      std::ranges::contains(surface_formats, kTargetSurfaceFormat)) {
    return kTargetSurfaceFormat;
  }
  return surface_formats.front();  // required to be non-empty by the Vulkan specification
}

vk::PresentModeKHR GetSwapchainPresentMode(const vk::PhysicalDevice physical_device, const vk::SurfaceKHR surface) {
  const auto present_modes = physical_device.getSurfacePresentModesKHR(surface);
  if (static constexpr auto kTargetPresentMode = vk::PresentModeKHR::eFifoRelaxed;
      std::ranges::contains(present_modes, kTargetPresentMode)) {
    return kTargetPresentMode;
  }
  return vk::PresentModeKHR::eFifo;  // required to be supported by the Vulkan specification
}

std::uint32_t GetSwapchainImageCount(const vk::SurfaceCapabilitiesKHR& surface_capabilities) {
  const auto min_image_count = surface_capabilities.minImageCount;
  auto max_image_count = surface_capabilities.maxImageCount;
  if (static constexpr std::uint32_t kNoLimitImageCount = 0; max_image_count == kNoLimitImageCount) {
    max_image_count = std::numeric_limits<std::uint32_t>::max();
  }
  return std::min(min_image_count + 1, max_image_count);
}

vk::Extent2D GetSwapchainImageExtent(const vk::SurfaceCapabilitiesKHR& surface_capabilities,
                                     const std::pair<int, int> framebuffer_size) {
  if (static constexpr auto kUndefinedExtent = std::numeric_limits<std::uint32_t>::max();
      surface_capabilities.currentExtent != vk::Extent2D{.width = kUndefinedExtent, .height = kUndefinedExtent}) {
    return surface_capabilities.currentExtent;
  }
  const auto [min_width, min_height] = surface_capabilities.minImageExtent;
  const auto [max_width, max_height] = surface_capabilities.maxImageExtent;
  const auto [framebuffer_width, framebuffer_height] = framebuffer_size;
  return vk::Extent2D{.width = std::clamp(static_cast<std::uint32_t>(framebuffer_width), min_width, max_width),
                      .height = std::clamp(static_cast<std::uint32_t>(framebuffer_height), min_height, max_height)};
}

std::vector<vk::UniqueImageView> CreateSwapchainImageViews(const vk::Device device,
                                                           const vk::SwapchainKHR swapchain,
                                                           const vk::Format image_format) {
  return device.getSwapchainImagesKHR(swapchain)  //
         | std::views::transform([device, image_format](const auto image) {
             return device.createImageViewUnique(vk::ImageViewCreateInfo{
                 .image = image,
                 .viewType = vk::ImageViewType::e2D,
                 .format = image_format,
                 .subresourceRange = vk::ImageSubresourceRange{.aspectMask = vk::ImageAspectFlagBits::eColor,
                                                               .levelCount = 1,
                                                               .layerCount = 1}});
           })
         | std::ranges::to<std::vector>();
}

std::tuple<vk::UniqueSwapchainKHR, vk::Format, vk::Extent2D> CreateSwapchain(const gfx::Device& device,
                                                                             const gfx::Window& window,
                                                                             const vk::SurfaceKHR surface) {
  const auto physical_device = device.physical_device();
  const auto surface_capabilities = physical_device.getSurfaceCapabilitiesKHR(surface);
  const auto [image_format, image_color_space] = GetSwapchainSurfaceFormat(physical_device, surface);
  const auto image_extent = GetSwapchainImageExtent(surface_capabilities, window.GetFramebufferSize());

  vk::SwapchainCreateInfoKHR swapchain_create_info{.surface = surface,
                                                   .minImageCount = GetSwapchainImageCount(surface_capabilities),
                                                   .imageFormat = image_format,
                                                   .imageColorSpace = image_color_space,
                                                   .imageExtent = image_extent,
                                                   .imageArrayLayers = 1,
                                                   .imageUsage = vk::ImageUsageFlagBits::eColorAttachment,
                                                   .presentMode = GetSwapchainPresentMode(physical_device, surface),
                                                   .clipped = vk::True};

  if (const auto [graphics_index, present_index, _] = device.queue_family_indices(); graphics_index != present_index) {
    const std::array queue_family_indices{graphics_index, present_index};
    swapchain_create_info.imageSharingMode = vk::SharingMode::eConcurrent;
    swapchain_create_info.queueFamilyIndexCount = 2;
    swapchain_create_info.pQueueFamilyIndices = queue_family_indices.data();
  } else {
    swapchain_create_info.imageSharingMode = vk::SharingMode::eExclusive;
    swapchain_create_info.queueFamilyIndexCount = 1;
    swapchain_create_info.pQueueFamilyIndices = &graphics_index;
  }

  return std::tuple{device->createSwapchainKHRUnique(swapchain_create_info), image_format, image_extent};
}

}  // namespace

namespace gfx {

Swapchain::Swapchain(const Window& window, const vk::SurfaceKHR surface, const Device& device) {
  std::tie(swapchain_, image_format_, image_extent_) = CreateSwapchain(device, window, surface);
  image_views_ = CreateSwapchainImageViews(*device, *swapchain_, image_format_);
}

}  // namespace gfx
