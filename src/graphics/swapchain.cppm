module;

#include <algorithm>
#include <array>
#include <cassert>
#include <cstdint>
#include <limits>
#include <ranges>
#include <vector>

#include <vulkan/vulkan.hpp>

export module swapchain;

import physical_device;

namespace gfx {

export class Swapchain {
public:
  Swapchain(vk::SurfaceKHR surface,
            vk::PhysicalDevice physical_device,
            vk::Device device,
            vk::Extent2D image_extent,
            const QueueFamilyIndices& queue_family_indices);

  [[nodiscard]] vk::SwapchainKHR operator*() const noexcept { return *swapchain_; }

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
  if (static constexpr vk::SurfaceFormatKHR kTargetFormat{vk::Format::eB8G8R8A8Srgb, vk::ColorSpaceKHR::eSrgbNonlinear};
      std::ranges::contains(surface_formats, kTargetFormat)) {
    return kTargetFormat;
  }
  assert(!surface_formats.empty());  // required by the Vulkan specification
  return surface_formats.front();
}

vk::PresentModeKHR GetSwapchainPresentMode(const vk::PhysicalDevice physical_device, const vk::SurfaceKHR surface) {
  const auto present_modes = physical_device.getSurfacePresentModesKHR(surface);
  if (static constexpr auto kTargetPresentMode = vk::PresentModeKHR::eFifoRelaxed;
      std::ranges::contains(present_modes, kTargetPresentMode)) {
    return kTargetPresentMode;
  }
  assert(std::ranges::contains(present_modes, vk::PresentModeKHR::eFifo));  // required by the Vulkan specification
  return vk::PresentModeKHR::eFifo;
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
                                     const vk::Extent2D framebuffer_extent) {
  if (static constexpr auto kUndefinedExtent = std::numeric_limits<std::uint32_t>::max();
      surface_capabilities.currentExtent != vk::Extent2D{.width = kUndefinedExtent, .height = kUndefinedExtent}) {
    return surface_capabilities.currentExtent;
  }
  const auto& [min_image_width, min_image_height] = surface_capabilities.minImageExtent;
  const auto& [max_image_width, max_image_height] = surface_capabilities.maxImageExtent;
  const auto& [framebuffer_width, framebuffer_height] = framebuffer_extent;
  return vk::Extent2D{.width = std::clamp(framebuffer_width, min_image_width, max_image_width),
                      .height = std::clamp(framebuffer_height, min_image_height, max_image_height)};
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

}  // namespace

namespace gfx {

Swapchain::Swapchain(const vk::SurfaceKHR surface,
                     const vk::PhysicalDevice physical_device,
                     const vk::Device device,
                     const vk::Extent2D image_extent,
                     const QueueFamilyIndices& queue_family_indices) {
  const auto surface_capabilities = physical_device.getSurfaceCapabilitiesKHR(surface);
  const auto surface_format = GetSwapchainSurfaceFormat(physical_device, surface);

  vk::SwapchainCreateInfoKHR swapchain_create_info{
      .surface = surface,
      .minImageCount = GetSwapchainImageCount(surface_capabilities),
      .imageFormat = surface_format.format,
      .imageColorSpace = surface_format.colorSpace,
      .imageExtent = GetSwapchainImageExtent(surface_capabilities, image_extent),
      .imageArrayLayers = 1,
      .imageUsage = vk::ImageUsageFlagBits::eColorAttachment,
      .presentMode = GetSwapchainPresentMode(physical_device, surface),
      .clipped = vk::True};

  const auto& [graphics_index, present_index] = queue_family_indices;
  const std::array graphics_and_present_index{graphics_index, present_index};
  if (graphics_index != present_index) {
    swapchain_create_info.imageSharingMode = vk::SharingMode::eConcurrent;
    swapchain_create_info.queueFamilyIndexCount = 2;
    swapchain_create_info.pQueueFamilyIndices = graphics_and_present_index.data();
  } else {
    swapchain_create_info.imageSharingMode = vk::SharingMode::eExclusive;
    swapchain_create_info.queueFamilyIndexCount = 1;
    swapchain_create_info.pQueueFamilyIndices = &graphics_index;
  }

  swapchain_ = device.createSwapchainKHRUnique(swapchain_create_info);
  image_format_ = surface_format.format;
  image_extent_ = swapchain_create_info.imageExtent;
  image_views_ = CreateSwapchainImageViews(device, *swapchain_, image_format_);
}

}  // namespace gfx
