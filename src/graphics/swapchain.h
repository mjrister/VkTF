#ifndef GRAPHICS_SWAPCHAIN_H_
#define GRAPHICS_SWAPCHAIN_H_

#include <algorithm>
#include <ranges>
#include <vector>

#include <vulkan/vulkan.hpp>

namespace gfx {
struct QueueFamilyIndices;

class Swapchain {
public:
  Swapchain(vk::Device device,
            vk::PhysicalDevice physical_device,
            vk::SurfaceKHR surface,
            vk::Extent2D framebuffer_extent,
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

#endif  // GRAPHICS_SWAPCHAIN_H_
