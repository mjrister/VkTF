#include "graphics/image.h"

gfx::Image::Image(const vk::Device device,
                  const vk::Format format,
                  const vk::Extent2D extent,
                  const vk::SampleCountFlagBits sample_count,
                  const vk::ImageUsageFlags image_usage_flags,
                  const vk::ImageAspectFlags image_aspect_flags,
                  const VmaAllocator allocator,
                  const VmaAllocationCreateInfo& allocation_create_info)
    : allocator_{allocator}, format_{format} {
  const VkImageCreateInfo image_create_info{
      .sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
      .imageType = VK_IMAGE_TYPE_2D,
      .format = static_cast<VkFormat>(format),
      .extent = VkExtent3D{.width = extent.width, .height = extent.height, .depth = 1},
      .mipLevels = 1,
      .arrayLayers = 1,
      .samples = static_cast<VkSampleCountFlagBits>(sample_count),
      .usage = static_cast<VkImageUsageFlags>(image_usage_flags)};

  VkImage image{};
  vmaCreateImage(allocator_, &image_create_info, &allocation_create_info, &image, &allocation_, nullptr);

  image_ = vk::Image{image};
  image_view_ = device.createImageViewUnique(vk::ImageViewCreateInfo{
      .image = image_,
      .viewType = vk::ImageViewType::e2D,
      .format = format,
      .subresourceRange =
          vk::ImageSubresourceRange{.aspectMask = image_aspect_flags, .levelCount = 1, .layerCount = 1}});
}

gfx::Image& gfx::Image::operator=(Image&& image) noexcept {
  if (this != &image) {
    allocator_ = std::exchange(image.allocator_, {});
    allocation_ = std::exchange(image.allocation_, {});
    image_ = std::exchange(image.image_, {});
    image_view_ = std::exchange(image.image_view_, {});
    format_ = std::exchange(image.format_, {});
  }
  return *this;
}

gfx::Image::~Image() {
  if (allocator_ != nullptr) {
    vmaDestroyImage(allocator_, image_, allocation_);
  }
}
