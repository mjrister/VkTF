#include "graphics/image.h"

gfx::Image::Image(const vk::Device device,
                  const vk::Format format,
                  const vk::Extent2D extent,
                  const vk::SampleCountFlagBits sample_count,
                  const vk::ImageUsageFlags image_usage_flags,
                  const vk::ImageAspectFlags image_aspect_flags,
                  const VmaAllocator allocator,
                  const VmaAllocationCreateFlags allocation_create_flags)
    : allocator_{allocator}, format_{format} {
  VkImageCreateInfo image_create_info{};
  image_create_info.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
  image_create_info.imageType = VK_IMAGE_TYPE_2D;
  image_create_info.format = static_cast<VkFormat>(format);
  image_create_info.extent = VkExtent3D{.width = extent.width, .height = extent.height, .depth = 1};
  image_create_info.mipLevels = 1;
  image_create_info.arrayLayers = 1;
  image_create_info.samples = static_cast<VkSampleCountFlagBits>(sample_count);
  image_create_info.usage = static_cast<VkImageUsageFlags>(image_usage_flags);

  VmaAllocationCreateInfo allocation_create_info{};
  allocation_create_info.usage = VMA_MEMORY_USAGE_AUTO;
  allocation_create_info.flags = allocation_create_flags;

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
