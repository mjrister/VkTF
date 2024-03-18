module;

#include <utility>

#include <vk_mem_alloc.h>
#include <vulkan/vulkan.hpp>

export module image;

namespace gfx {

export class Image {
public:
  Image(vk::Device device,
        vk::Format format,
        vk::Extent2D extent,
        vk::SampleCountFlagBits sample_count,
        vk::ImageUsageFlags image_usage_flags,
        vk::ImageAspectFlags image_aspect_flags,
        VmaAllocator allocator,
        const VmaAllocationCreateInfo& allocation_create_info);

  Image(const Image&) = delete;
  Image(Image&& image) noexcept { *this = std::move(image); }

  Image& operator=(const Image&) = delete;
  Image& operator=(Image&& image) noexcept;

  ~Image() noexcept;

  [[nodiscard]] vk::ImageView image_view() const noexcept { return *image_view_; }
  [[nodiscard]] vk::Format format() const noexcept { return format_; }

private:
  vk::Image image_;
  vk::UniqueImageView image_view_;
  vk::Format format_{};
  VmaAllocator allocator_{};
  VmaAllocation allocation_{};
};

}  // namespace gfx

module :private;

namespace gfx {

Image::Image(const vk::Device device,
             const vk::Format format,
             const vk::Extent2D extent,
             const vk::SampleCountFlagBits sample_count,
             const vk::ImageUsageFlags image_usage_flags,
             const vk::ImageAspectFlags image_aspect_flags,
             const VmaAllocator allocator,
             const VmaAllocationCreateInfo& allocation_create_info)
    : format_{format}, allocator_{allocator} {
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
  const auto result =
      vmaCreateImage(allocator_, &image_create_info, &allocation_create_info, &image, &allocation_, nullptr);
  vk::resultCheck(static_cast<vk::Result>(result), "Image creation failed");
  image_ = image;

  image_view_ = device.createImageViewUnique(vk::ImageViewCreateInfo{
      .image = image_,
      .viewType = vk::ImageViewType::e2D,
      .format = format,
      .subresourceRange =
          vk::ImageSubresourceRange{.aspectMask = image_aspect_flags, .levelCount = 1, .layerCount = 1}});
}

Image& Image::operator=(Image&& image) noexcept {
  if (this != &image) {
    allocator_ = std::exchange(image.allocator_, {});
    allocation_ = std::exchange(image.allocation_, {});
    image_ = std::exchange(image.image_, {});
    image_view_ = std::exchange(image.image_view_, {});
    format_ = std::exchange(image.format_, {});
  }
  return *this;
}

Image::~Image() noexcept {
  if (allocator_ != nullptr) {
    vmaDestroyImage(allocator_, image_, allocation_);
  }
}

}  // namespace gfx
