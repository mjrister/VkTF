#ifndef SRC_GRAPHICS_INCLUDE_GRAPHICS_IMAGE_H_
#define SRC_GRAPHICS_INCLUDE_GRAPHICS_IMAGE_H_

#include <utility>

#include <vk_mem_alloc.h>
#include <vulkan/vulkan.hpp>

namespace gfx {

class Image {
public:
  Image(const vk::Format format,
        const vk::Extent2D extent,
        const vk::SampleCountFlagBits sample_count,
        const vk::ImageUsageFlags image_usage_flags,
        const vk::ImageAspectFlags image_aspect_flags,
        const vk::Device device,
        const VmaAllocator allocator,
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
  vk::Format format_ = vk::Format::eUndefined;
  VmaAllocator allocator_ = nullptr;
  VmaAllocation allocation_ = nullptr;
};

}  // namespace gfx

#endif  // SRC_GRAPHICS_INCLUDE_GRAPHICS_IMAGE_H_
