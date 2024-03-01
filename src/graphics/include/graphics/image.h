#ifndef SRC_GRAPHICS_INCLUDE_GRAPHICS_IMAGE_H_
#define SRC_GRAPHICS_INCLUDE_GRAPHICS_IMAGE_H_

#include <utility>

#include <vk_mem_alloc.h>
#include <vulkan/vulkan.hpp>

namespace gfx {

class Image {
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
  VmaAllocator allocator_{};
  VmaAllocation allocation_{};
  vk::Image image_;
  vk::UniqueImageView image_view_;
  vk::Format format_{};
};

}  // namespace gfx

#endif  // SRC_GRAPHICS_INCLUDE_GRAPHICS_IMAGE_H_
