#ifndef SRC_GRAPHICS_INCLUDE_GRAPHICS_IMAGE_H_
#define SRC_GRAPHICS_INCLUDE_GRAPHICS_IMAGE_H_

#include <utility>
#include <vector>

#include <vk_mem_alloc.h>
#include <vulkan/vulkan.hpp>

namespace gfx {

class Image {
public:
  Image(vk::Device device,
        vk::Format format,
        vk::Extent2D extent,
        std::uint32_t mip_levels,
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

  void Copy(vk::Buffer src_buffer,
            vk::CommandBuffer command_buffer,
            const std::vector<vk::BufferImageCopy>& buffer_image_copies) const;

private:
  vk::Image image_;
  vk::UniqueImageView image_view_;
  vk::Format format_ = vk::Format::eUndefined;
  std::uint32_t mip_levels_ = 0;
  vk::ImageAspectFlags image_aspect_flags_;
  VmaAllocator allocator_ = nullptr;
  VmaAllocation allocation_ = nullptr;
};

}  // namespace gfx

#endif  // SRC_GRAPHICS_INCLUDE_GRAPHICS_IMAGE_H_
