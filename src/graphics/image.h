#ifndef SRC_GRAPHICS_IMAGE_H_
#define SRC_GRAPHICS_IMAGE_H_

#include <vulkan/vulkan.hpp>

#include "graphics/buffer.h"
#include "graphics/memory.h"

namespace gfx {
class Device;

class Image {
public:
  Image(const Device& device,
        vk::Format format,
        vk::Extent2D extent,
        vk::SampleCountFlagBits sample_count,
        vk::ImageUsageFlags image_usage_flags,
        vk::ImageAspectFlags image_aspect_flags,
        vk::MemoryPropertyFlags memory_property_flags);

  [[nodiscard]] vk::ImageView image_view() const noexcept { return *image_view_; }
  [[nodiscard]] vk::Format format() const noexcept { return format_; }

  void Copy(const Device& device, vk::Buffer src_buffer) const;

private:
  vk::UniqueImage image_;
  vk::UniqueImageView image_view_;
  Memory memory_;
  vk::Format format_;
  vk::Extent2D extent_;
  vk::ImageAspectFlags aspect_flags_;
};

template <typename T>
[[nodiscard]] Image CreateDeviceLocalImage(const Device& device,
                                           const vk::Format format,
                                           const vk::Extent2D extent,
                                           const vk::SampleCountFlagBits sample_count,
                                           const vk::ImageUsageFlags usage_flags,
                                           const vk::ImageAspectFlags aspect_flags,
                                           const vk::ArrayProxy<const T> data) {
  Buffer host_visible_buffer{device,
                             sizeof(T) * data.size(),
                             vk::BufferUsageFlagBits::eTransferSrc,
                             vk::MemoryPropertyFlagBits::eHostVisible};
  host_visible_buffer.Copy(data);

  Image device_local_image{device,
                           format,
                           extent,
                           sample_count,
                           usage_flags | vk::ImageUsageFlagBits::eTransferDst,
                           aspect_flags,
                           vk::MemoryPropertyFlagBits::eDeviceLocal};
  device_local_image.Copy(device, *host_visible_buffer);

  return device_local_image;
}

}  // namespace gfx

#endif  // SRC_GRAPHICS_IMAGE_H_
