module;

#include <cstdint>
#include <vector>

#include <ktx.h>
#include <vulkan/vulkan.hpp>

export module texture;

import buffer;
import data_view;
import image;
import ktx_texture;
import vma_allocator;

namespace vktf {

export class [[nodiscard]] StagingTexture {
public:
  StagingTexture(const vma::Allocator& allocator, const ktxTexture2& ktx_texture2);

  [[nodiscard]] const HostVisibleBuffer& buffer() const noexcept { return buffer_; }
  [[nodiscard]] const std::vector<vk::BufferImageCopy>& buffer_image_copies() const { return buffer_image_copies_; }
  [[nodiscard]] vk::Format image_format() const noexcept { return image_format_; }
  [[nodiscard]] vk::Extent2D image_extent() const noexcept { return image_extent_; }

private:
  HostVisibleBuffer buffer_;
  std::vector<vk::BufferImageCopy> buffer_image_copies_;
  vk::Format image_format_;
  vk::Extent2D image_extent_;
};

export class [[nodiscard]] Texture {
public:
  struct [[nodiscard]] CreateInfo {
    const StagingTexture& staging_texture;
    vk::Sampler sampler;
  };

  Texture(const vma::Allocator& allocator, vk::CommandBuffer command_buffer, const CreateInfo& create_info);

  [[nodiscard]] vk::ImageView image_view() const noexcept { return image_.image_view(); }
  [[nodiscard]] vk::Sampler sampler() const noexcept { return sampler_; }

private:
  Image image_;
  vk::Sampler sampler_;
};

}  // namespace vktf

module :private;

namespace vktf {

namespace {

Image CreateDeviceLocalImage(const vma::Allocator& allocator,
                             const vk::CommandBuffer command_buffer,
                             const StagingTexture& staging_texture) {
  Image device_local_image{
      allocator,
      Image::CreateInfo{.format = staging_texture.image_format(),
                        .extent = staging_texture.image_extent(),
                        .mip_levels = static_cast<std::uint32_t>(staging_texture.buffer_image_copies().size()),
                        .sample_count = vk::SampleCountFlagBits::e1,
                        .usage_flags = vk::ImageUsageFlagBits::eSampled | vk::ImageUsageFlagBits::eTransferDst,
                        .aspect_mask = vk::ImageAspectFlagBits::eColor,
                        .allocation_create_info = vma::kDeviceLocalAllocationCreateInfo}};
  device_local_image.Copy(*staging_texture.buffer(), staging_texture.buffer_image_copies(), command_buffer);
  return device_local_image;
}

}  // namespace

StagingTexture::StagingTexture(const vma::Allocator& allocator, const ktxTexture2& ktx_texture2)
    : buffer_{CreateStagingBuffer(allocator, DataView{ktx_texture2.pData, ktx_texture2.dataSize})},
      buffer_image_copies_{ktx::GetBufferImageCopies(ktx_texture2)},
      image_format_{static_cast<vk::Format>(ktx_texture2.vkFormat)},
      image_extent_{ktx_texture2.baseWidth, ktx_texture2.baseHeight} {}

Texture::Texture(const vma::Allocator& allocator, const vk::CommandBuffer command_buffer, const CreateInfo& create_info)
    : image_{CreateDeviceLocalImage(allocator, command_buffer, create_info.staging_texture)},
      sampler_{create_info.sampler} {}

}  // namespace vktf
