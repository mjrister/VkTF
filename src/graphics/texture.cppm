module;

#include <cstdint>
#include <vector>

#include <vk_mem_alloc.h>
#include <vulkan/vulkan.hpp>

export module texture;

import buffer;
import data_view;
import image;
import ktx_texture;

namespace vktf {

export class StagingTexture {
public:
  StagingTexture(const KtxTexture& ktx_texture, const VmaAllocator allocator);

  [[nodiscard]] const auto& staging_buffer() const noexcept { return staging_buffer_; }
  [[nodiscard]] const auto& buffer_image_copies() const noexcept { return buffer_image_copies_; }
  [[nodiscard]] auto image_format() const noexcept { return image_format_; }
  [[nodiscard]] auto image_extent() const noexcept { return image_extent_; }

private:
  HostVisibleBuffer staging_buffer_;
  std::vector<vk::BufferImageCopy> buffer_image_copies_;
  vk::Format image_format_;
  vk::Extent2D image_extent_;
};

export class Texture {
public:
  Texture(const StagingTexture& staging_texture,
          const vk::Sampler sampler,
          const vk::Device device,
          const vk::CommandBuffer command_buffer,
          const VmaAllocator allocator);

  [[nodiscard]] auto image_view() const noexcept { return image_.image_view(); }
  [[nodiscard]] auto sampler() const noexcept { return sampler_; }

private:
  Image image_;
  vk::Sampler sampler_;
};

}  // namespace vktf

module :private;

namespace vktf {

StagingTexture::StagingTexture(const KtxTexture& ktx_texture, const VmaAllocator allocator)
    : staging_buffer_{CreateStagingBuffer(DataView<const std::uint8_t>{ktx_texture->pData, ktx_texture->dataSize},
                                          allocator)},
      buffer_image_copies_{ktx_texture.GetBufferImageCopies()},
      image_format_{static_cast<vk::Format>(ktx_texture->vkFormat)},
      image_extent_{.width = ktx_texture->baseWidth, .height = ktx_texture->baseHeight} {}

Texture::Texture(const StagingTexture& staging_texture,
                 const vk::Sampler sampler,
                 const vk::Device device,
                 const vk::CommandBuffer command_buffer,
                 const VmaAllocator allocator)
    : image_{staging_texture.image_format(),
             staging_texture.image_extent(),
             static_cast<std::uint32_t>(staging_texture.buffer_image_copies().size()),
             vk::SampleCountFlagBits::e1,
             vk::ImageUsageFlagBits::eSampled | vk::ImageUsageFlagBits::eTransferDst,
             vk::ImageAspectFlagBits::eColor,
             device,
             allocator},
      sampler_{sampler} {
  image_.Copy(*staging_texture.staging_buffer(), staging_texture.buffer_image_copies(), command_buffer);
}

}  // namespace vktf
