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

export struct StagingTexture {
  StagingTexture(const KtxTexture& ktx_texture, const VmaAllocator allocator);

  HostVisibleBuffer staging_buffer;
  vk::Format image_format;
  vk::Extent2D image_extent;
  std::vector<vk::BufferImageCopy> buffer_image_copies;
};

export class Texture {
public:
  Texture(const StagingTexture& staging_texture,
          const vk::Sampler sampler,
          const vk::Device device,
          const vk::CommandBuffer command_buffer,
          const VmaAllocator allocator);

  [[nodiscard]] vk::ImageView image_view() const noexcept { return image_.image_view(); }
  [[nodiscard]] vk::Sampler sampler() const noexcept { return sampler_; }

private:
  Image image_;
  vk::Sampler sampler_;
};

}  // namespace vktf

module :private;

namespace vktf {

StagingTexture::StagingTexture(const KtxTexture& ktx_texture, const VmaAllocator allocator)
    : staging_buffer{CreateStagingBuffer(DataView<const std::uint8_t>{ktx_texture->pData, ktx_texture->dataSize},
                                         allocator)},
      image_format{static_cast<vk::Format>(ktx_texture->vkFormat)},
      image_extent{.width = ktx_texture->baseWidth, .height = ktx_texture->baseHeight},
      buffer_image_copies{ktx_texture.GetBufferImageCopies()} {}

Texture::Texture(const StagingTexture& staging_texture,
                 const vk::Sampler sampler,
                 const vk::Device device,
                 const vk::CommandBuffer command_buffer,
                 const VmaAllocator allocator)
    : image_{staging_texture.image_format,
             staging_texture.image_extent,
             static_cast<std::uint32_t>(staging_texture.buffer_image_copies.size()),
             vk::SampleCountFlagBits::e1,
             vk::ImageUsageFlagBits::eSampled | vk::ImageUsageFlagBits::eTransferDst,
             vk::ImageAspectFlagBits::eColor,
             device,
             allocator},
      sampler_{sampler} {
  image_.Copy(*staging_texture.staging_buffer, staging_texture.buffer_image_copies, command_buffer);
}

}  // namespace vktf
