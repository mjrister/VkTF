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
  std::uint32_t width;
  std::uint32_t height;
  std::uint32_t mip_levels;
  vk::Format format;
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
      width{ktx_texture->baseWidth},
      height{ktx_texture->baseHeight},
      mip_levels{ktx_texture->numLevels},
      format{static_cast<vk::Format>(ktx_texture->vkFormat)},
      buffer_image_copies{ktx_texture.GetBufferImageCopies()} {}

Texture::Texture(const StagingTexture& staging_texture,
                 const vk::Sampler sampler,
                 const vk::Device device,
                 const vk::CommandBuffer command_buffer,
                 const VmaAllocator allocator)
    : image_{staging_texture.format,
             vk::Extent2D{.width = staging_texture.width, .height = staging_texture.height},
             staging_texture.mip_levels,
             vk::SampleCountFlagBits::e1,
             vk::ImageUsageFlagBits::eSampled | vk::ImageUsageFlagBits::eTransferDst,
             vk::ImageAspectFlagBits::eColor,
             device,
             allocator},
      sampler_{sampler} {
  image_.Copy(*staging_texture.staging_buffer, staging_texture.buffer_image_copies, command_buffer);
}

}  // namespace vktf
