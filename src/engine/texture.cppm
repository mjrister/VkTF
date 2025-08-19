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

/**
 * @brief A texture in host-visible memory.
 * @details This class handles creating a host-visible staging buffer with raw image data for a KTX texture.
 */
export class [[nodiscard]] StagingTexture {
public:
  /**
   * @brief Creates a @ref StagingTexture.
   * @param allocator The allocator for creating a staging buffer.
   * @param ktx_texture2 The KTX texture to copy to a staging buffer.
   */
  StagingTexture(const vma::Allocator& allocator, const ktxTexture2& ktx_texture2);

  /** @brief Gets the staging buffer containing texture image data. */
  [[nodiscard]] const HostVisibleBuffer& buffer() const noexcept { return buffer_; }

  /** @brief Gets the subregions to copy for each mipmap image in the staging buffer. */
  [[nodiscard]] const std::vector<vk::BufferImageCopy>& buffer_image_copies() const { return buffer_image_copies_; }

  /** @brief Gets the texture image format. */
  [[nodiscard]] vk::Format image_format() const noexcept { return image_format_; }

  /** @brief Gets the texture image dimensions. */
  [[nodiscard]] vk::Extent2D image_extent() const noexcept { return image_extent_; }

private:
  HostVisibleBuffer buffer_;
  std::vector<vk::BufferImageCopy> buffer_image_copies_;
  vk::Format image_format_;
  vk::Extent2D image_extent_;
};

/**
 * @brief A texture in device-local memory.
 * @details This class handles creating a device-local image for a PBR material, recording copy commands to transfer
 *          data from a host-visible staging buffer, and assigning a sampler to filter the image during rendering.
 */
export class [[nodiscard]] Texture {
public:
  /** @brief The parameters for creating a @ref Texture. */
  struct [[nodiscard]] CreateInfo {
    /** @brief The staging texture to copy to device-local memory. */
    const StagingTexture& staging_texture;

    /** @brief The sampler defining how to filter the underlying texture image. */
    vk::Sampler sampler;
  };

  /**
   * @brief Creates a @ref Texture.
   * @param allocator The allocator for creating the device-local image.
   * @param command_buffer The command buffer for recording copy commands.
   * @param create_info @copybrief Texture::CreateInfo
   * @warning The caller is responsible for submitting @p command_buffer to a Vulkan queue to begin execution.
   */
  Texture(const vma::Allocator& allocator, vk::CommandBuffer command_buffer, const CreateInfo& create_info);

  /** @brief Gets the texture image view. */
  [[nodiscard]] vk::ImageView image_view() const noexcept { return image_.image_view(); }

  /** @brief Gets the texture sampler. */
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
