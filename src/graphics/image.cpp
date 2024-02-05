#include "graphics/image.h"

#include <utility>

#include "graphics/device.h"

namespace {

vk::UniqueImage CreateImage(const vk::Device device,
                            const vk::Format format,
                            const vk::Extent2D extent,
                            const vk::SampleCountFlagBits sample_count,
                            const vk::ImageUsageFlags image_usage_flags) {
  return device.createImageUnique(
      vk::ImageCreateInfo{.imageType = vk::ImageType::e2D,
                          .format = format,
                          .extent = vk::Extent3D{.width = extent.width, .height = extent.height, .depth = 1},
                          .mipLevels = 1,
                          .arrayLayers = 1,
                          .samples = sample_count,
                          .usage = image_usage_flags});
}

vk::UniqueImageView CreateImageView(const vk::Device device,
                                    const vk::Image image,
                                    const vk::Format format,
                                    const vk::ImageAspectFlags image_aspect_flags) {
  return device.createImageViewUnique(vk::ImageViewCreateInfo{
      .image = image,
      .viewType = vk::ImageViewType::e2D,
      .format = format,
      .subresourceRange =
          vk::ImageSubresourceRange{.aspectMask = image_aspect_flags, .levelCount = 1, .layerCount = 1}});
}

void TransitionImageLayout(const vk::CommandBuffer command_buffer,
                           const vk::Image image,
                           const vk::ImageSubresourceRange& image_subresource_range,
                           const std::pair<vk::PipelineStageFlags, vk::PipelineStageFlags> stage_masks,
                           const std::pair<vk::AccessFlags, vk::AccessFlags> access_masks,
                           const std::pair<vk::ImageLayout, vk::ImageLayout> image_layouts) {
  const auto [src_stage_mask, dst_stage_mask] = stage_masks;
  const auto [src_access_mask, dst_access_mask] = access_masks;
  const auto [old_layout, new_layout] = image_layouts;

  command_buffer.pipelineBarrier(src_stage_mask,
                                 dst_stage_mask,
                                 vk::DependencyFlags{},
                                 nullptr,
                                 nullptr,
                                 vk::ImageMemoryBarrier{.srcAccessMask = src_access_mask,
                                                        .dstAccessMask = dst_access_mask,
                                                        .oldLayout = old_layout,
                                                        .newLayout = new_layout,
                                                        .image = image,
                                                        .subresourceRange = image_subresource_range});
}

}  // namespace

gfx::Image::Image(const Device& device,
                  const vk::Format format,
                  const vk::Extent2D extent,
                  const vk::SampleCountFlagBits sample_count,
                  const vk::ImageUsageFlags image_usage_flags,
                  const vk::ImageAspectFlags image_aspect_flags,
                  const vk::MemoryPropertyFlags memory_property_flags)
    : image_{CreateImage(*device, format, extent, sample_count, image_usage_flags)},
      memory_{device, device->getImageMemoryRequirements(*image_), memory_property_flags},
      format_{format},
      extent_{extent},
      aspect_flags_{image_aspect_flags} {
  device->bindImageMemory(*image_, *memory_, 0);
  image_view_ = CreateImageView(*device, *image_, format, image_aspect_flags);
}

void gfx::Image::Copy(const Device& device, const vk::Buffer src_buffer) const {
  device.SubmitOneTimeTransferCommandBuffer([this, src_buffer](const auto command_buffer) {
    const vk::ImageSubresourceRange image_subresource_range{.aspectMask = aspect_flags_,
                                                            .levelCount = 1,
                                                            .layerCount = 1};

    TransitionImageLayout(command_buffer,
                          *image_,
                          image_subresource_range,
                          {vk::PipelineStageFlagBits::eHost, vk::PipelineStageFlagBits::eTransfer},
                          {vk::AccessFlagBits::eNone, vk::AccessFlagBits::eTransferWrite},
                          {vk::ImageLayout::eUndefined, vk::ImageLayout::eTransferDstOptimal});

    command_buffer.copyBufferToImage(
        src_buffer,
        *image_,
        vk::ImageLayout::eTransferDstOptimal,
        vk::BufferImageCopy{
            .imageSubresource = vk::ImageSubresourceLayers{.aspectMask = aspect_flags_, .layerCount = 1},
            .imageExtent = vk::Extent3D{.width = extent_.width, .height = extent_.height, .depth = 1}});

    TransitionImageLayout(command_buffer,
                          *image_,
                          image_subresource_range,
                          {vk::PipelineStageFlagBits::eTransfer, vk::PipelineStageFlagBits::eFragmentShader},
                          {vk::AccessFlagBits::eTransferWrite, vk::AccessFlagBits::eShaderRead},
                          {vk::ImageLayout::eTransferDstOptimal, vk::ImageLayout::eShaderReadOnlyOptimal});
  });
}
