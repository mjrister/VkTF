module;

#include <cstdint>
#include <utility>
#include <vector>

#include <vk_mem_alloc.h>
#include <vulkan/vulkan.hpp>

export module image;

import buffer;
import vma_allocator;

namespace vktf {

/**
 * @brief An abstraction for a Vulkan image.
 * @see https://registry.khronos.org/vulkan/specs/latest/man/html/VkImage.html VkImage
 */
export class [[nodiscard]] Image {
public:
  /** @brief The parameters for creating a @ref Image. */
  struct [[nodiscard]] CreateInfo {
    /** @brief The image format (e.g., BC7, ASTC4x4). */
    vk::Format format = vk::Format::eUndefined;

    /** @brief The image dimensions. */
    vk::Extent2D extent;

    /** @brief The number of levels of detail available for mipmap image sampling. */
    std::uint32_t mip_levels = 0;

    /** @brief The number of samples for multisample antialiasing. */
    vk::SampleCountFlagBits sample_count = vk::SampleCountFlagBits::e1;

    /** @brief The bit flags specifying how the image will be used. */
    vk::ImageUsageFlags usage_flags;

    /** @brief The bit flags specifying the image view aspect mask.  */
    vk::ImageAspectFlagBits aspect_mask = vk::ImageAspectFlagBits::eNone;

    /** @brief The parameters for allocating image memory. */
    const VmaAllocationCreateInfo& allocation_create_info;
  };

  /**
   * @brief Creates a @ref Image.
   * @param allocator The allocator for creating the image.
   * @param create_info @copybrief Image::CreateInfo
   */
  Image(const vma::Allocator& allocator, const CreateInfo& create_info);

  Image(const Image&) = delete;
  Image(Image&& image) noexcept { *this = std::move(image); }

  Image& operator=(const Image&) = delete;
  Image& operator=(Image&& image) noexcept;

  /** @brief Frees the underlying memory and destroys the image. */
  ~Image() noexcept;

  /** @brief Gets the image view. */
  [[nodiscard]] vk::ImageView image_view() const noexcept { return *image_view_; }

  /** @brief Gets the image format. */
  [[nodiscard]] vk::Format format() const noexcept { return format_; }

  /**
   * @brief Records copy commands to transfer data to this image.
   * @param src_buffer The source buffer to copy data from.
   * @param buffer_image_copies The subregions to copy corresponding to each mipmap in @p src_buffer.
   * @param command_buffer The command buffer for recording copy commands.
   * @warning The caller is responsible for submitting @p command_buffer to a Vulkan queue to begin execution.
   */
  void Copy(vk::Buffer src_buffer,
            const std::vector<vk::BufferImageCopy>& buffer_image_copies,
            vk::CommandBuffer command_buffer);

private:
  VmaAllocator allocator_ = nullptr;
  VmaAllocation allocation_ = nullptr;
  vk::Image image_;
  vk::UniqueImageView image_view_;
  vk::Format format_ = vk::Format::eUndefined;
  std::uint32_t mip_levels_ = 0;
  vk::ImageAspectFlagBits aspect_mask_ = vk::ImageAspectFlagBits::eNone;
};

}  // namespace vktf

module :private;

namespace vktf {

namespace {

std::pair<VmaAllocation, vk::Image> CreateImage(const vma::Allocator& allocator, const Image::CreateInfo& create_info) {
  const auto& [format, extent, mip_levels, sample_count, usage_flags, aspect_mask, allocation_create_info] =
      create_info;

  const VkImageCreateInfo image_create_info{
      .sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
      .imageType = VK_IMAGE_TYPE_2D,
      .format = static_cast<VkFormat>(format),
      .extent = VkExtent3D{.width = extent.width, .height = extent.height, .depth = 1},
      .mipLevels = mip_levels,
      .arrayLayers = 1,
      .samples = static_cast<VkSampleCountFlagBits>(sample_count),
      .usage = static_cast<VkImageUsageFlags>(usage_flags)};

  VkImage image = nullptr;
  VmaAllocation allocation = nullptr;
  const auto result =
      vmaCreateImage(*allocator, &image_create_info, &allocation_create_info, &image, &allocation, nullptr);
  vk::detail::resultCheck(static_cast<vk::Result>(result), "Image creation failed");

  return std::pair{allocation, image};
}

vk::UniqueImageView CreateImageView(const vk::Device device,
                                    const vk::Image image,
                                    const vk::Format format,
                                    const std::uint32_t mip_levels,
                                    const vk::ImageAspectFlagBits aspect_mask) {
  return device.createImageViewUnique(vk::ImageViewCreateInfo{
      .image = image,
      .viewType = vk::ImageViewType::e2D,
      .format = format,
      .subresourceRange =
          vk::ImageSubresourceRange{.aspectMask = aspect_mask, .levelCount = mip_levels, .layerCount = 1}});
}

void DestroyImage(const VmaAllocator allocator, const vk::Image image, const VmaAllocation allocation) noexcept {
  if (allocator != nullptr) {
    vmaDestroyImage(allocator, image, allocation);
  }
}

void TransitionImageLayout(const vk::Image image,
                           const vk::ImageSubresourceRange& image_subresource_range,
                           const std::pair<vk::PipelineStageFlags, vk::PipelineStageFlags>& stage_masks,
                           const std::pair<vk::AccessFlags, vk::AccessFlags>& access_masks,
                           const std::pair<vk::ImageLayout, vk::ImageLayout>& image_layouts,
                           const vk::CommandBuffer command_buffer) {
  const auto& [src_stage_mask, dst_stage_mask] = stage_masks;
  const auto& [src_access_mask, dst_access_mask] = access_masks;
  const auto& [old_layout, new_layout] = image_layouts;

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

Image::Image(const vma::Allocator& allocator, const CreateInfo& create_info)
    : allocator_{*allocator},
      format_{create_info.format},
      mip_levels_{create_info.mip_levels},
      aspect_mask_{create_info.aspect_mask} {
  std::tie(allocation_, image_) = CreateImage(allocator, create_info);
  image_view_ = CreateImageView(allocator.device(), image_, format_, mip_levels_, aspect_mask_);
}

Image& Image::operator=(Image&& image) noexcept {
  if (this != &image) {
    DestroyImage(allocator_, image_, allocation_);
    image_ = std::exchange(image.image_, nullptr);
    image_view_ = std::exchange(image.image_view_, vk::UniqueImageView{});
    format_ = std::exchange(image.format_, vk::Format::eUndefined);
    mip_levels_ = std::exchange(image.mip_levels_, 0);
    aspect_mask_ = std::exchange(image.aspect_mask_, vk::ImageAspectFlagBits::eNone);
    allocator_ = std::exchange(image.allocator_, nullptr);
    allocation_ = std::exchange(image.allocation_, nullptr);
  }
  return *this;
}

Image::~Image() noexcept { DestroyImage(allocator_, image_, allocation_); }

void Image::Copy(const vk::Buffer src_buffer,
                 const std::vector<vk::BufferImageCopy>& buffer_image_copies,
                 const vk::CommandBuffer command_buffer) {
  const vk::ImageSubresourceRange image_subresource_range{.aspectMask = aspect_mask_,
                                                          .levelCount = mip_levels_,
                                                          .layerCount = 1};

  TransitionImageLayout(image_,
                        image_subresource_range,
                        std::pair{vk::PipelineStageFlagBits::eTransfer, vk::PipelineStageFlagBits::eTransfer},
                        std::pair{vk::AccessFlagBits::eNone, vk::AccessFlagBits::eTransferWrite},
                        std::pair{vk::ImageLayout::eUndefined, vk::ImageLayout::eTransferDstOptimal},
                        command_buffer);

  command_buffer.copyBufferToImage(src_buffer, image_, vk::ImageLayout::eTransferDstOptimal, buffer_image_copies);

  TransitionImageLayout(image_,
                        image_subresource_range,
                        std::pair{vk::PipelineStageFlagBits::eTransfer, vk::PipelineStageFlagBits::eFragmentShader},
                        std::pair{vk::AccessFlagBits::eTransferWrite, vk::AccessFlagBits::eShaderRead},
                        std::pair{vk::ImageLayout::eTransferDstOptimal, vk::ImageLayout::eShaderReadOnlyOptimal},
                        command_buffer);
}

}  // namespace vktf
