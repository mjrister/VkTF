#include "graphics/ktx_texture.h"

#include <array>
#include <cassert>
#include <format>
#include <iostream>
#include <print>
#include <ranges>
#include <span>
#include <stdexcept>
#include <string>
#include <utility>

#include <stb_image.h>
#include <vulkan/vulkan.hpp>

namespace {

struct Image {
  int width = 0;
  int height = 0;
  int channels = 0;
  std::unique_ptr<stbi_uc, decltype(&stbi_image_free)> data;
};

struct TranscodeFormat {
  vk::Format srgb_format = vk::Format::eUndefined;
  vk::Format unorm_format = vk::Format::eUndefined;
  ktx_transcode_fmt_e ktx_transcode_format = KTX_TTF_NOSELECTION;
};

ktxTexture* AsKtxTexture(ktxTexture2* const ktx_texture2) noexcept {
  // NOLINT(cppcoreguidelines-pro-type-cstyle-cast): macro definition requires c-style cast to the ktxTexture base class
  return ktxTexture(ktx_texture2);
}

void DestroyKtxTexture2(ktxTexture2* const ktx_texture2) noexcept { ktxTexture_Destroy(AsKtxTexture(ktx_texture2)); }
using UniqueKtxTexture2 = std::unique_ptr<ktxTexture2, decltype(&DestroyKtxTexture2)>;

constexpr TranscodeFormat kBc1TranscodeFormat{.srgb_format = vk::Format::eBc1RgbSrgbBlock,
                                              .unorm_format = vk::Format::eBc1RgbUnormBlock,
                                              .ktx_transcode_format = KTX_TTF_BC1_RGB};
constexpr TranscodeFormat kBc3TranscodeFormat{.srgb_format = vk::Format::eBc3SrgbBlock,
                                              .unorm_format = vk::Format::eBc3UnormBlock,
                                              .ktx_transcode_format = KTX_TTF_BC3_RGBA};
constexpr TranscodeFormat kBc7TranscodeFormat{.srgb_format = vk::Format::eBc7SrgbBlock,
                                              .unorm_format = vk::Format::eBc7UnormBlock,
                                              .ktx_transcode_format = KTX_TTF_BC7_RGBA};
constexpr TranscodeFormat kEtc1TranscodeFormat{.srgb_format = vk::Format::eEtc2R8G8B8SrgbBlock,
                                               .unorm_format = vk::Format::eEtc2R8G8B8UnormBlock,
                                               .ktx_transcode_format = KTX_TTF_ETC1_RGB};
constexpr TranscodeFormat kEtc2TranscodeFormat{.srgb_format = vk::Format::eEtc2R8G8B8A8SrgbBlock,
                                               .unorm_format = vk::Format::eEtc2R8G8B8A8UnormBlock,
                                               .ktx_transcode_format = KTX_TTF_ETC2_RGBA};
constexpr TranscodeFormat kAstc4x4TranscodeFormat{.srgb_format = vk::Format::eAstc4x4SrgbBlock,
                                                  .unorm_format = vk::Format::eAstc4x4UnormBlock,
                                                  .ktx_transcode_format = KTX_TTF_ASTC_4x4_RGBA};
constexpr TranscodeFormat kRgba32TranscodeFormat{.srgb_format = vk::Format::eR8G8B8A8Srgb,
                                                 .unorm_format = vk::Format::eR8G8B8A8Unorm,
                                                 .ktx_transcode_format = KTX_TTF_RGBA32};

ktx_transcode_fmt_e FindSupportedKtxTranscodeFormat(const std::span<const TranscodeFormat> target_transcode_formats,
                                                    const gfx::ColorSpace color_space,
                                                    const std::unordered_set<vk::Format>& supported_transcode_formats) {
  for (const auto [srgb_format, unorm_format, ktx_transcode_format] : target_transcode_formats) {
    if (const auto target_format = color_space == gfx::ColorSpace::kSrgb ? srgb_format : unorm_format;
        supported_transcode_formats.contains(target_format)) {
      return ktx_transcode_format;
    }
  }
  const auto [srgb_format, unorm_format, ktx_transcode_format] = kRgba32TranscodeFormat;
  if (const auto rgba32_transcode_format = color_space == gfx::ColorSpace::kSrgb ? srgb_format : unorm_format;
      supported_transcode_formats.contains(rgba32_transcode_format)) {
#ifndef NDEBUG
    std::println(std::clog,
                 "No supported texture compression format could be found. Decompressing to {}",
                 ktxTranscodeFormatString(ktx_transcode_format));
#endif
    return ktx_transcode_format;
  }
  throw std::runtime_error{"No supported KTX transcode formats could be found"};
}

ktx_transcode_fmt_e SelectKtxTranscodeFormat(ktxTexture2& ktx_texture2,
                                             const gfx::ColorSpace color_space,
                                             const std::unordered_set<vk::Format>& supported_transcode_formats) {
  const auto components = ktxTexture2_GetNumComponents(&ktx_texture2);
  if (components < 3 || components > 4) {  // TODO(matthew-rister): add support for one and two component images
    throw std::runtime_error{std::format("Unsupported image format with {} components", components)};
  }

  // format selection based on https://github.com/KhronosGroup/3D-Formats-Guidelines/blob/main/KTXDeveloperGuide.md
  switch (const auto has_alpha = components == 4; ktxTexture2_GetColorModel_e(&ktx_texture2)) {
    case KHR_DF_MODEL_ETC1S: {
      static constexpr std::array kEtc1sRgbTranscodeFormats{kEtc1TranscodeFormat,
                                                            kBc7TranscodeFormat,
                                                            kBc1TranscodeFormat};
      static constexpr std::array kEtc1sRgbaTranscodeFormats{kEtc2TranscodeFormat,
                                                             kBc7TranscodeFormat,
                                                             kBc3TranscodeFormat};
      const auto& etc1s_transcode_formats = has_alpha ? kEtc1sRgbaTranscodeFormats : kEtc1sRgbTranscodeFormats;
      return FindSupportedKtxTranscodeFormat(etc1s_transcode_formats, color_space, supported_transcode_formats);
    }
    case KHR_DF_MODEL_UASTC: {
      static constexpr std::array kUastcRgbTranscodeFormats{kAstc4x4TranscodeFormat,
                                                            kBc7TranscodeFormat,
                                                            kEtc1TranscodeFormat,
                                                            kBc1TranscodeFormat};
      static constexpr std::array kUastcRgbaTranscodeFormats{kAstc4x4TranscodeFormat,
                                                             kBc7TranscodeFormat,
                                                             kEtc2TranscodeFormat,
                                                             kBc3TranscodeFormat};
      const auto& uastc_transcode_formats = has_alpha ? kUastcRgbaTranscodeFormats : kUastcRgbTranscodeFormats;
      return FindSupportedKtxTranscodeFormat(uastc_transcode_formats, color_space, supported_transcode_formats);
    }
    default:
      std::unreachable();  // basis universal only supports UASTC/ETC1S transmission formats
  }
}

UniqueKtxTexture2 CreateKtxTexture2FromKtxFile(const std::filesystem::path& ktx_filepath,
                                               const gfx::ColorSpace color_space,
                                               const std::unordered_set<vk::Format>& supported_transcode_formats) {
  UniqueKtxTexture2 ktx_texture2{nullptr, nullptr};
  if (const auto ktx_error_code = ktxTexture2_CreateFromNamedFile(ktx_filepath.string().c_str(),
                                                                  KTX_TEXTURE_CREATE_CHECK_GLTF_BASISU_BIT,
                                                                  std::out_ptr(ktx_texture2, DestroyKtxTexture2));
      ktx_error_code != KTX_SUCCESS) {
    throw std::runtime_error{std::format("Failed to create KTX texture for {} with error {}",
                                         ktx_filepath.string(),
                                         ktxErrorString(ktx_error_code))};
  }

  if (ktxTexture2_NeedsTranscoding(ktx_texture2.get())) {
    const auto ktx_transcode_format = SelectKtxTranscodeFormat(*ktx_texture2, color_space, supported_transcode_formats);
    assert(ktx_transcode_format != KTX_TTF_NOSELECTION);
    if (const auto ktx_error_code = ktxTexture2_TranscodeBasis(ktx_texture2.get(), ktx_transcode_format, 0);
        ktx_error_code != KTX_SUCCESS) {
      throw std::runtime_error{std::format("Failed to transcode {} to {} with error {}",
                                           ktx_filepath.string(),
                                           ktxTranscodeFormatString(ktx_transcode_format),
                                           ktxErrorString(ktx_error_code))};
    }
  }

  return ktx_texture2;
}

Image LoadImage(const std::filesystem::path& image_filepath) {
  static constexpr auto kRequiredChannels = 4;  // require RGBA for improved device compatibility
  int width = 0;
  int height = 0;
  int channels = 0;

  std::unique_ptr<stbi_uc, decltype(&stbi_image_free)> data{
      stbi_load(image_filepath.string().c_str(), &width, &height, &channels, kRequiredChannels),
      stbi_image_free};
  if (data == nullptr) {
    throw std::runtime_error{
        std::format("Failed to load {} with error {}", image_filepath.string(), stbi_failure_reason())};
  }

#ifndef NDEBUG
  if (channels != kRequiredChannels) {
    std::println(std::clog,
                 "{} contains {} color channels but was requested to load with {}",
                 image_filepath.string(),
                 channels,
                 kRequiredChannels);
  }
#endif

  return Image{.width = width, .height = height, .channels = kRequiredChannels, .data = std::move(data)};
}

UniqueKtxTexture2 CreateKtxTexture2FromImageFile(const std::filesystem::path& image_filepath,
                                                 const gfx::ColorSpace color_space) {
  const auto& [width, height, channels, data] = LoadImage(image_filepath);

  // R8G8B8A8 format support for images with VK_FORMAT_FEATURE_SAMPLED_IMAGE_BIT is required by the Vulkan specification
  const auto format = color_space == gfx::ColorSpace::kSrgb ? vk::Format::eR8G8B8A8Srgb : vk::Format::eR8G8B8A8Unorm;
  ktxTextureCreateInfo ktx_texture_create_info{.vkFormat = static_cast<ktx_uint32_t>(format),
                                               .baseWidth = static_cast<ktx_uint32_t>(width),
                                               .baseHeight = static_cast<ktx_uint32_t>(height),
                                               .baseDepth = 1,
                                               .numDimensions = 2,
                                               .numLevels = 1,
                                               .numLayers = 1,
                                               .numFaces = 1};

  UniqueKtxTexture2 ktx_texture2{nullptr, nullptr};
  if (const auto result = ktxTexture2_Create(&ktx_texture_create_info,
                                             KTX_TEXTURE_CREATE_ALLOC_STORAGE,
                                             std::out_ptr(ktx_texture2, DestroyKtxTexture2));
      result != KTX_SUCCESS) {
    throw std::runtime_error{std::format("Failed to create KTX texture for {} with error {}",
                                         image_filepath.string(),
                                         ktxErrorString(result))};
  }

  // TODO(matthew-rister): implement runtime mipmap generation for raw images
  const auto size_bytes = static_cast<ktx_size_t>(width) * height * channels;
  if (const auto result = ktxTexture_SetImageFromMemory(
          AsKtxTexture(ktx_texture2.get()),
          0,
          0,
          KTX_FACESLICE_WHOLE_LEVEL,
          data.get(),  // image data is copied so ownership does not need to be transferred
          size_bytes);
      result != KTX_SUCCESS) {
    throw std::runtime_error{std::format("Failed to set KTX texture image for {} with error {}",
                                         image_filepath.string(),
                                         ktxErrorString(result))};
  }

  return ktx_texture2;
}

}  // namespace

namespace gfx {

KtxTexture::KtxTexture(const std::filesystem::path& texture_filepath,
                       const ColorSpace color_space,
                       const std::unordered_set<vk::Format>& supported_transcode_formats)
    : ktx_texture2_{texture_filepath.extension() == ".ktx2"
                        ? CreateKtxTexture2FromKtxFile(texture_filepath, color_space, supported_transcode_formats)
                        : CreateKtxTexture2FromImageFile(texture_filepath, color_space)} {}

std::unordered_set<vk::Format> GetSupportedTranscodeFormats(const vk::PhysicalDevice physical_device) {
  static constexpr std::array kTargetTranscodeFormats{
      // clang-format off
     kBc1TranscodeFormat.srgb_format, kBc1TranscodeFormat.unorm_format,
     kBc3TranscodeFormat.srgb_format, kBc3TranscodeFormat.unorm_format,
     kBc7TranscodeFormat.srgb_format, kBc7TranscodeFormat.unorm_format,
     kEtc1TranscodeFormat.srgb_format, kEtc1TranscodeFormat.unorm_format,
     kEtc2TranscodeFormat.srgb_format, kEtc2TranscodeFormat.unorm_format,
     kAstc4x4TranscodeFormat.srgb_format, kAstc4x4TranscodeFormat.unorm_format,
     kRgba32TranscodeFormat.srgb_format, kRgba32TranscodeFormat.unorm_format
      // clang-format on
  };
  return kTargetTranscodeFormats  //
         | std::views::filter([physical_device](const auto transcode_format) {
             const auto format_properties = physical_device.getFormatProperties(transcode_format);
             return static_cast<bool>(format_properties.optimalTilingFeatures
                                      & vk::FormatFeatureFlagBits::eSampledImage);
           })
         | std::ranges::to<std::unordered_set>();
}

}  // namespace gfx
