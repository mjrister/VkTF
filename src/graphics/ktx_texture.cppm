module;

#include <array>
#include <cassert>
#include <cstdint>
#include <filesystem>
#include <format>
#include <iostream>
#include <memory>
#include <print>
#include <ranges>
#include <stdexcept>
#include <string>
#include <unordered_set>
#include <utility>

#include <ktx.h>
#include <stb_image.h>
#include <vulkan/vulkan.hpp>

export module ktx_texture;

namespace gfx {

export enum class ColorSpace : std::uint8_t { kLinear, kSrgb };

export class KtxTexture {
public:
  KtxTexture(const std::filesystem::path& texture_filepath, ColorSpace color_space, vk::PhysicalDevice physical_device);

  [[nodiscard]] const ktxTexture2& operator*() const noexcept { return *ktx_texture2_; }
  [[nodiscard]] const ktxTexture2* operator->() const noexcept { return ktx_texture2_.get(); }

private:
  std::unique_ptr<ktxTexture2, void (*)(ktxTexture2*)> ktx_texture2_;
};

}  // namespace gfx

module :private;

namespace {

struct StbImage {
  int width = 0;
  int height = 0;
  int channels = 0;
  std::unique_ptr<stbi_uc, decltype(&stbi_image_free)> data;
};

struct TranscodeTarget {
  vk::Format srgb_format = vk::Format::eUndefined;
  vk::Format unorm_format = vk::Format::eUndefined;
  ktx_transcode_fmt_e ktx_transcode_format = KTX_TTF_NOSELECTION;
};

ktxTexture* AsKtxTexture(ktxTexture2* const ktx_texture2) {
  // NOLINT(cppcoreguidelines-pro-type-cstyle-cast): macro definition requires c-style cast to the ktxTexture base class
  return ktxTexture(ktx_texture2);
}

void DestroyKtxTexture2(ktxTexture2* const ktx_texture2) noexcept { ktxTexture_Destroy(AsKtxTexture(ktx_texture2)); }
using KtxTexture2 = std::unique_ptr<ktxTexture2, decltype(&DestroyKtxTexture2)>;

constexpr TranscodeTarget kBc1TranscodeTarget{.srgb_format = vk::Format::eBc1RgbSrgbBlock,
                                              .unorm_format = vk::Format::eBc1RgbUnormBlock,
                                              .ktx_transcode_format = KTX_TTF_BC1_RGB};
constexpr TranscodeTarget kBc3TranscodeTarget{.srgb_format = vk::Format::eBc3SrgbBlock,
                                              .unorm_format = vk::Format::eBc3UnormBlock,
                                              .ktx_transcode_format = KTX_TTF_BC3_RGBA};
constexpr TranscodeTarget kBc7TranscodeTarget{.srgb_format = vk::Format::eBc7SrgbBlock,
                                              .unorm_format = vk::Format::eBc7UnormBlock,
                                              .ktx_transcode_format = KTX_TTF_BC7_RGBA};
constexpr TranscodeTarget kEtc1TranscodeTarget{.srgb_format = vk::Format::eEtc2R8G8B8SrgbBlock,
                                               .unorm_format = vk::Format::eEtc2R8G8B8UnormBlock,
                                               .ktx_transcode_format = KTX_TTF_ETC1_RGB};
constexpr TranscodeTarget kEtc2TranscodeTarget{.srgb_format = vk::Format::eEtc2R8G8B8A8SrgbBlock,
                                               .unorm_format = vk::Format::eEtc2R8G8B8A8UnormBlock,
                                               .ktx_transcode_format = KTX_TTF_ETC2_RGBA};
constexpr TranscodeTarget kAstc4x4TranscodeTarget{.srgb_format = vk::Format::eAstc4x4SrgbBlock,
                                                  .unorm_format = vk::Format::eAstc4x4UnormBlock,
                                                  .ktx_transcode_format = KTX_TTF_ASTC_4x4_RGBA};
constexpr TranscodeTarget kRgba32TranscodeTarget{.srgb_format = vk::Format::eR8G8B8A8Srgb,
                                                 .unorm_format = vk::Format::eR8G8B8A8Unorm,
                                                 .ktx_transcode_format = KTX_TTF_RGBA32};

bool IsFormatSupported(const vk::PhysicalDevice physical_device, const vk::Format format) {
  const auto format_properties = physical_device.getFormatProperties(format);
  return static_cast<bool>(format_properties.optimalTilingFeatures & vk::FormatFeatureFlagBits::eSampledImage);
}

std::unordered_set<vk::Format> GetSupportedTranscodeFormats(const vk::PhysicalDevice physical_device) {
  static constexpr auto kAllTranscodeFormats = {
      // clang-format off
     kBc1TranscodeTarget.srgb_format, kBc1TranscodeTarget.unorm_format,
     kBc3TranscodeTarget.srgb_format, kBc3TranscodeTarget.unorm_format,
     kBc7TranscodeTarget.srgb_format, kBc7TranscodeTarget.unorm_format,
     kEtc1TranscodeTarget.srgb_format, kEtc1TranscodeTarget.unorm_format,
     kEtc2TranscodeTarget.srgb_format, kEtc2TranscodeTarget.unorm_format,
     kAstc4x4TranscodeTarget.srgb_format, kAstc4x4TranscodeTarget.unorm_format
      // clang-format on
  };
  return kAllTranscodeFormats  //
         | std::views::filter([physical_device](const auto transcode_format) {
             return IsFormatSupported(physical_device, transcode_format);
           })
         | std::ranges::to<std::unordered_set>();
}

template <std::size_t N>
ktx_transcode_fmt_e FindSupportedKtxTranscodeFormat(const std::array<TranscodeTarget, N>& transcode_targets,
                                                    const gfx::ColorSpace color_space,
                                                    const vk::PhysicalDevice physical_device) {
  static const auto kSupportTranscodeFormats = GetSupportedTranscodeFormats(physical_device);
  for (const auto& [srgb_format, unorm_format, ktx_transcode_format] : transcode_targets) {
    if (const auto transcode_format = color_space == gfx::ColorSpace::kLinear ? unorm_format : srgb_format;
        kSupportTranscodeFormats.contains(transcode_format)) {
      return ktx_transcode_format;
    }
  }
  static constexpr auto kRgba32KtxTranscodeFormat = kRgba32TranscodeTarget.ktx_transcode_format;
#ifndef NDEBUG
  std::println(std::clog,
               "No supported texture compression format could be found. Decompressing to {}",
               ktxTranscodeFormatString(kRgba32KtxTranscodeFormat));
#endif
  return kRgba32KtxTranscodeFormat;  // fallback to RGBA32 if no supported transcode format is found
}

ktx_transcode_fmt_e SelectKtxTranscodeFormat(ktxTexture2& ktx_texture2,
                                             const gfx::ColorSpace color_space,
                                             const vk::PhysicalDevice physical_device) {
  const auto components = ktxTexture2_GetNumComponents(&ktx_texture2);
  if (components != 3 && components != 4) {  // TODO(matthew-rister): add support for one and two component images
    throw std::runtime_error{std::format("Unsupported image format with {} components", components)};
  }

  // format selection based on https://github.com/KhronosGroup/3D-Formats-Guidelines/blob/main/KTXDeveloperGuide.md
  switch (const auto has_alpha = components == 4; ktxTexture2_GetColorModel_e(&ktx_texture2)) {
    case KHR_DF_MODEL_ETC1S: {
      static constexpr std::array kEtc1sRgbTranscodeTargets{kEtc1TranscodeTarget,
                                                            kBc7TranscodeTarget,
                                                            kBc1TranscodeTarget};
      static constexpr std::array kEtc1sRgbaTranscodeTargets{kEtc2TranscodeTarget,
                                                             kBc7TranscodeTarget,
                                                             kBc3TranscodeTarget};
      const auto& etc1s_transcode_targets = has_alpha ? kEtc1sRgbaTranscodeTargets : kEtc1sRgbTranscodeTargets;
      return FindSupportedKtxTranscodeFormat(etc1s_transcode_targets, color_space, physical_device);
    }
    case KHR_DF_MODEL_UASTC: {
      static constexpr std::array kUastcRgbTranscodeTargets{kAstc4x4TranscodeTarget,
                                                            kBc7TranscodeTarget,
                                                            kEtc1TranscodeTarget,
                                                            kBc1TranscodeTarget};
      static constexpr std::array kUastcRgbaTranscodeTargets{kAstc4x4TranscodeTarget,
                                                             kBc7TranscodeTarget,
                                                             kEtc2TranscodeTarget,
                                                             kBc3TranscodeTarget};
      const auto& uastc_transcode_targets = has_alpha ? kUastcRgbaTranscodeTargets : kUastcRgbTranscodeTargets;
      return FindSupportedKtxTranscodeFormat(uastc_transcode_targets, color_space, physical_device);
    }
    default:
      std::unreachable();  // basis universal only supports ETC1S/UASTC transmission formats
  }
}

KtxTexture2 CreateKtxTexture2FromKtxFile(const std::filesystem::path& ktx_filepath,
                                         const gfx::ColorSpace color_space,
                                         const vk::PhysicalDevice physical_device) {
  KtxTexture2 ktx_texture2{nullptr, DestroyKtxTexture2};
  if (const auto ktx_error_code = ktxTexture2_CreateFromNamedFile(ktx_filepath.string().c_str(),
                                                                  KTX_TEXTURE_CREATE_CHECK_GLTF_BASISU_BIT,
                                                                  std::out_ptr(ktx_texture2));
      ktx_error_code != KTX_SUCCESS) {
    throw std::runtime_error{std::format("Failed to create KTX texture for {} with error {}",
                                         ktx_filepath.string(),
                                         ktxErrorString(ktx_error_code))};
  }

  if (ktxTexture2_NeedsTranscoding(ktx_texture2.get())) {
    const auto ktx_transcode_format = SelectKtxTranscodeFormat(*ktx_texture2, color_space, physical_device);
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

StbImage Load(const std::filesystem::path& image_filepath) {
  static constexpr auto kRequiredChannels = 4;  // require RGBA to guarantee Vulkan format support
  auto width = 0;
  auto height = 0;
  auto channels = 0;

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
                 "Converting {} from {} to {} color channels",
                 image_filepath.string(),
                 channels,
                 kRequiredChannels);
  }
#endif

  return StbImage{.width = width, .height = height, .channels = kRequiredChannels, .data = std::move(data)};
}

KtxTexture2 CreateKtxTexture2FromImageFile(const std::filesystem::path& image_filepath,
                                           const gfx::ColorSpace color_space) {
  const auto& [width, height, channels, data] = Load(image_filepath);
  static_assert(sizeof(decltype(data)::element_type) == 1, "8-bit image data is required");
  const auto data_size_bytes = static_cast<ktx_size_t>(width) * height * channels;

  const auto& [rgba32_srgb_format, rgba32_unorm_format, _] = kRgba32TranscodeTarget;
  const auto rgba32_format = color_space == gfx::ColorSpace::kLinear ? rgba32_unorm_format : rgba32_srgb_format;

  KtxTexture2 ktx_texture2{nullptr, DestroyKtxTexture2};
  ktxTextureCreateInfo ktx_texture_create_info{.vkFormat = static_cast<ktx_uint32_t>(rgba32_format),
                                               .baseWidth = static_cast<ktx_uint32_t>(width),
                                               .baseHeight = static_cast<ktx_uint32_t>(height),
                                               .baseDepth = 1,
                                               .numDimensions = 2,
                                               .numLevels = 1,
                                               .numLayers = 1,
                                               .numFaces = 1,
                                               // TODO(matthew-rister): implement runtime mipmap generation
                                               .generateMipmaps = true};
  if (const auto result =
          ktxTexture2_Create(&ktx_texture_create_info, KTX_TEXTURE_CREATE_ALLOC_STORAGE, std::out_ptr(ktx_texture2));
      result != KTX_SUCCESS) {
    throw std::runtime_error{std::format("Failed to create KTX texture for {} with error {}",
                                         image_filepath.string(),
                                         ktxErrorString(result))};
  }

  if (const auto result =
          ktxTexture_SetImageFromMemory(AsKtxTexture(ktx_texture2.get()),
                                        0,
                                        0,
                                        KTX_FACESLICE_WHOLE_LEVEL,
                                        data.get(),  // ownership retained because the underlying pointer data is copied
                                        data_size_bytes);
      result != KTX_SUCCESS) {
    throw std::runtime_error{std::format("Failed to set KTX texture image for {} with error {}",
                                         image_filepath.string(),
                                         ktxErrorString(result))};
  }

  return ktx_texture2;
}

KtxTexture2 CreateKtxTexture2(const std::filesystem::path& texture_filepath,
                              const gfx::ColorSpace color_space,
                              const vk::PhysicalDevice physical_device) {
#ifndef NDEBUG
  // R8G8B8A8 format support for images with VK_FORMAT_FEATURE_SAMPLED_IMAGE_BIT is required by the Vulkan specification
  const auto& [rgba32_srgb_format, rgba32_unorm_format, _] = kRgba32TranscodeTarget;
  assert(IsFormatSupported(physical_device, rgba32_srgb_format));
  assert(IsFormatSupported(physical_device, rgba32_unorm_format));
#endif
  return texture_filepath.extension() == ".ktx2"
             ? CreateKtxTexture2FromKtxFile(texture_filepath, color_space, physical_device)
             : CreateKtxTexture2FromImageFile(texture_filepath, color_space);
}

}  // namespace

namespace gfx {

// TODO(matthew-rister): validate image dimensions are a power-of-two
KtxTexture::KtxTexture(const std::filesystem::path& texture_filepath,
                       const ColorSpace color_space,
                       const vk::PhysicalDevice physical_device)
    : ktx_texture2_{CreateKtxTexture2(texture_filepath, color_space, physical_device)} {}

}  // namespace gfx
