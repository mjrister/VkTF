module;

#include <filesystem>
#include <format>
#include <memory>
#include <ranges>
#include <stdexcept>
#include <utility>
#include <vector>

#include <ktx.h>
#include <vulkan/vulkan.hpp>

export module ktx_texture;

import log;

namespace vktf::ktx {

export using UniqueKtxTexture2 = std::unique_ptr<ktxTexture2, void (*)(ktxTexture2*) noexcept>;

export [[nodiscard]] UniqueKtxTexture2 Load(const std::filesystem::path& ktx_filepath,
                                            const vk::PhysicalDeviceFeatures& physical_device_features,
                                            Log& log);

export [[nodiscard]] std::vector<vk::BufferImageCopy> GetBufferImageCopies(const ktxTexture2& ktx_texture2);

}  // namespace vktf::ktx

module :private;

namespace vktf::ktx {

namespace {

using Severity = Log::Severity;

void DestroyKtxTexture2(ktxTexture2* const ktx_texture2) noexcept {
  auto* const ktx_texture = ktxTexture(ktx_texture2);
  ktxTexture_Destroy(ktx_texture);
}

ktx_transcode_fmt_e SelectKtxTranscodeFormat(ktxTexture2& ktx_texture2,
                                             const vk::PhysicalDeviceFeatures& physical_device_features,
                                             [[maybe_unused]] Log& log) {
  const auto components = ktxTexture2_GetNumComponents(&ktx_texture2);
  if (components != 3 && components != 4) {  // TODO: add support for one and two component images
    throw std::runtime_error{std::format("Unsupported image format with {} components", components)};
  }

  const auto has_etc2_support = physical_device_features.textureCompressionETC2 == vk::True;
  const auto has_bc_support = physical_device_features.textureCompressionBC == vk::True;
  const auto has_astc_support = physical_device_features.textureCompressionASTC_LDR == vk::True;

  // format selection based on https://github.com/KhronosGroup/3D-Formats-Guidelines/blob/main/KTXDeveloperGuide.md
  switch (const auto has_alpha = components == 4; ktxTexture2_GetColorModel_e(&ktx_texture2)) {
    case KHR_DF_MODEL_ETC1S:
      if (has_etc2_support) return has_alpha ? KTX_TTF_ETC2_RGBA : KTX_TTF_ETC1_RGB;
      if (has_bc_support) return KTX_TTF_BC7_RGBA;
      if (has_astc_support) return KTX_TTF_ASTC_4x4_RGBA;
      break;
    case KHR_DF_MODEL_UASTC:
      if (has_astc_support) return KTX_TTF_ASTC_4x4_RGBA;
      if (has_bc_support) return KTX_TTF_BC7_RGBA;
      if (has_etc2_support) return has_alpha ? KTX_TTF_ETC2_RGBA : KTX_TTF_ETC1_RGB;
      break;
    default:
      std::unreachable();  // basis universal only supports ETC1S/UASTC transmission formats
  }

#ifndef NDEBUG
  log(Severity::kInfo) << std::format("No supported texture compression format could be found. Decompressing to {}",
                                      ktxTranscodeFormatString(KTX_TTF_RGBA32));
#endif

  return KTX_TTF_RGBA32;  // fallback to RGBA32 if no supported transcode format is found
}

}  // namespace

UniqueKtxTexture2 Load(const std::filesystem::path& ktx_filepath,
                       const vk::PhysicalDeviceFeatures& physical_device_features,
                       Log& log) {
  UniqueKtxTexture2 ktx_texture2{nullptr, DestroyKtxTexture2};

  if (const auto ktx_error_code = ktxTexture2_CreateFromNamedFile(ktx_filepath.string().c_str(),
                                                                  KTX_TEXTURE_CREATE_CHECK_GLTF_BASISU_BIT,
                                                                  std::out_ptr(ktx_texture2));
      ktx_error_code != KTX_SUCCESS) {
    throw std::runtime_error{std::format("Failed to create KTX texture for {} with error {}",
                                         ktx_filepath.string(),
                                         ktxErrorString(ktx_error_code))};
  }

  if (ktxTexture2_NeedsTranscoding(ktx_texture2.get())) {
    const auto ktx_transcode_format = SelectKtxTranscodeFormat(*ktx_texture2, physical_device_features, log);

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

std::vector<vk::BufferImageCopy> GetBufferImageCopies(const ktxTexture2& ktx_texture2) {
  return std::views::iota(0u, ktx_texture2.numLevels)
         | std::views::transform([ktx_texture = ktxTexture(&ktx_texture2)](const auto mip_level) {
             ktx_size_t image_offset = 0;
             if (const auto ktx_error_code = ktxTexture_GetImageOffset(ktx_texture, mip_level, 0, 0, &image_offset);
                 ktx_error_code != KTX_SUCCESS) {
               throw std::runtime_error{std::format("Failed to get image offset for mip level {} with error {}",
                                                    mip_level,
                                                    ktxErrorString(ktx_error_code))};
             }
             return vk::BufferImageCopy{
                 .bufferOffset = static_cast<vk::DeviceSize>(image_offset),
                 .imageSubresource = vk::ImageSubresourceLayers{.aspectMask = vk::ImageAspectFlagBits::eColor,
                                                                .mipLevel = mip_level,
                                                                .layerCount = 1},
                 .imageExtent = vk::Extent3D{.width = std::max(ktx_texture->baseWidth >> mip_level, 1u),
                                             .height = std::max(ktx_texture->baseHeight >> mip_level, 1u),
                                             .depth = 1}};
           })
         | std::ranges::to<std::vector>();
}

}  // namespace vktf::ktx
