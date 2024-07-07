#ifndef GRAPHICS_KTX_TEXTURE_H_
#define GRAPHICS_KTX_TEXTURE_H_

#include <filesystem>
#include <memory>
#include <unordered_set>

#include <ktx.h>
#include <vulkan/vulkan.hpp>

namespace gfx {

enum class ColorSpace { kLinear, kSrgb };

class KtxTexture {
public:
  KtxTexture(const std::filesystem::path& texture_filepath,
             ColorSpace color_space,
             const std::unordered_set<vk::Format>& supported_transcode_formats);

  [[nodiscard]] auto& operator*(this auto& self) noexcept { return *self.ktx_texture2_; }
  [[nodiscard]] auto* operator->(this auto& self) noexcept { return self.ktx_texture2_.get(); }

private:
  std::unique_ptr<ktxTexture2, void (*)(ktxTexture2*)> ktx_texture2_;
};

[[nodiscard]] std::unordered_set<vk::Format> GetSupportedTranscodeFormats(vk::PhysicalDevice physical_device);

}  // namespace gfx

#endif  // GRAPHICS_KTX_TEXTURE_H_
