#ifndef SRC_GRAPHICS_TEXTURE2D_H_
#define SRC_GRAPHICS_TEXTURE2D_H_

#include <filesystem>

#include <vulkan/vulkan.hpp>

#include "graphics/image.h"

namespace gfx {
class Device;

class Texture2d {
public:
  Texture2d(const Device& device, vk::Format format, const std::filesystem::path& filepath);

  [[nodiscard]] vk::ImageView image_view() const noexcept { return image_.image_view(); }
  [[nodiscard]] vk::Sampler sampler() const noexcept { return *sampler_; }

private:
  Image image_;
  vk::UniqueSampler sampler_;
};

}  // namespace gfx

#endif  //  SRC_GRAPHICS_TEXTURE2D_H_
