#ifndef SRC_GRAPHICS_MATERIAL_H_
#define SRC_GRAPHICS_MATERIAL_H_

#include <cassert>
#include <utility>
#include <vector>

#include <vulkan/vulkan.hpp>

#include "graphics/texture2d.h"

namespace gfx {

class Material {
public:
  explicit Material(const vk::DescriptorSet& descriptor_set) noexcept : descriptor_set_{descriptor_set} {}

  [[nodiscard]] const vk::DescriptorSet& descriptor_set() const noexcept { return descriptor_set_; }

  void UpdateDescriptorSet(const vk::Device& device, Texture2d&& diffuse_map, Texture2d&& normal_map);

private:
  vk::DescriptorSet descriptor_set_;
  std::optional<Texture2d> diffuse_map_;
  std::optional<Texture2d> normal_map_;
};

class Materials {
public:
  Materials() noexcept = default;
  Materials(const vk::Device& device, std::size_t size);

  [[nodiscard]] auto&& operator[](this auto&& self, const std::size_t index) noexcept {
    assert(index < self.materials_.size());
    return std::forward<decltype(self)>(self).materials_[index];
  }

private:
  vk::UniqueDescriptorSetLayout descriptor_set_layout_;  // TODO(matthew-rister): avoid duplicating in gfx::Engine
  vk::UniqueDescriptorPool descriptor_pool_;
  std::vector<Material> materials_;
};

}  // namespace gfx

#endif  // SRC_GRAPHICS_MATERIAL_H_
