module;

#include <vulkan/vulkan.hpp>

export module material;

import buffer;
import texture;

namespace vktf {

// TODO: design an API for creating and using materials
export struct Material {
  Texture base_color_texture;
  Texture metallic_roughness_texture;
  Texture normal_texture;
  Buffer properties_buffer;
  vk::DescriptorSet descriptor_set;
};

}  // namespace vktf
