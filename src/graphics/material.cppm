module;

#include <vulkan/vulkan.hpp>

export module material;

import buffer;
import texture;

namespace vktf {

// TODO: design an API for creating and using materials
export struct Material {
  Buffer properties_buffer;
  Texture base_color_texture;
  Texture metallic_roughness_texture;
  Texture normal_texture;
  vk::DescriptorSet descriptor_set;
};

}  // namespace vktf
