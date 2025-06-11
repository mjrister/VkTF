module;

#include <array>

#include <ktx.h>
#include <glm/glm.hpp>
#include <vulkan/vulkan.hpp>

export module material;

import buffer;
import texture;
import vma_allocator;

namespace vktf::pbr_metallic_roughness {

export struct [[nodiscard]] MaterialProperties {
  glm::vec4 base_color_factor{0.0f};
  glm::vec2 metallic_roughness_factor{0.0f};
  float normal_scale = 0.0f;
};

export class [[nodiscard]] StagingMaterial {
public:
  struct [[nodiscard]] CreateInfo {
    const MaterialProperties& material_properties;
    const ktxTexture2& base_color_ktx_texture;
    const ktxTexture2& metallic_roughness_ktx_texture;
    const ktxTexture2& normal_ktx_texture;
  };

  StagingMaterial(const vma::Allocator& allocator, const CreateInfo& create_info);

  [[nodiscard]] const HostVisibleBuffer& properties_buffer() const noexcept { return properties_buffer_; }
  [[nodiscard]] const StagingTexture& base_color_texture() const noexcept { return base_color_texture_; }
  [[nodiscard]] const StagingTexture& metallic_roughness_texture() const { return metallic_roughness_texture_; }
  [[nodiscard]] const StagingTexture& normal_texture() const noexcept { return normal_texture_; }

private:
  HostVisibleBuffer properties_buffer_;
  StagingTexture base_color_texture_;
  StagingTexture metallic_roughness_texture_;
  StagingTexture normal_texture_;
};

export class [[nodiscard]] Material {
public:
  struct [[nodiscard]] CreateInfo {
    const StagingMaterial& staging_material;
    vk::Sampler base_color_sampler;
    vk::Sampler metallic_roughness_sampler;
    vk::Sampler normal_sampler;
    vk::DescriptorSet descriptor_set;
  };

  Material(const vma::Allocator& allocator, vk::CommandBuffer command_buffer, const CreateInfo& create_info);

  [[nodiscard]] vk::DescriptorSet descriptor_set() const noexcept { return descriptor_set_; }

private:
  Buffer properties_uniform_buffer_;
  Texture base_color_texture_;
  Texture metallic_roughness_texture_;
  Texture normal_texture_;
  vk::DescriptorSet descriptor_set_;
};

}  // namespace vktf::pbr_metallic_roughness

module :private;

namespace vktf::pbr_metallic_roughness {

namespace {

void UpdateDescriptorSet(const vk::Device device,
                         const vk::DescriptorSet descriptor_set,
                         const Buffer& properties_uniform_buffer,
                         const Texture& base_color_texture,
                         const Texture& metallic_roughness_texture,
                         const Texture& normal_texture) {
  const vk::DescriptorBufferInfo descriptor_buffer_info{.buffer = *properties_uniform_buffer, .range = vk::WholeSize};

  const std::array descriptor_image_info{
      vk::DescriptorImageInfo{.sampler = base_color_texture.sampler(),
                              .imageView = base_color_texture.image_view(),
                              .imageLayout = vk::ImageLayout::eShaderReadOnlyOptimal},
      vk::DescriptorImageInfo{.sampler = metallic_roughness_texture.sampler(),
                              .imageView = metallic_roughness_texture.image_view(),
                              .imageLayout = vk::ImageLayout::eShaderReadOnlyOptimal},
      vk::DescriptorImageInfo{.sampler = normal_texture.sampler(),
                              .imageView = normal_texture.image_view(),
                              .imageLayout = vk::ImageLayout::eShaderReadOnlyOptimal}};

  device.updateDescriptorSets(
      std::array{vk::WriteDescriptorSet{.dstSet = descriptor_set,
                                        .dstBinding = 0,
                                        .dstArrayElement = 0,
                                        .descriptorCount = 1,
                                        .descriptorType = vk::DescriptorType::eUniformBuffer,
                                        .pBufferInfo = &descriptor_buffer_info},
                 vk::WriteDescriptorSet{.dstSet = descriptor_set,
                                        .dstBinding = 1,
                                        .dstArrayElement = 0,
                                        .descriptorCount = static_cast<uint32_t>(descriptor_image_info.size()),
                                        .descriptorType = vk::DescriptorType::eCombinedImageSampler,
                                        .pImageInfo = descriptor_image_info.data()}},
      nullptr);
}

}  // namespace

StagingMaterial::StagingMaterial(const vma::Allocator& allocator, const CreateInfo& create_info)
    : properties_buffer_{CreateStagingBuffer<MaterialProperties>(allocator, create_info.material_properties)},
      base_color_texture_{allocator, create_info.base_color_ktx_texture},
      metallic_roughness_texture_{allocator, create_info.metallic_roughness_ktx_texture},
      normal_texture_{allocator, create_info.normal_ktx_texture} {}

Material::Material(const vma::Allocator& allocator,
                   const vk::CommandBuffer command_buffer,
                   const CreateInfo& create_info)
    : properties_uniform_buffer_{CreateDeviceLocalBuffer(allocator,
                                                         command_buffer,
                                                         create_info.staging_material.properties_buffer(),
                                                         vk::BufferUsageFlagBits::eUniformBuffer)},
      base_color_texture_{allocator,
                          command_buffer,
                          Texture::CreateInfo{.staging_texture = create_info.staging_material.base_color_texture(),
                                              .sampler = create_info.base_color_sampler}},
      metallic_roughness_texture_{
          allocator,
          command_buffer,
          Texture::CreateInfo{.staging_texture = create_info.staging_material.metallic_roughness_texture(),
                              .sampler = create_info.metallic_roughness_sampler}},
      normal_texture_{allocator,
                      command_buffer,
                      Texture::CreateInfo{.staging_texture = create_info.staging_material.normal_texture(),
                                          .sampler = create_info.normal_sampler}},
      descriptor_set_{create_info.descriptor_set} {
  UpdateDescriptorSet(allocator.device(),
                      descriptor_set_,
                      properties_uniform_buffer_,
                      base_color_texture_,
                      metallic_roughness_texture_,
                      normal_texture_);
}

}  // namespace vktf::pbr_metallic_roughness
