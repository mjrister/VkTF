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

/** @brief A structure representing constant PBR material properties. */
export struct [[nodiscard]] MaterialProperties {
  /**
   * @brief The base color factor for a material.
   * @details If @ref base_color_texture is specified, this value acts as a linear multiplier with the sampled texel;
   *          otherwise, it defines the absolute base color value for the material.
   */
  glm::vec4 base_color_factor{0.0f};

  /**
   * @brief The metallic-roughness factor for a material.
   * @details The x-component is the metallic factor and the y-component is the roughness factor. If a material has a
   *          metallic-roughness texture, this value acts as a linear multiplier with the sampled texel; otherwise, it
   *          defines the absolute metallic-roughness value for the material.
   */
  glm::vec2 metallic_roughness_factor{0.0f};

  /** @brief The amount to scale sampled normals in the x/y directions. */
  float normal_scale = 0.0f;
};

/**
 * @brief A PBR material in host-visible memory.
 * @details This class handles creating host-visible staging buffers with image and properties data for a PBR material.
 */
export class [[nodiscard]] StagingMaterial {
public:
  /** @brief The parameters for creating a @ref StagingMaterial. */
  struct [[nodiscard]] CreateInfo {
    /** @brief @copybrief MaterialProperties */
    const MaterialProperties& material_properties;

    /** @brief The base color KTX texture. */
    const ktxTexture2& base_color_ktx_texture;

    /** @brief The metallic-roughness KTX texture. */
    const ktxTexture2& metallic_roughness_ktx_texture;

    /** @brief The normal map KTX texture. */
    const ktxTexture2& normal_ktx_texture;
  };

  /**
   * @brief Creates a @ref StagingMaterial.
   * @param allocator The allocator for creating staging buffers.
   * @param create_info @copybrief StagingMaterial::CreateInfo
   */
  StagingMaterial(const vma::Allocator& allocator, const CreateInfo& create_info);

  /** @brief Gets the properties staging buffer. */
  [[nodiscard]] const HostVisibleBuffer& properties_buffer() const noexcept { return properties_buffer_; }

  /** @brief Gets the base color staging texture. */
  [[nodiscard]] const StagingTexture& base_color_texture() const noexcept { return base_color_texture_; }

  /** @brief Gets the metallic-roughness staging texture. */
  [[nodiscard]] const StagingTexture& metallic_roughness_texture() const { return metallic_roughness_texture_; }

  /** @brief Gets the normal map staging texture. */
  [[nodiscard]] const StagingTexture& normal_texture() const noexcept { return normal_texture_; }

private:
  HostVisibleBuffer properties_buffer_;
  StagingTexture base_color_texture_;
  StagingTexture metallic_roughness_texture_;
  StagingTexture normal_texture_;
};

/**
 * @brief A PBR material in device-local memory.
 * @details This class handles creating device-local images and buffers for a PBR material, recording copy commands to
 *          transfer data from host-visible staging buffers, and assigning a descriptor set to bind material resources.
 */
export class [[nodiscard]] Material {
public:
  /** @brief The parameters for creating a @ref Material. */
  struct [[nodiscard]] CreateInfo {
    /** @brief The staging material to copy to device-local memory. */
    const StagingMaterial& staging_material;

    /** @brief The sampler for the base color texture. */
    vk::Sampler base_color_sampler;

    /** @brief The sampler for the metallic-roughness texture. */
    vk::Sampler metallic_roughness_sampler;

    /** @brief The sampler for the normal map texture. */
    vk::Sampler normal_sampler;

    /** @brief The descriptor set to update with this material's resources. */
    vk::DescriptorSet descriptor_set;
  };

  /**
   * @brief Creates a @ref Material.
   * @param allocator The allocator for creating device-local buffers and images.
   * @param command_buffer The command buffer for recording copy commands.
   * @param create_info @copybrief Material::CreateInfo
   * @warning The caller is responsible for submitting @p command_buffer to a Vulkan queue to begin execution.
   */
  Material(const vma::Allocator& allocator, vk::CommandBuffer command_buffer, const CreateInfo& create_info);

  /** @brief Gets the material descriptor set. */
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
