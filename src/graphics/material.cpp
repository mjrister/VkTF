#include "graphics/material.h"

#include <cstdint>
#include <ranges>

namespace {

vk::UniqueDescriptorSetLayout CreateDescriptorSetLayout(const vk::Device device) {
  static constexpr vk::DescriptorSetLayoutBinding kDescriptorSetLayoutBinding{
      .binding = 1,
      .descriptorType = vk::DescriptorType::eCombinedImageSampler,
      .descriptorCount = 2,
      .stageFlags = vk::ShaderStageFlagBits::eFragment};

  return device.createDescriptorSetLayoutUnique(
      vk::DescriptorSetLayoutCreateInfo{.bindingCount = 1, .pBindings = &kDescriptorSetLayoutBinding});
}

vk::UniqueDescriptorPool CreateDescriptorPool(const vk::Device device, const std::size_t size) {
  static constexpr vk::DescriptorPoolSize kDescriptorPoolSize{.type = vk::DescriptorType::eCombinedImageSampler,
                                                              .descriptorCount = 2};

  return device.createDescriptorPoolUnique(vk::DescriptorPoolCreateInfo{.maxSets = static_cast<std::uint32_t>(size),
                                                                        .poolSizeCount = 1,
                                                                        .pPoolSizes = &kDescriptorPoolSize});
}

std::vector<vk::DescriptorSet> AllocateDescriptorSets(const vk::Device device,
                                                      const vk::DescriptorSetLayout descriptor_set_layout,
                                                      const vk::DescriptorPool descriptor_pool,
                                                      const std::size_t size) {
  const std::vector descriptor_set_layouts(size, descriptor_set_layout);

  return device.allocateDescriptorSets(
      vk::DescriptorSetAllocateInfo{.descriptorPool = descriptor_pool,
                                    .descriptorSetCount = static_cast<std::uint32_t>(size),
                                    .pSetLayouts = descriptor_set_layouts.data()});
}

std::vector<gfx::Material> CreateMaterials(const vk::Device device,
                                           const vk::DescriptorSetLayout descriptor_set_layout,
                                           const vk::DescriptorPool descriptor_pool,
                                           const std::size_t size) {
  return AllocateDescriptorSets(device, descriptor_set_layout, descriptor_pool, size)  //
         | std::views::transform([](const auto descriptor_set) { return gfx::Material{descriptor_set}; })
         | std::ranges::to<std::vector>();
}

}  // namespace

void gfx::Material::UpdateDescriptorSet(const vk::Device device, Texture2d&& diffuse_map, Texture2d&& normal_map) {
  const std::array descriptor_image_info{
      vk::DescriptorImageInfo{.sampler = diffuse_map.sampler(),
                              .imageView = diffuse_map.image_view(),
                              .imageLayout = vk::ImageLayout::eShaderReadOnlyOptimal},
      vk::DescriptorImageInfo{.sampler = normal_map.sampler(),
                              .imageView = normal_map.image_view(),
                              .imageLayout = vk::ImageLayout::eShaderReadOnlyOptimal}};

  device.updateDescriptorSets(
      vk::WriteDescriptorSet{.dstSet = descriptor_set_,
                             .dstBinding = 1,
                             .dstArrayElement = 0,
                             .descriptorCount = static_cast<std::uint32_t>(descriptor_image_info.size()),
                             .descriptorType = vk::DescriptorType::eCombinedImageSampler,
                             .pImageInfo = descriptor_image_info.data()},
      nullptr);

  diffuse_map_ = std::move(diffuse_map);
  normal_map_ = std::move(normal_map);
}

gfx::Materials::Materials(const vk::Device device, const std::size_t size)
    : descriptor_set_layout_{CreateDescriptorSetLayout(device)},
      descriptor_pool_{CreateDescriptorPool(device, size)},
      materials_{CreateMaterials(device, *descriptor_set_layout_, *descriptor_pool_, size)} {}
