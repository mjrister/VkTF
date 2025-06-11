module;

#include <cstdint>
#include <vector>

#include <vulkan/vulkan.hpp>

export module descriptor_pool;

namespace vktf {

export class [[nodiscard]] DescriptorPool {
public:
  struct [[nodiscard]] CreateInfo {
    const std::vector<vk::DescriptorPoolSize>& descriptor_pool_sizes;
    vk::DescriptorSetLayout descriptor_set_layout;
    std::uint32_t descriptor_set_count = 0;
  };

  DescriptorPool(vk::Device device, const CreateInfo& create_info);

  [[nodiscard]] const std::vector<vk::DescriptorSet>& descriptor_sets() const noexcept { return descriptor_sets_; }

private:
  vk::UniqueDescriptorPool descriptor_pool_;
  std::vector<vk::DescriptorSet> descriptor_sets_;  // descriptor sets are freed when the descriptor pool is destroyed
};

}  // namespace vktf

module :private;

namespace vktf {

namespace {

vk::UniqueDescriptorPool CreateDescriptorPool(const vk::Device device, const DescriptorPool::CreateInfo& create_info) {
  const auto& [descriptor_pool_sizes, _, descriptor_set_count] = create_info;
  return device.createDescriptorPoolUnique(
      vk::DescriptorPoolCreateInfo{.maxSets = descriptor_set_count,
                                   .poolSizeCount = static_cast<std::uint32_t>(descriptor_pool_sizes.size()),
                                   .pPoolSizes = descriptor_pool_sizes.data()});
}

std::vector<vk::DescriptorSet> AllocateDescriptorSets(const vk::Device device,
                                                      const vk::DescriptorPool descriptor_pool,
                                                      const vk::DescriptorSetLayout descriptor_set_layout,
                                                      const std::uint32_t descriptor_set_count) {
  const std::vector descriptor_set_layouts(descriptor_set_count, descriptor_set_layout);

  return device.allocateDescriptorSets(vk::DescriptorSetAllocateInfo{.descriptorPool = descriptor_pool,
                                                                     .descriptorSetCount = descriptor_set_count,
                                                                     .pSetLayouts = descriptor_set_layouts.data()});
}

}  // namespace

DescriptorPool::DescriptorPool(const vk::Device device, const CreateInfo& create_info)
    : descriptor_pool_{CreateDescriptorPool(device, create_info)},
      descriptor_sets_{AllocateDescriptorSets(device,
                                              *descriptor_pool_,
                                              create_info.descriptor_set_layout,
                                              create_info.descriptor_set_count)} {}

}  // namespace vktf
