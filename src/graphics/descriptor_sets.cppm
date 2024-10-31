module;

#include <cassert>
#include <cstdint>
#include <span>
#include <vector>

#include <vulkan/vulkan.hpp>

export module descriptor_sets;

namespace gfx {

export class DescriptorSets {
public:
  DescriptorSets(vk::Device device,
                 std::uint32_t max_descriptor_sets,
                 std::span<const vk::DescriptorPoolSize> descriptor_pool_sizes,
                 std::span<const vk::DescriptorSetLayoutBinding> descriptor_set_layout_bindings);

  [[nodiscard]] vk::DescriptorSet operator[](const std::size_t index) const noexcept {
    assert(index < descriptor_sets_.size());
    return descriptor_sets_[index];
  }

  [[nodiscard]] auto begin() const noexcept { return descriptor_sets_.cbegin(); }
  [[nodiscard]] auto end() const noexcept { return descriptor_sets_.cend(); }

  [[nodiscard]] vk::DescriptorSetLayout descriptor_set_layout() const noexcept { return *descriptor_set_layout_; }

private:
  vk::UniqueDescriptorPool descriptor_pool_;
  vk::UniqueDescriptorSetLayout descriptor_set_layout_;
  std::vector<vk::DescriptorSet> descriptor_sets_;  // descriptor sets are freed when the descriptor pool is destroyed
};

}  // namespace gfx

module :private;

namespace gfx {

DescriptorSets::DescriptorSets(const vk::Device device,
                               const std::uint32_t max_descriptor_sets,
                               const std::span<const vk::DescriptorPoolSize> descriptor_pool_sizes,
                               const std::span<const vk::DescriptorSetLayoutBinding> descriptor_set_layout_bindings)
    : descriptor_pool_{device.createDescriptorPoolUnique(
        vk::DescriptorPoolCreateInfo{.maxSets = max_descriptor_sets,
                                     .poolSizeCount = static_cast<std::uint32_t>(descriptor_pool_sizes.size()),
                                     .pPoolSizes = descriptor_pool_sizes.data()})},
      descriptor_set_layout_{device.createDescriptorSetLayoutUnique(vk::DescriptorSetLayoutCreateInfo{
          .bindingCount = static_cast<std::uint32_t>(descriptor_set_layout_bindings.size()),
          .pBindings = descriptor_set_layout_bindings.data()})} {
  const std::vector descriptor_set_layouts(max_descriptor_sets, *descriptor_set_layout_);

  descriptor_sets_ =
      device.allocateDescriptorSets(vk::DescriptorSetAllocateInfo{.descriptorPool = *descriptor_pool_,
                                                                  .descriptorSetCount = max_descriptor_sets,
                                                                  .pSetLayouts = descriptor_set_layouts.data()});
}

}  // namespace gfx
