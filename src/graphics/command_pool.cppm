module;

#include <cstdint>
#include <vector>

#include <vulkan/vulkan.hpp>

export module command_pool;

namespace vktf {

export class [[nodiscard]] CommandPool {
public:
  struct [[nodiscard]] CreateInfo {
    vk::CommandPoolCreateFlags command_pool_create_flags;
    std::uint32_t queue_family_index = 0;
    std::uint32_t command_buffer_count = 0;
  };

  CommandPool(vk::Device device, const CreateInfo& create_info);

  [[nodiscard]] vk::CommandPool operator*() const noexcept { return *command_pool_; }

  [[nodiscard]] const std::vector<vk::CommandBuffer>& command_buffers() const noexcept { return command_buffers_; }

private:
  vk::UniqueCommandPool command_pool_;
  std::vector<vk::CommandBuffer> command_buffers_;  // command buffers are freed when the command pool is destroyed
};

}  // namespace vktf

module :private;

namespace vktf {

namespace {

vk::UniqueCommandPool CreateCommandPool(const vk::Device device, const CommandPool::CreateInfo& create_info) {
  const auto& [command_pool_create_flags, queue_family_index, _] = create_info;
  return device.createCommandPoolUnique(
      vk::CommandPoolCreateInfo{.flags = command_pool_create_flags, .queueFamilyIndex = queue_family_index});
}

std::vector<vk::CommandBuffer> AllocateCommandBuffers(const vk::Device device,
                                                      const vk::CommandPool command_pool,
                                                      const std::uint32_t command_buffer_count) {
  return device.allocateCommandBuffers(vk::CommandBufferAllocateInfo{.commandPool = command_pool,
                                                                     .level = vk::CommandBufferLevel::ePrimary,
                                                                     .commandBufferCount = command_buffer_count});
}

}  // namespace

CommandPool::CommandPool(const vk::Device device, const CreateInfo& create_info)
    : command_pool_{CreateCommandPool(device, create_info)},
      command_buffers_{AllocateCommandBuffers(device, *command_pool_, create_info.command_buffer_count)} {}

}  // namespace vktf
