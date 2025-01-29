module;

#include <cstdint>
#include <vector>

#include <vulkan/vulkan.hpp>

export module command_pool;

namespace vktf {

export class CommandPool {
public:
  CommandPool(vk::Device device,
              vk::CommandPoolCreateFlags command_pool_create_flags,
              std::uint32_t queue_family_index,
              std::uint32_t command_buffer_count);

  [[nodiscard]] const auto& command_buffers() const noexcept { return command_buffers_; }

private:
  vk::UniqueCommandPool command_pool_;
  std::vector<vk::UniqueCommandBuffer> command_buffers_;
};

}  // namespace vktf

module :private;

namespace vktf {

CommandPool::CommandPool(const vk::Device device,
                         const vk::CommandPoolCreateFlags command_pool_create_flags,
                         const std::uint32_t queue_family_index,
                         const std::uint32_t command_buffer_count)
    : command_pool_{device.createCommandPoolUnique(
          vk::CommandPoolCreateInfo{.flags = command_pool_create_flags, .queueFamilyIndex = queue_family_index})},
      command_buffers_{device.allocateCommandBuffersUnique(
          vk::CommandBufferAllocateInfo{.commandPool = *command_pool_,
                                        .level = vk::CommandBufferLevel::ePrimary,
                                        .commandBufferCount = command_buffer_count})} {}

}  // namespace vktf
