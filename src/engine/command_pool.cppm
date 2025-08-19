module;

#include <cstdint>
#include <vector>

#include <vulkan/vulkan.hpp>

export module command_pool;

namespace vktf {

/**
 * @brief An abstraction for a Vulkan Command Pool.
 * @details This class handles creating a command pool and allocating a fixed number of command buffers.
 * @see https://registry.khronos.org/vulkan/specs/latest/man/html/VkCommandPool.html VkCommandPool
 */
export class [[nodiscard]] CommandPool {
public:
  /** @brief The parameters for creating a @ref CommandPool. */
  struct [[nodiscard]] CreateInfo {
    /** @brief The bitmask indicating how the command pool should be created. */
    vk::CommandPoolCreateFlags command_pool_create_flags;

    /** @brief The queue family index that allocated command buffers must be submitted to. */
    std::uint32_t queue_family_index = 0;

    /** @brief The number of command buffers to allocate for the command pool. */
    std::uint32_t command_buffer_count = 0;
  };

  /**
   * @brief Creates a @ref CommandPool.
   * @param device The device for creating the command pool.
   * @param create_info @copybrief CommandPool::CreateInfo
   */
  CommandPool(vk::Device device, const CreateInfo& create_info);

  /** @brief Gets the underlying Vulkan command pool handle. */
  [[nodiscard]] vk::CommandPool operator*() const noexcept { return *command_pool_; }

  /** @brief Gets the command buffers allocated for this command pool. */
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
