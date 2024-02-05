#ifndef SRC_GRAPHICS_INCLUDE_GRAPHICS_DEVICE_H_
#define SRC_GRAPHICS_INCLUDE_GRAPHICS_DEVICE_H_

#include <concepts>

#include <vulkan/vulkan.hpp>

#include "graphics/physical_device.h"

namespace gfx {

class Device {
public:
  Device(vk::Instance instance, vk::SurfaceKHR surface);

  [[nodiscard]] vk::Device operator*() const noexcept { return *device_; }
  [[nodiscard]] const vk::Device* operator->() const noexcept { return &(*device_); }

  [[nodiscard]] const PhysicalDevice& physical_device() const noexcept { return physical_device_; }

  [[nodiscard]] vk::Queue graphics_queue() const noexcept { return graphics_queue_; }
  [[nodiscard]] vk::Queue present_queue() const noexcept { return present_queue_; }
  [[nodiscard]] vk::Queue transfer_queue() const noexcept { return transfer_queue_; }

  void SubmitOneTimeTransferCommandBuffer(std::invocable<const vk::CommandBuffer> auto&& command_sequence) const {
    const auto command_buffers =
        device_->allocateCommandBuffersUnique(vk::CommandBufferAllocateInfo{.commandPool = *transfer_command_pool_,
                                                                            .level = vk::CommandBufferLevel::ePrimary,
                                                                            .commandBufferCount = 1});
    const auto command_buffer = *command_buffers.front();

    command_buffer.begin(vk::CommandBufferBeginInfo{.flags = vk::CommandBufferUsageFlagBits::eOneTimeSubmit});
    command_sequence(command_buffer);
    command_buffer.end();

    transfer_queue_.submit(vk::SubmitInfo{.commandBufferCount = 1, .pCommandBuffers = &command_buffer});
    transfer_queue_.waitIdle();
  }

private:
  PhysicalDevice physical_device_;
  vk::UniqueDevice device_;
  vk::Queue graphics_queue_, present_queue_, transfer_queue_;
  vk::UniqueCommandPool transfer_command_pool_;
};

}  // namespace gfx

#endif  // SRC_GRAPHICS_INCLUDE_GRAPHICS_DEVICE_H_
