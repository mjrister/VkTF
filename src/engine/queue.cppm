module;

#include <cassert>
#include <cstdint>

#include <vulkan/vulkan.hpp>

export module queue;

namespace vktf {

export struct [[nodiscard]] QueueFamily {
  std::uint32_t index = vk::QueueFamilyIgnored;
  std::uint32_t queue_count = 0;
};

export struct [[nodiscard]] QueueFamilies {
  QueueFamily graphics_queue_family;
  QueueFamily present_queue_family;
};

export class [[nodiscard]] Queue {
public:
  struct [[nodiscard]] CreateInfo {
    QueueFamily queue_family;
    std::uint32_t queue_index = 0;
  };

  Queue(vk::Device device, const CreateInfo& create_info);

  [[nodiscard]] vk::Queue operator*() const noexcept { return queue_; }
  [[nodiscard]] const vk::Queue* operator->() const noexcept { return &queue_; }

  [[nodiscard]] std::uint32_t queue_family_index() const noexcept { return queue_family_index_; }

private:
  vk::Queue queue_;
  std::uint32_t queue_family_index_;
};

}  // namespace vktf

module :private;

namespace vktf {

namespace {

vk::Queue GetQueue(const vk::Device device, const QueueFamily queue_family, const std::uint32_t queue_index) {
  assert(queue_index < queue_family.queue_count);
  return device.getQueue(queue_family.index, queue_index);
}

}  // namespace

Queue::Queue(const vk::Device device, const CreateInfo& create_info)
    : queue_{GetQueue(device, create_info.queue_family, create_info.queue_index)},
      queue_family_index_{create_info.queue_family.index} {}

}  // namespace vktf
