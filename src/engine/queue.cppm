module;

#include <cassert>
#include <cstdint>

#include <vulkan/vulkan.hpp>

export module queue;

namespace vktf {

/** \brief An abstraction for a Vulkan queue family. */
export struct [[nodiscard]] QueueFamily {
  /** \brief The index of this queue family when enumerating the queue family properties of a physical device. */
  std::uint32_t index = vk::QueueFamilyIgnored;

  /** \brief The number of queues supported by this queue family. */
  std::uint32_t queue_count = 0;
};

/**
 * \brief An abstraction that aggregates various queue families selected by the application.
 * \todo Add support for a dedicated transfer queue family.
 */
export struct [[nodiscard]] QueueFamilies {
  /** \brief A queue family that supports submitting commands that require graphics capabilities. */
  QueueFamily graphics_queue_family;

  /** \brief A queue family that can be used to present images to a Vulkan surface. */
  QueueFamily present_queue_family;
};

/** \brief An abstraction for a Vulkan queue. */
export class [[nodiscard]] Queue {
public:
  /** \brief Parameters used to create a Vulkan queue. */
  struct [[nodiscard]] CreateInfo {
    /** \brief The queue family the queue will belong to. */
    QueueFamily queue_family;

    /** \brief The queue index in its respective queue family. */
    std::uint32_t queue_index = 0;
  };

  /**
   * \brief Constructs a Vulkan queue.
   * \param device The logical device to get the Vulkan queue from.
   * \param create_info \copybrief Queue::CreateInfo
   */
  Queue(vk::Device device, const CreateInfo& create_info);

  /** \brief Gets the underlying vk::Queue handle. */
  [[nodiscard]] vk::Queue operator*() const noexcept { return queue_; }

  /** \brief Gets a pointer to the underlying vk::Queue handle. */
  [[nodiscard]] const vk::Queue* operator->() const noexcept { return &queue_; }

  /** \brief Gets the queue family index this queue belongs to. */
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
