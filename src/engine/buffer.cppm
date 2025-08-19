module;

#include <cassert>
#include <cstring>
#include <utility>

#include <vk_mem_alloc.h>
#include <vulkan/vulkan.hpp>

export module buffer;

import data_view;
import vma_allocator;

namespace vktf {

/**
 * @brief An abstraction for a Vulkan buffer.
 * @see https://registry.khronos.org/vulkan/specs/latest/man/html/VkBuffer.html VkBuffer
 */
export class [[nodiscard]] Buffer {
public:
  /** @brief The parameters for creating a @ref Buffer. */
  struct [[nodiscard]] CreateInfo {
    /** @brief The size of the buffer in bytes. */
    vk::DeviceSize size_bytes = 0;

    /** @brief The bit flags specifying how the buffer will be used. */
    vk::BufferUsageFlags usage_flags;

    /** @brief The parameters for allocating buffer memory. */
    const VmaAllocationCreateInfo& allocation_create_info;
  };

  /**
   * @brief Creates a @ref Buffer.
   * @param allocator The allocator for creating the buffer.
   * @param create_info @copybrief Buffer::CreateInfo
   */
  Buffer(const vma::Allocator& allocator, const CreateInfo& create_info);

  Buffer(const Buffer&) = delete;
  Buffer(Buffer&& buffer) noexcept { *this = std::move(buffer); }

  Buffer& operator=(const Buffer&) = delete;
  Buffer& operator=(Buffer&& buffer) noexcept;

  /** @brief Frees the underlying memory and destroys the buffer. */
  virtual ~Buffer() noexcept;

  /** @brief Gets the underlying Vulkan buffer handle. */
  [[nodiscard]] vk::Buffer operator*() const noexcept { return buffer_; }

  /** @brief Gets the size of the buffer in bytes. */
  [[nodiscard]] vk::DeviceSize size_bytes() const noexcept { return size_bytes_; }

  /**
   * @brief Records copy commands to transfer data to this buffer.
   * @param src_buffer The source buffer to copy data from.
   * @param command_buffer The command buffer for recording copy commands.
   * @warning The caller is responsible for submitting @p command_buffer to a Vulkan queue to begin execution.
   */
  void Copy(const Buffer& src_buffer, const vk::CommandBuffer command_buffer);

protected:
  Buffer() noexcept = default;

  VmaAllocator allocator_ = nullptr;
  VmaAllocation allocation_ = nullptr;
  vk::Buffer buffer_;
  vk::DeviceSize size_bytes_ = 0;
  vk::BufferUsageFlags usage_flags_;
};

/**
 * @brief An abstraction for a Vulkan buffer with host-visible memory.
 * @details This class represents a buffer residing in host-visible memory for data that must be accessible by both the
 *          host and device such as uniform buffers that require frequent read/write access.
 */
export class [[nodiscard]] HostVisibleBuffer final : public Buffer {
public:
  /** @brief The parameters for creating a @ref HostVisibleBuffer. */
  struct [[nodiscard]] CreateInfo {
    /** @brief @copybrief Buffer::CreateInfo::size_bytes */
    vk::DeviceSize size_bytes = 0;

    /** @brief @copybrief Buffer::CreateInfo::usage_flags */
    vk::BufferUsageFlags usage_flags;
  };

  /**
   * @brief Creates a @ref HostVisibleBuffer.
   * @param allocator The allocator for creating the buffer.
   * @param create_info @copybrief HostVisibleBuffer::CreateInfo
   */
  HostVisibleBuffer(const vma::Allocator& allocator, const CreateInfo& create_info);

  HostVisibleBuffer(const HostVisibleBuffer&) = delete;
  HostVisibleBuffer(HostVisibleBuffer&& host_visible_buffer) noexcept { *this = std::move(host_visible_buffer); }

  HostVisibleBuffer& operator=(const HostVisibleBuffer&) = delete;
  HostVisibleBuffer& operator=(HostVisibleBuffer&& host_visible_buffer) noexcept;

  /** @brief Unmaps the underlying memory and destroys the buffer. */
  ~HostVisibleBuffer() noexcept override { UnmapMemory(); }

  /**
   * @brief Maps the memory for this buffer allowing data to be written to it by the host.
   * @warning This function must be called before invoking @ref HostVisibleBuffer::Copy.
   * @note If this buffer is already mapped, this function does nothing.
   */
  void MapMemory();

  /**
   * @brief Unmaps the memory for this buffer releasing system resources used to maintain that mapping.
   * @warning This function should be called when the host is done writing data to this buffer.
   * @note If this buffer is already unmapped, this function does nothing.
   */
  void UnmapMemory() noexcept;

  /**
   * @brief Copies data to this buffer.
   * @details This function provides a type-safe interface for copying arbitrary data to a host-visible buffer which is
   *          useful in situations that require working with one or many elements such as a uniform buffer representing
   *          a single structure of related properties (e.g., camera transformations) or a range of homogeneous data
   *          types (e.g., world-space lights).
   * @tparam T The type of each element in @p data_view.
   * @param data_view A view of the data to copy.
   * @warning @ref HostVisibleBuffer::MapMemory must be called before invoking this function.
   */
  template <typename T>
  void Copy(const DataView<T> data_view) {
    assert(mapped_memory_ != nullptr);
    assert(data_view.size_bytes() <= size_bytes_);
    memcpy(mapped_memory_, data_view.data(), data_view.size_bytes());
    const auto result = vmaFlushAllocation(allocator_, allocation_, 0, vk::WholeSize);
    vk::detail::resultCheck(static_cast<vk::Result>(result), "Flush allocation failed");
  }

private:
  void* mapped_memory_ = nullptr;
};

/**
 * @brief Creates an intermediate staging buffer in host-visible memory and copies data to it from the host.
 * @details To transfer data to device-local memory, data must first be copied to an intermediate staging buffer which
 *          can then be used to copy data directly to device-local memory.
 * @tparam T The type of each element in @p data_view.
 * @param allocator The allocator for creating the staging buffer.
 * @param data_view A view of the data to copy to the staging buffer.
 * @return An unmapped staging buffer containing the data in @p data_view.
 */
export template <typename T>
[[nodiscard]] HostVisibleBuffer CreateStagingBuffer(const vma::Allocator& allocator, const DataView<T> data_view) {
  HostVisibleBuffer staging_buffer{allocator,
                                   HostVisibleBuffer::CreateInfo{.size_bytes = data_view.size_bytes(),
                                                                 .usage_flags = vk::BufferUsageFlagBits::eTransferSrc}};
  staging_buffer.MapMemory();
  staging_buffer.Copy(data_view);
  staging_buffer.UnmapMemory();  // staging buffers are copied once so they can be unmapped immediately
  return staging_buffer;
}

/**
 * @brief Creates a buffer in device-local memory and copies data to it from an intermediate staging buffer.
 * @details For memory that is infrequently updated and requires fast memory access on the device, it's more performant
 *          to create a buffer in device-local memory and copy data to it from a staging buffer in host-visible memory.
 * @param allocator The allocator for creating the device-local buffer.
 * @param command_buffer The command buffer for recording copy commands to transfer data to the device-local buffer.
 * @param host_visible_buffer The host-visible buffer to copy data from.
 * @param usage_flags The bitwise flags indicating how the buffer will be used.
 * @return A buffer in device-local memory that will contain the data in @p host_visible_buffer when @p command_buffer
 *         completes queue submission.
 * @warning The caller is responsible for submitting @p command_buffer to a Vulkan queue to begin execution.
 */
export [[nodiscard]] Buffer CreateDeviceLocalBuffer(const vma::Allocator& allocator,
                                                    const vk::CommandBuffer command_buffer,
                                                    const HostVisibleBuffer& host_visible_buffer,
                                                    const vk::BufferUsageFlags usage_flags) {
  Buffer device_local_buffer{allocator,
                             Buffer::CreateInfo{.size_bytes = host_visible_buffer.size_bytes(),
                                                .usage_flags = usage_flags | vk::BufferUsageFlagBits::eTransferDst,
                                                .allocation_create_info = vma::kDeviceLocalAllocationCreateInfo}};
  device_local_buffer.Copy(host_visible_buffer, command_buffer);
  return device_local_buffer;
}

}  // namespace vktf

module :private;

namespace vktf {

namespace {

std::pair<VmaAllocation, vk::Buffer> CreateBuffer(const VmaAllocator allocator, const Buffer::CreateInfo& create_info) {
  const auto& [size_bytes, usage_flags, allocation_create_info] = create_info;
  const VkBufferCreateInfo buffer_create_info{.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
                                              .size = size_bytes,
                                              .usage = static_cast<VkBufferUsageFlags>(usage_flags)};

  VkBuffer buffer = nullptr;
  VmaAllocation allocation = nullptr;
  const auto result =
      vmaCreateBuffer(allocator, &buffer_create_info, &allocation_create_info, &buffer, &allocation, nullptr);
  vk::detail::resultCheck(static_cast<vk::Result>(result), "Buffer creation failed");

  return std::pair{allocation, buffer};
}

void DestroyBuffer(const VmaAllocator allocator, const vk::Buffer buffer, const VmaAllocation allocation) noexcept {
  if (allocator != nullptr) {
    vmaDestroyBuffer(allocator, buffer, allocation);
  }
}

}  // namespace

Buffer::Buffer(const vma::Allocator& allocator, const CreateInfo& create_info)
    : allocator_{*allocator}, size_bytes_{create_info.size_bytes}, usage_flags_{create_info.usage_flags} {
  std::tie(allocation_, buffer_) = CreateBuffer(*allocator, create_info);
}

Buffer& Buffer::operator=(Buffer&& buffer) noexcept {
  if (this != &buffer) {
    DestroyBuffer(allocator_, buffer_, allocation_);
    buffer_ = std::exchange(buffer.buffer_, nullptr);
    usage_flags_ = std::exchange(buffer.usage_flags_, {});
    size_bytes_ = std::exchange(buffer.size_bytes_, 0);
    allocator_ = std::exchange(buffer.allocator_, nullptr);
    allocation_ = std::exchange(buffer.allocation_, nullptr);
  }
  return *this;
}

Buffer::~Buffer() noexcept { DestroyBuffer(allocator_, buffer_, allocation_); }

void Buffer::Copy(const Buffer& src_buffer, const vk::CommandBuffer command_buffer) {
  const auto& dst_buffer = *this;
  assert(src_buffer.size_bytes_ <= dst_buffer.size_bytes_);
  assert(src_buffer.usage_flags_ & vk::BufferUsageFlagBits::eTransferSrc);
  assert(dst_buffer.usage_flags_ & vk::BufferUsageFlagBits::eTransferDst);
  command_buffer.copyBuffer(*src_buffer, *dst_buffer, vk::BufferCopy{.size = src_buffer.size_bytes_});
}

HostVisibleBuffer::HostVisibleBuffer(const vma::Allocator& allocator, const CreateInfo& create_info)
    : Buffer{allocator,
             Buffer::CreateInfo{.size_bytes = create_info.size_bytes,
                                .usage_flags = create_info.usage_flags,
                                .allocation_create_info = vma::kHostVisibleAllocationCreateInfo}} {}

HostVisibleBuffer& HostVisibleBuffer::operator=(HostVisibleBuffer&& host_visible_buffer) noexcept {
  if (this != &host_visible_buffer) {
    UnmapMemory();
    mapped_memory_ = std::exchange(host_visible_buffer.mapped_memory_, nullptr);
    Buffer::operator=(std::move(host_visible_buffer));
  }
  return *this;
}

void HostVisibleBuffer::MapMemory() {
  if (mapped_memory_ == nullptr) {
    const auto result = vmaMapMemory(allocator_, allocation_, &mapped_memory_);
    vk::detail::resultCheck(static_cast<vk::Result>(result), "Map memory failed");
  }
}

void HostVisibleBuffer::UnmapMemory() noexcept {
  if (mapped_memory_ != nullptr) {
    vmaUnmapMemory(allocator_, allocation_);
    mapped_memory_ = nullptr;
  }
}

}  // namespace vktf
