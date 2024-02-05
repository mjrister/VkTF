#include "graphics/memory.h"

#include <algorithm>
#include <cassert>
#include <cstdint>
#include <ranges>
#include <stdexcept>

#include "graphics/device.h"

namespace {

std::uint32_t FindMemoryTypeIndex(const vk::PhysicalDeviceMemoryProperties& physical_device_memory_properties,
                                  const vk::MemoryRequirements& memory_requirements,
                                  const vk::MemoryPropertyFlags memory_property_flags) {
  for (std::uint32_t index = 0; const auto memory_type : physical_device_memory_properties.memoryTypes) {
    if (memory_requirements.memoryTypeBits & (1u << index)
        && (memory_property_flags & memory_type.propertyFlags) == memory_property_flags) {
      return index;
    }
    ++index;
  }
  throw std::runtime_error{"Unsupported memory type"};
}

vk::UniqueDeviceMemory AllocateMemory(const gfx::Device& device,
                                      const vk::MemoryRequirements& memory_requirements,
                                      const vk::MemoryPropertyFlags memory_property_flags) {
  const auto memory_type_index =
      FindMemoryTypeIndex(device.physical_device().memory_properties(), memory_requirements, memory_property_flags);

  return device->allocateMemoryUnique(
      vk::MemoryAllocateInfo{.allocationSize = memory_requirements.size, .memoryTypeIndex = memory_type_index});
}

}  // namespace

gfx::Memory::Memory(const Device& device,
                    const vk::MemoryRequirements& memory_requirements,
                    const vk::MemoryPropertyFlags memory_property_flags)
    : device_{*device}, memory_{AllocateMemory(device, memory_requirements, memory_property_flags)} {}

gfx::Memory& gfx::Memory::operator=(Memory&& memory) noexcept {
  if (this != &memory) {
    Unmap();
    device_ = std::exchange(memory.device_, {});
    memory_ = std::exchange(memory.memory_, {});
    mapped_memory_ = std::exchange(memory.mapped_memory_, {});
  }
  return *this;
}

void* gfx::Memory::Map() {
  if (mapped_memory_ == nullptr) {
    mapped_memory_ = device_.mapMemory(*memory_, 0, vk::WholeSize);
    assert(mapped_memory_ != nullptr);
  }
  return mapped_memory_;
}

void gfx::Memory::Unmap() noexcept {
  if (mapped_memory_ != nullptr) {
    device_.unmapMemory(*memory_);
    mapped_memory_ = nullptr;
  }
}
