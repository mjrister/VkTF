module;

#include <concepts>
#include <cstdint>
#include <vector>

#include <vk_mem_alloc.h>
#include <glm/glm.hpp>
#include <vulkan/vulkan.hpp>

export module mesh;

import buffer;
import data_view;
import material;

namespace vktf {

export struct Vertex {
  glm::vec3 position{0.0f};
  glm::vec3 normal{0.0f};
  glm::vec4 tangent{0.0f};
  glm::vec2 texture_coordinates_0{0.0f};
};

export template <typename T>
concept IndexType = std::same_as<T, std::uint8_t> || std::same_as<T, std::uint16_t> || std::same_as<T, std::uint32_t>;

template <IndexType T>
consteval vk::IndexType GetIndexType() {
  if constexpr (std::same_as<T, std::uint8_t>) {
    return vk::IndexType::eUint8;
  } else if constexpr (std::same_as<T, std::uint16_t>) {
    return vk::IndexType::eUint16;
  } else {
    static_assert(std::same_as<T, std::uint32_t>);
    return vk::IndexType::eUint32;
  }
}

export class StagingPrimitive {
public:
  template <IndexType T>
  StagingPrimitive(const std::vector<Vertex>& vertices, const std::vector<T>& indices, const VmaAllocator allocator)
      : vertex_staging_buffer_{CreateStagingBuffer<Vertex>(vertices, allocator)},
        index_staging_buffer_{CreateStagingBuffer<T>(indices, allocator)},
        index_type_{GetIndexType<T>()},
        index_count_{static_cast<std::uint32_t>(indices.size())} {}

  [[nodiscard]] const auto& vertex_staging_buffer() const noexcept { return vertex_staging_buffer_; }
  [[nodiscard]] const auto& index_staging_buffer() const noexcept { return index_staging_buffer_; }
  [[nodiscard]] auto index_type() const noexcept { return index_type_; }
  [[nodiscard]] auto index_count() const noexcept { return index_count_; }

private:
  HostVisibleBuffer vertex_staging_buffer_;
  HostVisibleBuffer index_staging_buffer_;
  vk::IndexType index_type_;
  std::uint32_t index_count_;
};

export class Primitive {
public:
  Primitive(const StagingPrimitive& staging_primitive,
            const Material* const material,
            const vk::CommandBuffer command_buffer);

  [[nodiscard]] const auto* material() const noexcept { return material_; }

  void Render(const vk::CommandBuffer command_buffer) const;

private:
  Buffer vertex_buffer_;
  Buffer index_buffer_;
  vk::IndexType index_type_;
  std::uint32_t index_count_;
  const Material* material_;
};

export using Mesh = std::vector<Primitive>;

}  // namespace vktf

module :private;

namespace vktf {

using enum vk::BufferUsageFlagBits;

Primitive::Primitive(const StagingPrimitive& staging_primitive,
                     const Material* const material,
                     const vk::CommandBuffer command_buffer)
    : vertex_buffer_{staging_primitive.vertex_staging_buffer().CreateDeviceLocalBuffer(eVertexBuffer, command_buffer)},
      index_buffer_{staging_primitive.index_staging_buffer().CreateDeviceLocalBuffer(eIndexBuffer, command_buffer)},
      index_type_{staging_primitive.index_type()},
      index_count_{staging_primitive.index_count()},
      material_{material} {}

void Primitive::Render(const vk::CommandBuffer command_buffer) const {
  command_buffer.bindVertexBuffers(0, *vertex_buffer_, static_cast<vk::DeviceSize>(0));
  command_buffer.bindIndexBuffer(*index_buffer_, 0, index_type_);
  command_buffer.drawIndexed(index_count_, 1, 0, 0, 0);
}

}  // namespace vktf
