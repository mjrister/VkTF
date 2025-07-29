module;

#include <concepts>
#include <cstdint>
#include <vector>

#include <glm/glm.hpp>
#include <vulkan/vulkan.hpp>

export module mesh;

import bounding_box;
import buffer;
import data_view;
import material;
import vma_allocator;

namespace vktf {

export struct [[nodiscard]] Vertex {
  glm::vec3 position{0.0f};
  glm::vec3 normal{0.0f};
  glm::vec4 tangent{0.0f};
  glm::vec2 texcoord_0{0.0f};
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
    static_assert(std::same_as<T, std::uint32_t>, "Unsupported index type");
    return vk::IndexType::eUint32;
  }
}

export class [[nodiscard]] StagingPrimitive {
public:
  template <IndexType T>
  struct [[nodiscard]] CreateInfo {
    const std::vector<Vertex>& vertices;
    const std::vector<T>& indices;
  };

  template <IndexType T>
  StagingPrimitive(const vma::Allocator& allocator, const CreateInfo<T>& create_info)
      : vertex_buffer_{CreateStagingBuffer<Vertex>(allocator, create_info.vertices)},
        index_buffer_{CreateStagingBuffer<T>(allocator, create_info.indices)},
        index_type_{GetIndexType<T>()},
        index_count_{static_cast<std::uint32_t>(create_info.indices.size())} {}

  [[nodiscard]] const HostVisibleBuffer& vertex_buffer() const noexcept { return vertex_buffer_; }
  [[nodiscard]] const HostVisibleBuffer& index_buffer() const noexcept { return index_buffer_; }
  [[nodiscard]] vk::IndexType index_type() const noexcept { return index_type_; }
  [[nodiscard]] std::uint32_t index_count() const noexcept { return index_count_; }

private:
  HostVisibleBuffer vertex_buffer_;
  HostVisibleBuffer index_buffer_;
  vk::IndexType index_type_;
  std::uint32_t index_count_;
};

export class [[nodiscard]] Primitive {
  using Material = pbr_metallic_roughness::Material;

public:
  struct [[nodiscard]] CreateInfo {
    const StagingPrimitive& staging_primitive;
    const BoundingBox& bounding_box;
    const Material* material = nullptr;
  };

  Primitive(const vma::Allocator& allocator, vk::CommandBuffer command_buffer, const CreateInfo& create_info);

  [[nodiscard]] const BoundingBox& bounding_box() const noexcept { return bounding_box_; }
  [[nodiscard]] const Material* material() const noexcept { return material_; }

  void Render(const vk::CommandBuffer command_buffer) const {
    command_buffer.bindVertexBuffers(0, *vertex_buffer_, static_cast<vk::DeviceSize>(0));
    command_buffer.bindIndexBuffer(*index_buffer_, 0, index_type_);
    command_buffer.drawIndexed(index_count_, 1, 0, 0, 0);
  }

private:
  Buffer vertex_buffer_;
  Buffer index_buffer_;
  vk::IndexType index_type_;
  std::uint32_t index_count_;
  BoundingBox bounding_box_;
  const Material* material_;
};

export using Mesh = std::vector<Primitive>;

}  // namespace vktf

module :private;

namespace vktf {

Primitive::Primitive(const vma::Allocator& allocator,
                     const vk::CommandBuffer command_buffer,
                     const CreateInfo& create_info)
    : vertex_buffer_{CreateDeviceLocalBuffer(allocator,
                                             command_buffer,
                                             create_info.staging_primitive.vertex_buffer(),
                                             vk::BufferUsageFlagBits::eVertexBuffer)},
      index_buffer_{CreateDeviceLocalBuffer(allocator,
                                            command_buffer,
                                            create_info.staging_primitive.index_buffer(),
                                            vk::BufferUsageFlagBits::eIndexBuffer)},
      index_type_{create_info.staging_primitive.index_type()},
      index_count_{create_info.staging_primitive.index_count()},
      bounding_box_{create_info.bounding_box},
      material_{create_info.material} {}

}  // namespace vktf
