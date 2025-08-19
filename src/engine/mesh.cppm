module;

#include <concepts>
#include <cstdint>
#include <utility>
#include <vector>

#include <glm/glm.hpp>
#include <vulkan/vulkan.hpp>

export module mesh;

import bounding_box;
import buffer;
import data_view;
import vma_allocator;

namespace vktf {

/** @brief A mesh primitive vertex. */
export struct [[nodiscard]] Vertex {
  /** @brief The local-space vertex position. */
  glm::vec3 position{0.0f};

  /**
   * @brief The local-space normal vector.
   * @warning This vector is assumed to be unit length.
   */
  glm::vec3 normal{0.0f};

  /**
   * @brief The local-space tangent vector.
   * @warning This vector is assumed to be unit length.
   */
  glm::vec4 tangent{0.0f};

  /** @brief The first texture coordinate set. */
  glm::vec2 texcoord_0{0.0f};
};

/**
 * @brief A concept defining the allowable types for a variable-width vertex index.
 * @tparam T The vertex index type.
 */
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

/**
 * @brief A mesh primitive in host-visible memory.
 * @details This class handles creating host-visible staging buffers with vertex and index data for a mesh primitive.
 */
export class [[nodiscard]] StagingPrimitive {
public:
  /**
   * @brief The parameters for creating a @ref StagingPrimitive.
   * @tparam T The vertex index type.
   */
  template <IndexType T>
  struct [[nodiscard]] CreateInfo {
    /** @brief The primitive vertices. */
    const std::vector<Vertex>& vertices;

    /** @brief The primitive indices. */
    const std::vector<T>& indices;
  };

  /**
   * @brief Creates a @ref StagingPrimitive.
   * @tparam T The vertex index type.
   * @param allocator The allocator for creating staging buffers.
   * @param create_info @copybrief StagingPrimitive::CreateInfo
   */
  template <IndexType T>
  StagingPrimitive(const vma::Allocator& allocator, const CreateInfo<T>& create_info)
      : vertex_buffer_{CreateStagingBuffer<Vertex>(allocator, create_info.vertices)},
        index_buffer_{CreateStagingBuffer<T>(allocator, create_info.indices)},
        index_type_{GetIndexType<T>()},
        index_count_{static_cast<std::uint32_t>(create_info.indices.size())} {}

  /** @brief Gets the vertex staging buffer. */
  [[nodiscard]] const HostVisibleBuffer& vertex_buffer() const noexcept { return vertex_buffer_; }

  /** @brief Gets the index staging buffer. */
  [[nodiscard]] const HostVisibleBuffer& index_buffer() const noexcept { return index_buffer_; }

  /** @brief Gets the vertex index type. */
  [[nodiscard]] vk::IndexType index_type() const noexcept { return index_type_; }

  /** @brief Gets the number of vertex indices. */
  [[nodiscard]] std::uint32_t index_count() const noexcept { return index_count_; }

private:
  HostVisibleBuffer vertex_buffer_;
  HostVisibleBuffer index_buffer_;
  vk::IndexType index_type_;
  std::uint32_t index_count_;
};

/**
 * @brief A mesh primitive in device-local memory.
 * @details This class handles creating device-local vertex and index buffers, recording copy commands to transfer data
 *          from host-visible staging buffers, and assigning a descriptor set that references its material resources.
 */
export class [[nodiscard]] Primitive {
public:
  /** @brief The parameters for creating a @ref Primitive. */
  struct [[nodiscard]] CreateInfo {
    /** @brief The staging primitive to copy to device-local memory. */
    const StagingPrimitive& staging_primitive;

    /** @brief The descriptor set for the primitive material. */
    vk::DescriptorSet material_descriptor_set;
  };

  /**
   * @brief Creates a @ref Primitive.
   * @param allocator The allocator for creating device-local buffers.
   * @param command_buffer The command buffer for recording copy commands.
   * @param create_info @copybrief Primitive::CreateInfo
   * @warning The caller is responsible for submitting @p command_buffer to a Vulkan queue to begin execution.
   */
  Primitive(const vma::Allocator& allocator, vk::CommandBuffer command_buffer, const CreateInfo& create_info);

  /** @brief Gets the descriptor set for the primitive material. */
  [[nodiscard]] vk::DescriptorSet material_descriptor_set() const noexcept { return material_descriptor_set_; }

  /**
   * @brief Records draw commands to render the primitive.
   * @param command_buffer The command buffer for recording draw commands.
   * @warning The caller is responsible for submitting @p command_buffer to a Vulkan queue to begin execution.
   */
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
  vk::DescriptorSet material_descriptor_set_;
};

/**
 * @brief A mesh in device-local memory.
 * @details This class represents a collection of mesh primitives that form a complete mesh.
 */
export class Mesh {
public:
  /**
   * @brief Creates a @ref Mesh.
   * @param primitives The mesh primitives.
   * @param bounding_box The bounding box that encloses all vertices in the mesh.
   */
  Mesh(std::vector<Primitive> primitives, const BoundingBox& bounding_box);

  /** @brief Gets the mesh primitives. */
  [[nodiscard]] const std::vector<Primitive>& primitives() const { return primitives_; }

  /** @brief Gets the bounding box that encloses all vertices in the mesh. */
  [[nodiscard]] const BoundingBox& bounding_box() const { return bounding_box_; }

private:
  std::vector<Primitive> primitives_;
  BoundingBox bounding_box_;
};

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
      material_descriptor_set_{create_info.material_descriptor_set} {}

Mesh::Mesh(std::vector<Primitive> primitives, const BoundingBox& bounding_box)
    : primitives_{std::move(primitives)}, bounding_box_{bounding_box} {}

}  // namespace vktf
