module;

#include <array>
#include <cassert>
#include <concepts>
#include <cstdint>
#include <utility>
#include <variant>
#include <vector>

#include <vk_mem_alloc.h>
#include <glm/glm.hpp>
#include <glm/gtc/epsilon.hpp>
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
concept IndexType = std::same_as<T, std::uint16_t> || std::same_as<T, std::uint32_t>;

export class Primitive {
public:
  template <IndexType T>
  Primitive(std::vector<Vertex> vertices, std::vector<T> indices, const Material* const material)
      : vertices_{std::move(vertices)}, indices_{std::move(indices)}, material_{material} {
    assert(material != nullptr);
  }

  [[nodiscard]] const Material* material() const noexcept { return material_; }

  void CreateBuffers(const vk::CommandBuffer command_buffer,
                     const VmaAllocator allocator,
                     std::vector<Buffer>& staging_buffers);

  void Render(const vk::CommandBuffer command_buffer) const;

private:
  struct IndexBuffer {
    Buffer buffer;
    vk::IndexType index_type = vk::IndexType::eUint16;
    std::uint32_t index_count = 0;
  };

  using VertexData = std::vector<Vertex>;
  using IndexData = std::variant<std::vector<std::uint16_t>, std::vector<std::uint32_t>>;

  std::variant<VertexData, Buffer> vertices_;
  std::variant<IndexData, IndexBuffer> indices_;
  const Material* material_ = nullptr;
};

export using Mesh = std::vector<Primitive>;

}  // namespace vktf

module :private;

namespace {

template <vktf::IndexType T>
constexpr vk::IndexType GetIndexType() {
  if constexpr (std::same_as<T, std::uint16_t>) {
    return vk::IndexType::eUint16;
  } else {
    static_assert(std::same_as<T, std::uint32_t>);
    return vk::IndexType::eUint32;
  }
}

}  // namespace

namespace vktf {

void Primitive::CreateBuffers(const vk::CommandBuffer command_buffer,
                              const VmaAllocator allocator,
                              std::vector<Buffer>& staging_buffers) {
  const auto* const vertex_data = std::get_if<VertexData>(&vertices_);
  assert(vertex_data != nullptr);

  using enum vk::BufferUsageFlagBits;
  vertices_ = CreateBuffer<Vertex>(*vertex_data, eVertexBuffer, command_buffer, allocator, staging_buffers);

  const auto* const index_data = std::get_if<IndexData>(&indices_);
  assert(index_data != nullptr);

  indices_ = std::visit(
      [command_buffer, allocator, &staging_buffers]<IndexType T>(const std::vector<T>& index_data_t) {
        return IndexBuffer{
            .buffer = CreateBuffer<T>(index_data_t, eIndexBuffer, command_buffer, allocator, staging_buffers),
            .index_type = GetIndexType<T>(),
            .index_count = static_cast<std::uint32_t>(index_data_t.size())};
      },
      *index_data);
}

void Primitive::Render(const vk::CommandBuffer command_buffer) const {
  const auto* const vertex_buffer = std::get_if<Buffer>(&vertices_);
  assert(vertex_buffer != nullptr);
  command_buffer.bindVertexBuffers(0, **vertex_buffer, static_cast<vk::DeviceSize>(0));

  const auto* const index_buffer = std::get_if<IndexBuffer>(&indices_);
  assert(index_buffer != nullptr);
  command_buffer.bindIndexBuffer(*index_buffer->buffer, 0, index_buffer->index_type);
  command_buffer.drawIndexed(index_buffer->index_count, 1, 0, 0, 0);
}

}  // namespace vktf
