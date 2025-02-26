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
  struct IndexBuffer {
    Buffer buffer;
    vk::IndexType index_type = vk::IndexType::eUint16;
    std::uint32_t index_count = 0;
  };

  template <IndexType T>
  static consteval vk::IndexType GetIndexType() {
    if constexpr (std::same_as<T, std::uint16_t>) {
      return vk::IndexType::eUint16;
    } else {
      static_assert(std::same_as<T, std::uint32_t>);
      return vk::IndexType::eUint32;
    }
  }

public:
  template <IndexType T>
  Primitive(std::vector<Vertex> vertices,
            std::vector<T> indices,
            const Material* const material,
            const vk::CommandBuffer command_buffer,
            const VmaAllocator allocator,
            // TODO: combine vertex and index staging buffers into a single buffer
            std::unique_ptr<const Buffer>& vertex_staging_buffer,
            std::unique_ptr<const Buffer>& index_staging_buffer)
      : vertex_buffer_{CreateBuffer<Vertex>(vertices,
                                            vk::BufferUsageFlagBits::eVertexBuffer,
                                            command_buffer,
                                            allocator,
                                            vertex_staging_buffer)},
        index_buffer_{CreateBuffer<T>(indices,
                                      vk::BufferUsageFlagBits::eIndexBuffer,
                                      command_buffer,
                                      allocator,
                                      index_staging_buffer),
                      GetIndexType<T>(),
                      static_cast<std::uint32_t>(indices.size())},
        material_{material} {
    assert(material != nullptr);
  }

  [[nodiscard]] const Material* material() const noexcept { return material_; }

  void Render(const vk::CommandBuffer command_buffer) const {
    command_buffer.bindVertexBuffers(0, *vertex_buffer_, static_cast<vk::DeviceSize>(0));
    command_buffer.bindIndexBuffer(*index_buffer_.buffer, 0, index_buffer_.index_type);
    command_buffer.drawIndexed(index_buffer_.index_count, 1, 0, 0, 0);
  }

private:
  Buffer vertex_buffer_;
  IndexBuffer index_buffer_;
  const Material* material_ = nullptr;
};

export using Mesh = std::vector<Primitive>;

}  // namespace vktf
