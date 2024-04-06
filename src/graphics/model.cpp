#include "graphics/model.h"

#include <cassert>
#include <cstdint>
#include <cstring>
#include <format>
#include <limits>
#include <ranges>
#include <span>
#include <stdexcept>
#include <string_view>
#include <unordered_map>
#include <utility>

#include <cgltf.h>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>

#include "graphics/buffer.h"
#include "graphics/device.h"

template <>
struct std::formatter<cgltf_result> : std::formatter<std::string_view> {
  [[nodiscard]] auto format(const cgltf_result result, auto& format_context) const {
    return std::formatter<std::string_view>::format(to_string(result), format_context);
  }

private:
  static std::string_view to_string(const cgltf_result result) noexcept {
    switch (result) {
      // clang-format off
#define CASE(kResult) case kResult: return #kResult;  // NOLINT(cppcoreguidelines-macro-usage)
      CASE(cgltf_result_success)
      CASE(cgltf_result_data_too_short)
      CASE(cgltf_result_unknown_format)
      CASE(cgltf_result_invalid_json)
      CASE(cgltf_result_invalid_gltf)
      CASE(cgltf_result_invalid_options)
      CASE(cgltf_result_file_not_found)
      CASE(cgltf_result_io_error)
      CASE(cgltf_result_out_of_memory)
      CASE(cgltf_result_legacy_gltf)
#undef CASE
      // clang-format on
      default:
        std::unreachable();
    }
  }
};

namespace {

using UniqueCgltfData = std::unique_ptr<cgltf_data, decltype(&cgltf_free)>;

UniqueCgltfData ParseFile(const char* const gltf_filepath) {
  static constexpr cgltf_options kOptions{};
  UniqueCgltfData data{nullptr, nullptr};

  if (const auto result = cgltf_parse_file(&kOptions, gltf_filepath, std::out_ptr(data, cgltf_free));
      result != cgltf_result_success) {
    throw std::runtime_error{std::format("Parse file failed with with error {}", result)};
  }
  if (const auto result = cgltf_load_buffers(&kOptions, data.get(), gltf_filepath); result != cgltf_result_success) {
    throw std::runtime_error{std::format("Load buffers failed with error {}", result)};
  }
#ifndef NDEBUG
  if (const auto result = cgltf_validate(data.get()); result != cgltf_result_success) {
    throw std::runtime_error{std::format("Validation failed with error {}", result)};
  }
#endif

  return data;
}

template <glm::length_t N>
std::vector<glm::vec<N, float>> UnpackFloats(const cgltf_accessor& accessor) {
  if (const auto components = cgltf_num_components(accessor.type); components != N) {
    const auto* const name = accessor.name != nullptr && std::strlen(accessor.name) > 0 ? accessor.name : "unknown";
    throw std::runtime_error{std::format("Failed to unpack floats for {} with {} components", name, components)};
  }
  std::vector<glm::vec<N, float>> data(accessor.count);
  cgltf_accessor_unpack_floats(&accessor, glm::value_ptr(data.front()), N * accessor.count);
  return data;
}

std::vector<gfx::Vertex> GetVertices(const cgltf_primitive& primitive) {
  std::vector<glm::vec3> positions, normals;

  for (const auto& attribute : std::span{primitive.attributes, primitive.attributes_count}) {
    switch (const auto& accessor = *attribute.data; attribute.type) {
      case cgltf_attribute_type_position: {
        positions = UnpackFloats<3>(accessor);
        break;
      }
      case cgltf_attribute_type_normal: {
        normals = UnpackFloats<3>(accessor);
        break;
      }
      default:
        break;
    }
  }

  static constexpr auto kCreateVertex = [](const auto& position, const auto& normal) {
    return gfx::Vertex{.position = position, .normal = normal};
  };
  return std::views::zip_transform(kCreateVertex, positions, normals) | std::ranges::to<std::vector>();
}

std::vector<std::uint16_t> GetIndices(const cgltf_accessor& accessor) {
  return std::views::iota(0u, accessor.count)  //
         | std::views::transform([&accessor](const auto accessor_index) {
             const auto vertex_index = cgltf_accessor_read_index(&accessor, accessor_index);
             return static_cast<std::uint16_t>(vertex_index);
           })
         | std::ranges::to<std::vector>();
}

glm::mat4 GetTransform(const cgltf_node& node) {
  glm::mat4 transform{1.0f};
  if (node.has_matrix != 0 || node.has_translation != 0 || node.has_rotation != 0 || node.has_scale != 0) {
    cgltf_node_transform_local(&node, glm::value_ptr(transform));
  }
  return transform;
}

std::vector<gfx::Mesh> GetSubMeshes(const cgltf_node& node,
                                    std::unordered_map<const cgltf_mesh*, std::vector<gfx::Mesh>>& meshes) {
  if (node.mesh == nullptr) return {};
  const auto iterator = meshes.find(node.mesh);
  assert(iterator != meshes.cend());
  return std::move(iterator->second);  // TODO(matthew-rister): this assumes a 1:1 mapping between nodes and meshes
}

std::unique_ptr<gfx::Node> ImportNode(const cgltf_node& node,
                                      std::unordered_map<const cgltf_mesh*, std::vector<gfx::Mesh>>& meshes) {
  auto child_nodes =
      std::span{node.children, node.children_count}
      | std::views::transform([&meshes](const auto* const child_node) { return ImportNode(*child_node, meshes); })
      | std::ranges::to<std::vector>();

  return std::make_unique<gfx::Node>(GetSubMeshes(node, meshes), std::move(child_nodes), GetTransform(node));
}

std::unique_ptr<gfx::Node> ImportScene(const cgltf_scene& scene,
                                       std::unordered_map<const cgltf_mesh*, std::vector<gfx::Mesh>>& meshes) {
  auto scene_nodes = std::span{scene.nodes, scene.nodes_count}
                     | std::views::transform([&meshes](const auto* const node) { return ImportNode(*node, meshes); })
                     | std::ranges::to<std::vector>();

  return std::make_unique<gfx::Node>(std::vector<gfx::Mesh>{}, std::move(scene_nodes), glm::mat4{1.0f});
}

template <typename T>
gfx::Buffer CreateBuffer(const std::vector<T>& data,
                         const vk::BufferUsageFlags buffer_usage_flags,
                         const vk::CommandBuffer command_buffer,
                         const VmaAllocator allocator,
                         std::vector<gfx::Buffer>& staging_buffers) {
  const auto size_bytes = sizeof(T) * data.size();

  static constexpr VmaAllocationCreateInfo kStagingBufferAllocationCreateInfo{
      .flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT,
      .usage = VMA_MEMORY_USAGE_AUTO};
  auto& staging_buffer = staging_buffers.emplace_back(size_bytes,
                                                      vk::BufferUsageFlagBits::eTransferSrc,
                                                      allocator,
                                                      kStagingBufferAllocationCreateInfo);
  staging_buffer.template CopyOnce<T>(data);

  static constexpr VmaAllocationCreateInfo kBufferAllocationCreateInfo{.usage = VMA_MEMORY_USAGE_AUTO};
  gfx::Buffer buffer{size_bytes,
                     buffer_usage_flags | vk::BufferUsageFlagBits::eTransferDst,
                     allocator,
                     kBufferAllocationCreateInfo};
  command_buffer.copyBuffer(*staging_buffer, *buffer, vk::BufferCopy{.size = size_bytes});

  return buffer;
}

std::vector<gfx::Mesh> CreateSubMeshes(const cgltf_mesh& mesh,
                                       const vk::CommandBuffer command_buffer,
                                       const VmaAllocator allocator,
                                       std::vector<gfx::Buffer>& staging_buffers) {
  return std::span{mesh.primitives, mesh.primitives_count}
         | std::views::filter([](const auto& primitive) { return primitive.type == cgltf_primitive_type_triangles; })
         | std::views::transform([command_buffer, allocator, &staging_buffers](const auto& primitive) {
             using enum vk::BufferUsageFlagBits;
             const auto vertices = GetVertices(primitive);
             const auto indices = GetIndices(*primitive.indices);
             return gfx::Mesh{CreateBuffer(vertices, eVertexBuffer, command_buffer, allocator, staging_buffers),
                              CreateBuffer(indices, eIndexBuffer, command_buffer, allocator, staging_buffers),
                              static_cast<std::uint32_t>(indices.size())};
           })
         | std::ranges::to<std::vector>();
}

std::unordered_map<const cgltf_mesh*, std::vector<gfx::Mesh>> CreateMeshes(const cgltf_data& data,
                                                                           const gfx::Device& device,
                                                                           const VmaAllocator allocator) {
  const auto transfer_queue = device.transfer_queue();
  const auto transfer_index = device.queue_family_indices().transfer_index;

  const auto command_pool =
      device->createCommandPoolUnique(vk::CommandPoolCreateInfo{.flags = vk::CommandPoolCreateFlagBits::eTransient,
                                                                .queueFamilyIndex = transfer_index});

  const auto command_buffers =
      device->allocateCommandBuffersUnique(vk::CommandBufferAllocateInfo{.commandPool = *command_pool,
                                                                         .level = vk::CommandBufferLevel::ePrimary,
                                                                         .commandBufferCount = 1});
  const auto command_buffer = *command_buffers.front();
  command_buffer.begin(vk::CommandBufferBeginInfo{.flags = vk::CommandBufferUsageFlagBits::eOneTimeSubmit});

  std::vector<gfx::Buffer> staging_buffers;  // staging buffers must remain in scope until copy commands complete
  auto meshes = std::span{data.meshes, data.meshes_count}
                | std::views::transform([command_buffer, allocator, &staging_buffers](const auto& mesh) {
                    return std::pair{&mesh, CreateSubMeshes(mesh, command_buffer, allocator, staging_buffers)};
                  })
                | std::ranges::to<std::unordered_map>();

  command_buffer.end();

  const auto fence = device->createFenceUnique(vk::FenceCreateInfo{});
  transfer_queue.submit(vk::SubmitInfo{.commandBufferCount = 1, .pCommandBuffers = &command_buffer}, *fence);

  static constexpr auto kMaxTimeout = std::numeric_limits<std::uint64_t>::max();
  const auto result = device->waitForFences(*fence, vk::True, kMaxTimeout);
  vk::resultCheck(result, "Fence failed to enter a signaled state");

  return meshes;
}

void RenderNode(const gfx::Node& node,
                const vk::CommandBuffer command_buffer,
                const vk::PipelineLayout pipeline_layout,
                const glm::mat4& parent_transform = glm::mat4{1.0f}) {
  const auto node_transform = parent_transform * node.transform;
  command_buffer.pushConstants<gfx::PushConstants>(pipeline_layout,
                                                   vk::ShaderStageFlagBits::eVertex,
                                                   0,
                                                   gfx::PushConstants{.model_transform = node_transform});

  for (const auto& mesh : node.meshes) {
    mesh.Render(command_buffer);
  }

  for (const auto& child_node : node.children) {
    RenderNode(*child_node, command_buffer, pipeline_layout, node_transform);
  }
}

}  // namespace

namespace gfx {

Model::Model(const std::filesystem::path& gltf_filepath, const Device& device, const VmaAllocator allocator) {
  const auto data = ParseFile(gltf_filepath.string().c_str());
  auto meshes = CreateMeshes(*data, device, allocator);
  root_node_ = ImportScene(*data->scene, meshes);
}

void Model::Translate(const float dx, const float dy, const float dz) const {
  auto& root_transform = root_node_->transform;
  root_transform = glm::translate(root_transform, glm::vec3{dx, dy, dz});
}

void Model::Rotate(const glm::vec3& axis, const float angle) const {
  auto& root_transform = root_node_->transform;
  root_transform = glm::rotate(root_transform, angle, axis);
}

void Model::Scale(const float sx, const float sy, const float sz) const {
  auto& root_transform = root_node_->transform;
  root_transform = glm::scale(root_transform, glm::vec3{sx, sy, sz});
}

void Model::Render(const vk::CommandBuffer command_buffer, const vk::PipelineLayout pipeline_layout) const {
  RenderNode(*root_node_, command_buffer, pipeline_layout);
}

}  // namespace gfx
