#include "graphics/model.h"

#include <algorithm>
#include <cassert>
#include <cstdint>
#include <limits>
#include <ranges>
#include <span>
#include <stdexcept>
#include <string>
#include <utility>

#include <assimp/postprocess.h>
#include <assimp/scene.h>
#include <vk_mem_alloc.h>
#include <assimp/DefaultLogger.hpp>
#include <assimp/Importer.hpp>
#include <glm/glm.hpp>

#include "graphics/device.h"

namespace {

template <glm::length_t N, typename T>
  requires(N > 0 && N <= 3)
glm::vec<N, T> GetVec(const std::span<aiVector3t<T>> data, const std::size_t index, const bool normalize = false) {
  glm::vec<N, T> v{};
  if (index < data.size()) {
    for (std::uint32_t component = 0; component < N; ++component) {
      v[component] = data[index][component];
    }
    if (normalize) {
      const auto length = glm::length(v);
      assert(length > 0.0f);
      v = v / length;
    }
  }
  return v;
}

std::vector<gfx::Vertex> GetVertices(const aiMesh& mesh) {
  const std::span positions{mesh.mVertices, mesh.HasPositions() ? mesh.mNumVertices : 0};
  const std::span normals{mesh.mNormals, mesh.HasNormals() ? mesh.mNumVertices : 0};
  const std::span texture_coordinates{mesh.mTextureCoords[0], mesh.HasTextureCoords(0) ? mesh.mNumVertices : 0};

  return std::views::iota(0u, mesh.mNumVertices)
         | std::views::transform([positions, normals, texture_coordinates](const auto index) {
             return gfx::Vertex{.position = GetVec<3>(positions, index),
                                .texture_coordinates = GetVec<2>(texture_coordinates, index),
                                .normal = GetVec<3>(normals, index, true)};
           })
         | std::ranges::to<std::vector>();
}

std::vector<std::uint32_t> GetIndices(const aiMesh& mesh) {
  return std::span{mesh.mFaces, mesh.HasFaces() ? mesh.mNumFaces : 0}  //
         | std::views::transform([](const auto& face) {
             return std::span{face.mIndices, face.mNumIndices};
           })
         | std::views::join  //
         | std::ranges::to<std::vector>();
}

glm::mat4 GetTransform(const aiNode& node) {
  const auto& transform = node.mTransformation;
  const std::span x{transform[0], 4};
  const std::span y{transform[1], 4};
  const std::span z{transform[2], 4};
  const std::span w{transform[3], 4};

  // clang-format off
  return glm::mat4{x[0], y[0], z[0], w[0],
                   x[1], y[1], z[1], w[1],
                   x[2], y[2], z[2], w[2],
                   x[3], y[3], z[3], w[3]};
  // clang-format on
}

template <typename T>
std::pair<gfx::Buffer, gfx::Buffer> CreateBuffers(const std::vector<T>& data,
                                                  const vk::BufferUsageFlags buffer_usage_flags,
                                                  const vk::CommandBuffer command_buffer,
                                                  const VmaAllocator allocator) {
  const auto size_bytes = sizeof(T) * data.size();

  static constexpr VmaAllocationCreateInfo kStagingBufferAllocationCreateInfo{
      .flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT,
      .usage = VMA_MEMORY_USAGE_AUTO};
  gfx::Buffer staging_buffer{size_bytes,
                             vk::BufferUsageFlagBits::eTransferSrc,
                             allocator,
                             kStagingBufferAllocationCreateInfo};
  staging_buffer.CopyOnce<const T>(data);

  static constexpr VmaAllocationCreateInfo kBufferAllocationCreateInfo{.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE};
  gfx::Buffer buffer{size_bytes,
                     buffer_usage_flags | vk::BufferUsageFlagBits::eTransferDst,
                     allocator,
                     kBufferAllocationCreateInfo};
  command_buffer.copyBuffer(*staging_buffer, *buffer, vk::BufferCopy{.size = size_bytes});

  return std::pair{std::move(staging_buffer), std::move(buffer)};
}

std::unique_ptr<gfx::Model::Node> ImportNode(const aiNode& node, std::vector<gfx::Mesh>& scene_meshes) {
  auto node_meshes = std::span{node.mMeshes, node.mNumMeshes}
                     | std::views::transform([&scene_meshes](const auto index) {
                         assert(index < scene_meshes.size());
                         return std::move(scene_meshes[index]);
                       })
                     | std::ranges::to<std::vector>();

  auto node_children = std::span{node.mChildren, node.mNumChildren}
                       | std::views::transform([&scene_meshes](const auto* const child_node) {
                           return ImportNode(*child_node, scene_meshes);
                         })
                       | std::ranges::to<std::vector>();

  return std::make_unique<gfx::Model::Node>(std::move(node_meshes), std::move(node_children), GetTransform(node));
}

std::unique_ptr<gfx::Model::Node> ImportScene(const aiScene& scene,
                                              const vk::Device device,
                                              const vk::Queue queue,
                                              const std::uint32_t queue_family_index,
                                              const VmaAllocator allocator) {
  const auto command_pool =
      device.createCommandPoolUnique(vk::CommandPoolCreateInfo{.flags = vk::CommandPoolCreateFlagBits::eTransient,
                                                               .queueFamilyIndex = queue_family_index});

  const auto command_buffers =
      device.allocateCommandBuffersUnique(vk::CommandBufferAllocateInfo{.commandPool = *command_pool,
                                                                        .level = vk::CommandBufferLevel::ePrimary,
                                                                        .commandBufferCount = 1});
  assert(!command_buffers.empty());
  const auto command_buffer = *command_buffers.front();
  command_buffer.begin(vk::CommandBufferBeginInfo{.flags = vk::CommandBufferUsageFlagBits::eOneTimeSubmit});

  std::vector<gfx::Buffer> staging_buffers;
  staging_buffers.reserve(scene.mNumMeshes * 2);

  auto scene_meshes =
      std::span{scene.mMeshes, scene.mNumMeshes}  //
      | std::views::transform([=, &staging_buffers](const auto* const mesh) {
          assert(mesh != nullptr);

          const auto vertices = GetVertices(*mesh);
          auto [staging_vertex_buffer, vertex_buffer] =
              CreateBuffers(vertices, vk::BufferUsageFlagBits::eVertexBuffer, command_buffer, allocator);

          const auto indices = GetIndices(*mesh);
          auto [staging_index_buffer, index_buffer] =
              CreateBuffers(indices, vk::BufferUsageFlagBits::eIndexBuffer, command_buffer, allocator);

          // keep staging buffers alive until copy commands have completed
          staging_buffers.push_back(std::move(staging_vertex_buffer));
          staging_buffers.push_back(std::move(staging_index_buffer));

          return gfx::Mesh{std::move(vertex_buffer),
                           std::move(index_buffer),
                           static_cast<std::uint32_t>(indices.size())};
        })
      | std::ranges::to<std::vector>();

  command_buffer.end();

  const auto copy_fence = device.createFenceUnique(vk::FenceCreateInfo{});
  queue.submit(vk::SubmitInfo{.commandBufferCount = 1, .pCommandBuffers = &command_buffer}, *copy_fence);

  // import the rest of the scene while copy commands are processing
  auto root_node = ImportNode(*scene.mRootNode, scene_meshes);

  static constexpr auto kMaxTimeout = std::numeric_limits<std::uint64_t>::max();
  const auto result = device.waitForFences(*copy_fence, vk::True, kMaxTimeout);
  vk::resultCheck(result, "Fence failed to enter a signaled state");

  return root_node;
}

void RenderNode(const gfx::Model::Node& node,
                const vk::CommandBuffer command_buffer,
                const vk::PipelineLayout pipeline_layout,
                const glm::mat4& parent_transform = glm::mat4{1.0f}) {
  const auto node_transform = parent_transform * node.transform;
  command_buffer.pushConstants<gfx::Model::PushConstants>(pipeline_layout,
                                                          vk::ShaderStageFlagBits::eVertex,
                                                          0,
                                                          gfx::Model::PushConstants{.node_transform = node_transform});

  for (const auto& mesh : node.meshes) {
    mesh.Render(command_buffer);
  }

  for (const auto& child_node : node.children) {
    RenderNode(*child_node, command_buffer, pipeline_layout, node_transform);
  }
}

}  // namespace

gfx::Model::Model(const Device& device, const VmaAllocator allocator, const std::filesystem::path& filepath) {
  Assimp::Importer importer;
  std::uint32_t import_flags = aiProcessPreset_TargetRealtime_Fast;

#ifndef NDEBUG
  import_flags |= aiProcess_ValidateDataStructure;
  Assimp::DefaultLogger::create(ASSIMP_DEFAULT_LOG_NAME, Assimp::Logger::DEBUGGING);
#endif

  const auto* scene = importer.ReadFile(filepath.string(), import_flags);
  if (scene == nullptr) throw std::runtime_error{importer.GetErrorString()};

  const auto transfer_queue = device.transfer_queue();
  const auto transfer_index = device.physical_device().queue_family_indices().transfer_index;
  root_node_ = ImportScene(*scene, *device, transfer_queue, transfer_index, allocator);
}

void gfx::Model::Render(const vk::CommandBuffer command_buffer, const vk::PipelineLayout pipeline_layout) const {
  RenderNode(*root_node_, command_buffer, pipeline_layout);
}
