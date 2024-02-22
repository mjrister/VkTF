#include "graphics/model.h"

#include <algorithm>
#include <cassert>
#include <cstdint>
#include <optional>
#include <ranges>
#include <span>
#include <stdexcept>
#include <string>
#include <utility>

#include <assimp/postprocess.h>
#include <assimp/scene.h>
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

std::unique_ptr<gfx::Model::Node> ImportNode(const gfx::Device& device,
                                             const VmaAllocator allocator,
                                             const aiScene& scene,
                                             const aiNode& node) {
  auto node_meshes =
      std::span{node.mMeshes, node.mNumMeshes}
      | std::views::transform(
          [&device, allocator, scene_meshes = std::span{scene.mMeshes, scene.mNumMeshes}](const auto index) {
            const auto& mesh = *scene_meshes[index];
            return gfx::Mesh{device, allocator, GetVertices(mesh), GetIndices(mesh)};
          })
      | std::ranges::to<std::vector>();

  auto node_children = std::span{node.mChildren, node.mNumChildren}
                       | std::views::transform([&device, &scene, allocator](const auto* child_node) {
                           return ImportNode(device, allocator, scene, *child_node);
                         })
                       | std::ranges::to<std::vector>();

  return std::make_unique<gfx::Model::Node>(std::move(node_meshes), std::move(node_children), GetTransform(node));
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

  root_node_ = ImportNode(device, allocator, *scene, *scene->mRootNode);
}

void gfx::Model::Render(const vk::CommandBuffer command_buffer, const vk::PipelineLayout pipeline_layout) const {
  RenderNode(*root_node_, command_buffer, pipeline_layout);
}
