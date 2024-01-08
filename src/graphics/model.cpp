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
#include "graphics/texture2d.h"

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
      v = glm::normalize(v);
    }
  }
  return v;
}

std::vector<gfx::Mesh::Vertex> GetVertices(const aiMesh& mesh) {
  const std::span positions{mesh.mVertices, mesh.HasPositions() ? mesh.mNumVertices : 0};
  const std::span normals{mesh.mNormals, mesh.HasNormals() ? mesh.mNumVertices : 0};
  const std::span texture_coordinates{mesh.mTextureCoords[0], mesh.HasTextureCoords(0) ? mesh.mNumVertices : 0};
  const std::span tangents{mesh.mTangents, mesh.HasTangentsAndBitangents() ? mesh.mNumVertices : 0};

  return std::views::iota(0u, mesh.mNumVertices)
         | std::views::transform([positions, normals, texture_coordinates, tangents](const auto index) {
             return gfx::Mesh::Vertex{.position = GetVec<3>(positions, index),
                                      .texture_coordinates = GetVec<2>(texture_coordinates, index),
                                      .normal = GetVec<3>(normals, index, true),
                                      .tangent = GetVec<3>(tangents, index, true)};
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
                                             const aiScene& scene,
                                             const aiNode& node,
                                             const gfx::Materials& materials) {
  auto node_meshes =
      std::span{node.mMeshes, node.mNumMeshes}
      | std::views::transform([&, scene_meshes = std::span{scene.mMeshes, scene.mNumMeshes}](const auto index) {
          const auto& mesh = *scene_meshes[index];
          const auto& material = materials[mesh.mMaterialIndex];
          return gfx::Mesh{device, GetVertices(mesh), GetIndices(mesh), material.descriptor_set()};
        })
      | std::ranges::to<std::vector>();

  auto node_children =
      std::span{node.mChildren, node.mNumChildren}
      | std::views::transform([&](const auto* child_node) { return ImportNode(device, scene, *child_node, materials); })
      | std::ranges::to<std::vector>();

  return std::make_unique<gfx::Model::Node>(std::move(node_meshes), std::move(node_children), GetTransform(node));
}

std::optional<gfx::Texture2d> CreateTexture(const gfx::Device& device,
                                            const vk::Format format,
                                            const aiMaterial& material,
                                            const aiTextureType texture_type,
                                            const std::filesystem::path& parent_path) {
  if (material.GetTextureCount(texture_type) > 0) {
    aiString texture_path;
    material.GetTexture(texture_type, 0, &texture_path);
    return gfx::Texture2d{device, format, parent_path / texture_path.C_Str()};
  }
  return std::nullopt;
}

gfx::Materials CreateMaterials(const gfx::Device& device,
                               const aiScene& scene,
                               const std::filesystem::path& parent_path) {
  gfx::Materials materials{*device, scene.mNumMaterials};
  for (auto index = 0; const auto* material : std::span{scene.mMaterials, scene.mNumMaterials}) {
    auto diffuse_map = CreateTexture(device, vk::Format::eR8G8B8A8Srgb, *material, aiTextureType_DIFFUSE, parent_path);
    auto normal_map = CreateTexture(device, vk::Format::eR8G8B8A8Unorm, *material, aiTextureType_NORMALS, parent_path);
    if (diffuse_map.has_value() && normal_map.has_value()) {
      materials[index++].UpdateDescriptorSet(*device, std::move(*diffuse_map), std::move(*normal_map));
    }
  }
  return materials;
}

void RenderNode(const gfx::Model::Node& node,
                const vk::CommandBuffer& command_buffer,
                const vk::PipelineLayout& pipeline_layout,
                const glm::mat4& parent_transform = glm::mat4{1.0f}) {
  const auto node_transform = parent_transform * node.transform;
  command_buffer.pushConstants<gfx::Model::PushConstants>(pipeline_layout,
                                                          vk::ShaderStageFlagBits::eVertex,
                                                          0,
                                                          gfx::Model::PushConstants{.node_transform = node_transform});

  for (const auto& mesh : node.meshes) {
    mesh.Render(command_buffer, pipeline_layout);
  }

  for (const auto& child_node : node.children) {
    RenderNode(*child_node, command_buffer, pipeline_layout, node_transform);
  }
}

}  // namespace

gfx::Model::Model(const Device& device, const std::filesystem::path& filepath) {
  Assimp::Importer importer;
  std::uint32_t import_flags = aiProcessPreset_TargetRealtime_Fast | aiProcess_FlipUVs;

#ifndef NDEBUG
  import_flags |= aiProcess_ValidateDataStructure;  // NOLINT(hicpp-signed-bitwise)
  Assimp::DefaultLogger::create(ASSIMP_DEFAULT_LOG_NAME, Assimp::Logger::DEBUGGING);
#endif

  const auto* scene = importer.ReadFile(filepath.string(), import_flags);
  if (scene == nullptr) throw std::runtime_error{importer.GetErrorString()};

  materials_ = CreateMaterials(device, *scene, filepath.parent_path());
  root_node_ = ImportNode(device, *scene, *scene->mRootNode, materials_);
}

void gfx::Model::Render(const vk::CommandBuffer& command_buffer, const vk::PipelineLayout& pipeline_layout) const {
  RenderNode(*root_node_, command_buffer, pipeline_layout);
}
