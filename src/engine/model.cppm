module;

#include <algorithm>
#include <array>
#include <cassert>
#include <concepts>
#include <cstdint>
#include <filesystem>
#include <future>
#include <limits>
#include <memory>
#include <optional>
#include <ranges>
#include <string_view>
#include <type_traits>
#include <unordered_map>
#include <utility>
#include <variant>
#include <vector>

#include <ktx.h>
#include <glm/glm.hpp>
#include <vulkan/vulkan.hpp>

export module model;

import bounding_box;
import descriptor_pool;
import gltf_asset;
import graphics_pipeline;
import ktx_texture;
import log;
import material;
import mesh;
import view_frustum;
import vma_allocator;

namespace vktf {

/**
 * @brief A model in host-visible memory.
 * @details This class handles creating host-visible staging buffers with texture, material, and mesh data from a glTF
 *          asset containing hierarchical scene data.
 */
export class [[nodiscard]] StagingModel {
public:
  /** @brief A type alias for an optional staging material. */
  using Material = std::optional<pbr_metallic_roughness::StagingMaterial>;

  /** @brief A type alias for a vector of optional staging primitives. */
  using Mesh = std::vector<std::optional<StagingPrimitive>>;

  /** @brief The parameters to create a @ref StagingModel. */
  struct [[nodiscard]] CreateInfo {
    /** @brief The glTF asset to copy to host-visible memory. */
    const gltf::Asset& gltf_asset;

    /** @brief The physical device features for determining the transcode target of basis universal KTX textures. */
    const vk::PhysicalDeviceFeatures& physical_device_features;

    /** @brief The log for writing messages when creating a staging model. */
    Log& log;
  };

  /**
   * @brief Creates a @ref StagingModel.
   * @param allocator The allocator for creating staging buffers.
   * @param create_info @copybrief StagingModel::CreateInfo
   */
  explicit StagingModel(const vma::Allocator& allocator, const CreateInfo& create_info);

  /** @brief Gets a map of staging materials by glTF material key. */
  [[nodiscard]] const std::unordered_map<const gltf::Material*, Material>& materials() const { return materials_; }

  /** @brief Gets a map of staging meshes by glTF mesh key. */
  [[nodiscard]] const std::unordered_map<const gltf::Mesh*, Mesh>& meshes() const noexcept { return meshes_; }

private:
  std::unordered_map<const gltf::Material*, Material> materials_;
  std::unordered_map<const gltf::Mesh*, Mesh> meshes_;
};

/**
 * @brief A model in device-local memory.
 * @details The class handles creating device-local resource for textures, materials, and meshes from a glTF asset
 *          containing hierarchical scene data.
 */
export class [[nodiscard]] Model {
public:
  /** @brief A punctual light originating from an infinitely small point in space. */
  struct [[nodiscard]] Light {
    /** @brief The punctual light type. */
    enum class Type : std::uint8_t { kDirectional, kPoint };  // TODO: add support for spot lights

    /** @brief The linear-space light color. */
    glm::vec3 color{0.0f};

    /** @brief The light type (e.g., directional, point). */
    Type type = Type::kDirectional;
  };

  /** @brief A hierarchical graph node. */
  struct [[nodiscard]] Node {
    /** @brief The local transform representing the node position and orientation relative to its parent.  */
    glm::mat4 local_transform{0.0f};

    /**
     * @brief The global transform representing the world-space node position and orientation.
     * @note This property is calculated once per frame by @ref Model::Update.
     */
    glm::mat4 global_transform{0.0f};

    /** @brief The non-owning pointer to the node mesh. */
    const Mesh* mesh = nullptr;

    /** @brief The non-owning pointer to the node light. */
    const Light* light = nullptr;

    /** @brief A list of non-owning pointers to the node children. */
    std::vector<Node*> children;
  };

  /** @brief The parameters for creating a @ref Model. */
  struct [[nodiscard]] CreateInfo {
    /** @brief The glTF asset for importing hierarchical scene data.  */
    const gltf::Asset& gltf_asset;

    /** @brief The staging model to copy to device-local memory. */
    const StagingModel& staging_model;

    /**
     * @brief The descriptor set layout for all model materials.
     * @note Because this project has not yet implemented dynamic pipeline generation, all model materials must conform
     *       to a fixed descriptor set layout that requires PBR base color, metallic-roughness, and normal textures.
     */
    vk::DescriptorSetLayout material_descriptor_set_layout;

    /**
     * @brief The anisotropy for sampling textures.
     * @note A value of @c std::nullopt indicates this feature is not enabled.
     */
    std::optional<float> sampler_anisotropy;
  };

  /**
   * @brief Creates a @ref Model.
   * @param allocator The allocator for creating device-local buffers and images.
   * @param command_buffer The command buffer for recording copy commands.
   * @param create_info @copybrief Model::CreateInfo
   * @throws std::runtime_error Thrown if @ref Model::CreateInfo::gltf_asset does not contain scene data.
   * @warning The caller is responsible for submitting @p command_buffer to a Vulkan queue to begin execution.
   */
  Model(const vma::Allocator& allocator, vk::CommandBuffer command_buffer, const CreateInfo& create_info);

  /**
   * @brief Updates each node in the model.
   * @details This function traverses the node hierarchy, updates the global transform for each node, and invokes a node
   *          visitor on each node to allow the caller to respond to updates after the node and all its children have
   *          been updated (i.e., post-order traversal).
   * @tparam Fn The callable function type for @p node_visitor.
   * @param node_visitor The function to be invoked on each node after its entire subtree has been updated.
   */
  template <std::invocable<const Node&> Fn>
  void Update(Fn&& node_visitor) {
    for (auto* const root_node : root_nodes_) {
      assert(root_node != nullptr);  // guaranteed by root node construction
      Update(*root_node, root_transform_, std::forward<Fn>(node_visitor));
    }
  }

  /**
   * @brief Records draw commands to render visible meshes in the model.
   * @details This function traverses the node hierarchy and records draw commands for each visible mesh.
   * @param command_buffer The command buffer for recording draw commands.
   * @param pipeline_layout The pipeline layout for binding push constants and descriptor sets.
   * @param view_frustum The camera view frustum for determining mesh visibility.
   * @warning The caller is responsible for submitting @p command_buffer to a Vulkan queue to begin execution.
   */
  void Render(vk::CommandBuffer command_buffer,
              vk::PipelineLayout pipeline_layout,
              const ViewFrustum& view_frustum) const;

private:
  template <std::invocable<const Node&> Fn>
  static void Update(Node& node, const glm::mat4& parent_transform, Fn&& node_visitor) {
    node.global_transform = parent_transform * node.local_transform;

    for (const auto& child_node : node.children) {
      assert(child_node != nullptr);  // guaranteed by node construction
      Update(*child_node, node.global_transform, std::forward<Fn>(node_visitor));
    }

    std::forward<Fn>(node_visitor)(node);  // visit node after its children have been updated
  }

  using Material = pbr_metallic_roughness::Material;

  std::vector<vk::UniqueSampler> samplers_;
  std::vector<std::unique_ptr<const Material>> materials_;
  std::optional<DescriptorPool> material_descriptor_pool_;  // guaranteed to be valid upon model construction
  std::vector<std::unique_ptr<const Mesh>> meshes_;
  std::vector<std::unique_ptr<const Light>> lights_;
  std::vector<std::unique_ptr<Node>> nodes_;
  std::vector<Node*> root_nodes_;
  glm::mat4 root_transform_{1.0f};
};

}  // namespace vktf

module :private;

namespace vktf {

namespace {

// =====================================================================================================================
// Utilities
// =====================================================================================================================

using Severity = Log::Severity;

template <typename Key, typename Value>
using GltfResourceMap = std::unordered_map<const Key*, Value>;

template <typename Key, typename Value>
  requires std::default_initializable<Value>
const Value& Get(const Key* const gltf_element, const GltfResourceMap<Key, Value>& gltf_resource_map) {
  if (gltf_element == nullptr) {
    static const Value kDefaultValue;
    return kDefaultValue;
  }
  const auto iterator = gltf_resource_map.find(gltf_element);
  assert(iterator != gltf_resource_map.cend());  // resource maps are constructed in advance with all known glTF keys
  return iterator->second;
}

template <typename Key, typename Value>
std::vector<Value> GetValues(GltfResourceMap<Key, Value>&& gltf_resource_map) {
  // clang-format off
  auto values = gltf_resource_map
                | std::views::values
                | std::views::filter([](auto& value) { return static_cast<bool>(value); })  // skip unsupported values
                | std::views::as_rvalue
                | std::ranges::to<std::vector>();
  // clang-format on
  gltf_resource_map.clear();  // empty resource map after its values have been moved
  return values;
}

template <typename T>
  requires std::convertible_to<decltype(T::name), std::optional<std::string>>
std::string GetName(const T& gltf_element) {
  static constexpr std::string_view kDefaultName = "undefined";
  return gltf_element.name.value_or(std::string{kDefaultName});
}

// =====================================================================================================================
// Samplers
// =====================================================================================================================

vk::UniqueSampler CreateSampler(const vk::Device device,
                                const gltf::Sampler& gltf_sampler,
                                const std::optional<float> max_anisotropy) {
  const auto& [_, mag_filter, min_filter, mipmap_mode, address_mode_u, address_mode_v] = gltf_sampler;

  return device.createSamplerUnique(
      vk::SamplerCreateInfo{.magFilter = mag_filter,
                            .minFilter = min_filter,
                            .mipmapMode = mipmap_mode,
                            .addressModeU = address_mode_u,
                            .addressModeV = address_mode_v,
                            .anisotropyEnable = static_cast<vk::Bool32>(max_anisotropy.has_value()),
                            .maxAnisotropy = max_anisotropy.value_or(0.0f),
                            .maxLod = vk::LodClampNone});
}

GltfResourceMap<gltf::Sampler, vk::UniqueSampler> CreateSamplers(const vk::Device device,
                                                                 const std::vector<gltf::UniqueSampler>& gltf_samplers,
                                                                 const std::optional<float> max_anisotropy) {
  return gltf_samplers  //
         | std::views::transform([device, max_anisotropy](const auto& gltf_sampler) {
             assert(gltf_sampler != nullptr);  // guaranteed by glTF asset construction
             return std::pair{gltf_sampler.get(), CreateSampler(device, *gltf_sampler, max_anisotropy)};
           })
         | std::ranges::to<std::unordered_map>();
}

vk::Sampler GetSampler(const gltf::Texture* const gltf_texture,
                       const GltfResourceMap<gltf::Sampler, vk::UniqueSampler>& samplers) {
  assert(gltf_texture != nullptr);           // guaranteed by staging material construction
  assert(gltf_texture->sampler != nullptr);  // guaranteed by glTF texture construction
  return *Get(gltf_texture->sampler, samplers);
}

// =====================================================================================================================
// KTX Textures
// =====================================================================================================================

using KtxTextureFuture = std::shared_future<ktx::UniqueKtxTexture2>;

const std::optional<std::filesystem::path>& GetKtxFilepath(const gltf::Texture* const gltf_texture, Log& log) {
  static const std::optional<std::filesystem::path> kInvalidKtxFilepath = std::nullopt;
  if (gltf_texture == nullptr) return kInvalidKtxFilepath;

  const auto& ktx_filepath = gltf_texture->filepath;
  if (!ktx_filepath.has_value()) {
    log(Severity::kError) << std::format("Failed to get KTX filepath for texture {}", GetName(*gltf_texture));
    return kInvalidKtxFilepath;
  }

  // exclude raw image files because they do not encode what color space to be rendered in and supporting that scenario
  // requires coupling material and texture creation which introduces an unnecessary dependency for a suboptimal format
  if (static constexpr std::string_view kKtx2Extension = ".ktx2"; ktx_filepath->extension() != kKtx2Extension) {
    log(Severity::kError) << std::format("Failed to get KTX filepath for texture {} with bad file extension {}",
                                         GetName(*gltf_texture),
                                         ktx_filepath->extension().string());
    return kInvalidKtxFilepath;
  }

  return ktx_filepath;
}

ktx::UniqueKtxTexture2 CreateKtxTexture(const gltf::Texture* const gltf_texture,
                                        const vk::PhysicalDeviceFeatures& physical_device_features,
                                        Log& log) {
  return GetKtxFilepath(gltf_texture, log)
      .transform([&physical_device_features, &log](const auto& ktx_filepath) {
        return ktx::Load(ktx_filepath, physical_device_features, log);
      })
      .value_or(ktx::UniqueKtxTexture2{nullptr, nullptr});
}

GltfResourceMap<gltf::Texture, KtxTextureFuture> CreateKtxTexturesAsync(
    const std::vector<gltf::UniqueTexture>& gltf_textures,
    const vk::PhysicalDeviceFeatures& physical_device_features,
    Log& log) {
  return gltf_textures  //
         | std::views::transform([&physical_device_features, &log](const auto& gltf_texture) {
             assert(gltf_texture != nullptr);  // guaranteed by glTF asset construction
             const auto* const gltf_texture_ptr = gltf_texture.get();
             auto future = std::async(std::launch::async,
                                      CreateKtxTexture,
                                      gltf_texture_ptr,
                                      physical_device_features,
                                      std::ref(log));
             return std::pair{gltf_texture_ptr, future.share()};
           })
         | std::ranges::to<std::unordered_map>();
}

const ktx::UniqueKtxTexture2& GetKtxTexture(
    const gltf::Texture* const gltf_texture,
    const GltfResourceMap<gltf::Texture, KtxTextureFuture>& ktx_texture_futures) {
  const auto& ktx_texture_future = Get(gltf_texture, ktx_texture_futures);
  if (!ktx_texture_future.valid()) {
    static constexpr ktx::UniqueKtxTexture2 kInvalidKtxTexture{nullptr, nullptr};
    return kInvalidKtxTexture;
  }
  return ktx_texture_future.get();
}

// =====================================================================================================================
// Staging Materials
// =====================================================================================================================

using namespace pbr_metallic_roughness;

StagingModel::Material CreateStagingMaterial(
    const vma::Allocator& allocator,
    const gltf::Material& gltf_material,
    const GltfResourceMap<gltf::Texture, KtxTextureFuture>& ktx_texture_futures,
    Log& log) {
  const auto& [_, pbr_metallic_roughness, normal_scale, normal_texture] = gltf_material;

  if (!pbr_metallic_roughness.has_value()) {
    log(Severity::kError) << std::format(
        "Failed to create material {} because it does not support PBR metallic-roughness properties",
        GetName(gltf_material));
    return std::nullopt;  // TODO: add support for non-PBR metallic-roughness materials
  }

  const auto& [base_color_factor, base_color_texture, metallic_factor, roughness_factor, metallic_roughness_texture] =
      *pbr_metallic_roughness;

  const auto& base_color_ktx_texture = GetKtxTexture(base_color_texture, ktx_texture_futures);
  const auto& metallic_roughness_ktx_texture = GetKtxTexture(metallic_roughness_texture, ktx_texture_futures);
  const auto& normal_ktx_texture = GetKtxTexture(normal_texture, ktx_texture_futures);

  for (const auto& [texture_name, ktx_texture] : std::views::zip(
           std::array{"base color", "metallic-roughness", "normal"},
           std::array{base_color_ktx_texture.get(), metallic_roughness_ktx_texture.get(), normal_ktx_texture.get()})) {
    if (ktx_texture == nullptr) {
      log(Severity::kError) << std::format("Failed to create material {} with missing {} texture",
                                           GetName(gltf_material),
                                           texture_name);
      return std::nullopt;  // TODO: add support for optional material textures
    }
  }

  return StagingMaterial{allocator,
                         StagingMaterial::CreateInfo{
                             .material_properties = MaterialProperties{.base_color_factor = base_color_factor,
                                                                       .metallic_roughness_factor =
                                                                           glm::vec2{metallic_factor, roughness_factor},
                                                                       .normal_scale = normal_scale},
                             .base_color_ktx_texture = *base_color_ktx_texture,
                             .metallic_roughness_ktx_texture = *metallic_roughness_ktx_texture,
                             .normal_ktx_texture = *normal_ktx_texture}};
}

GltfResourceMap<gltf::Material, StagingModel::Material> CreateStagingMaterials(
    const vma::Allocator& allocator,
    const std::vector<gltf::UniqueMaterial>& gltf_materials,
    const GltfResourceMap<gltf::Texture, KtxTextureFuture>& ktx_texture_futures,
    Log& log) {
  return gltf_materials  //
         | std::views::transform([&allocator, &ktx_texture_futures, &log](const auto& gltf_material) {
             assert(gltf_material != nullptr);  // guaranteed by glTF asset construction
             return std::pair{gltf_material.get(),
                              CreateStagingMaterial(allocator, *gltf_material, ktx_texture_futures, log)};
           })
         | std::ranges::to<std::unordered_map>();
}

// =====================================================================================================================
// Materials
// =====================================================================================================================

using UniqueMaterial = std::unique_ptr<const Material>;

std::uint32_t CountSupportedMaterials(std::ranges::constant_range auto&& staging_materials) {
  const auto material_count = std::ranges::count_if(staging_materials, [](const auto& staging_material) {
    return staging_material.has_value();
  });
  return static_cast<std::uint32_t>(material_count);
}

DescriptorPool CreateMaterialDescriptorPool(
    const vk::Device device,
    const GltfResourceMap<const gltf::Material, StagingModel::Material>& staging_materials,
    const vk::DescriptorSetLayout material_descriptor_set_layout) {
  static constexpr std::uint32_t kImagesPerMaterial = 3;  // base color, metallic-roughness, normal

  const auto material_count = CountSupportedMaterials(staging_materials | std::views::values);
  const std::vector descriptor_pool_sizes{
      vk::DescriptorPoolSize{.type = vk::DescriptorType::eUniformBuffer, .descriptorCount = material_count},
      vk::DescriptorPoolSize{.type = vk::DescriptorType::eCombinedImageSampler,
                             .descriptorCount = kImagesPerMaterial * material_count}};

  return DescriptorPool{device,
                        DescriptorPool::CreateInfo{.descriptor_pool_sizes = descriptor_pool_sizes,
                                                   .descriptor_set_layout = material_descriptor_set_layout,
                                                   .descriptor_set_count = material_count}};
}

UniqueMaterial CreateMaterial(const vma::Allocator& allocator,
                              const vk::CommandBuffer command_buffer,
                              const gltf::Material& gltf_material,
                              const StagingMaterial& staging_material,
                              const GltfResourceMap<gltf::Sampler, vk::UniqueSampler>& samplers,
                              const vk::DescriptorSet descriptor_set) {
  const auto& pbr_metallic_roughness = gltf_material.pbr_metallic_roughness;
  assert(pbr_metallic_roughness.has_value());  // guaranteed by staging material construction

  return std::make_unique<const Material>(
      allocator,
      command_buffer,
      Material::CreateInfo{
          .staging_material = staging_material,
          .base_color_sampler = GetSampler(pbr_metallic_roughness->base_color_texture, samplers),
          .metallic_roughness_sampler = GetSampler(pbr_metallic_roughness->metallic_roughness_texture, samplers),
          .normal_sampler = GetSampler(gltf_material.normal_texture, samplers),
          .descriptor_set = descriptor_set});
}

GltfResourceMap<gltf::Material, UniqueMaterial> CreateMaterials(
    const vma::Allocator& allocator,
    const vk::CommandBuffer command_buffer,
    const GltfResourceMap<gltf::Material, StagingModel::Material>& staging_materials,
    const GltfResourceMap<gltf::Sampler, vk::UniqueSampler>& samplers,
    const std::vector<vk::DescriptorSet>& descriptor_sets) {
  // descriptor sets are allocated based on the number of supported materials
  assert(descriptor_sets.size() == CountSupportedMaterials(staging_materials | std::views::values));
  std::size_t descriptor_set_index = 0;

  return staging_materials  //
         | std::views::transform(
             [=, &allocator, &samplers, &descriptor_sets, &descriptor_set_index](const auto& key_value_pair) {
               const auto& [gltf_material, staging_material] = key_value_pair;
               assert(gltf_material != nullptr);  // guaranteed by staging material construction

               if (!staging_material.has_value()) {
                 return std::pair{gltf_material, UniqueMaterial{nullptr}};
               }

               return std::pair{gltf_material,
                                CreateMaterial(allocator,
                                               command_buffer,
                                               *gltf_material,
                                               *staging_material,
                                               samplers,
                                               descriptor_sets[descriptor_set_index++])};
             })
         | std::ranges::to<std::unordered_map>();
}

// =====================================================================================================================
// Staging Meshes
// =====================================================================================================================

using PositionAttribute = gltf::Attributes::Position;
using NormalAttribute = gltf::Attributes::Normal;
using TangentAttribute = gltf::Attributes::Tangent;
using TexCoord0Attribute = gltf::Attributes::TexCoord0;

std::vector<Vertex> CreateVertices(const PositionAttribute::Data& position_data,
                                   const NormalAttribute::Data& normal_data,
                                   const TangentAttribute::Data& tangent_data,
                                   const TexCoord0Attribute::Data& texcoord_0_data) {
  return std::views::zip_transform(
             [](const auto position, const auto& normal, const auto& tangent, const auto& texcoord_0) {
               return Vertex{.position = position, .normal = normal, .tangent = tangent, .texcoord_0 = texcoord_0};
             },
             position_data,
             normal_data,
             tangent_data,
             texcoord_0_data)
         | std::ranges::to<std::vector>();
}

std::optional<std::string_view> FindMissingAttributeName(const gltf::Attributes& attributes) {
  const auto& [_, normal_attribute, tangent_attribute, texcoord_0_attribute] = attributes;
  if (!normal_attribute.data.has_value()) return NormalAttribute::kName;
  if (!tangent_attribute.data.has_value()) return TangentAttribute::kName;
  if (!texcoord_0_attribute.data.has_value()) return TexCoord0Attribute::kName;
  return std::nullopt;
}

std::optional<StagingPrimitive> CreateStagingPrimitive(
    const vma::Allocator& allocator,
    const gltf::Mesh& gltf_mesh,
    const std::size_t primitive_index,
    const GltfResourceMap<gltf::Material, StagingModel::Material>& staging_materials,
    Log& log) {
  assert(primitive_index < gltf_mesh.primitives.size());
  const auto& [attributes, indices_variant, gltf_material] = gltf_mesh.primitives[primitive_index];

  if (const auto attribute_name = FindMissingAttributeName(attributes); attribute_name.has_value()) {
    log(Severity::kError) << std::format("Failed to create mesh primitive {}[{}] with missing {} attribute",
                                         GetName(gltf_mesh),
                                         primitive_index,
                                         *attribute_name);
    return std::nullopt;  // TODO: add support for optional vertex attributes
  }

  if (!indices_variant.has_value()) {
    log(Severity::kError) << std::format("Failed to create mesh primitive {}[{}] with missing indices",
                                         GetName(gltf_mesh),
                                         primitive_index);
    return std::nullopt;  // TODO: add support for non-indexed mesh primitives
  }

  if (const auto& staging_material = Get(gltf_material, staging_materials); !staging_material.has_value()) {
    log(Severity::kError) << std::format("Failed to create mesh primitive {}[{}] with unsupported material",
                                         GetName(gltf_mesh),
                                         primitive_index);
    return std::nullopt;  // TODO: add support for default materials
  }

  return std::visit(
      [&allocator, &attributes]<typename Indices>(const Indices& indices) {
        const auto& [position_attribute, normal_attribute, tangent_attribute, texcoord_0_attribute] = attributes;
        using IndexType = typename Indices::value_type;

        return StagingPrimitive{
            allocator,
            StagingPrimitive::CreateInfo<IndexType>{.vertices = CreateVertices(position_attribute.data,
                                                                               *normal_attribute.data,
                                                                               *tangent_attribute.data,
                                                                               *texcoord_0_attribute.data),
                                                    .indices = indices}};
      },
      *indices_variant);
}

StagingModel::Mesh CreateStagingMesh(const vma::Allocator& allocator,
                                     const gltf::Mesh& gltf_mesh,
                                     const GltfResourceMap<gltf::Material, StagingModel::Material>& staging_materials,
                                     Log& log) {
  return std::views::iota(0uz, gltf_mesh.primitives.size())
         | std::views::transform([&allocator, &gltf_mesh, &staging_materials, &log](const auto primitive_index) {
             return CreateStagingPrimitive(allocator, gltf_mesh, primitive_index, staging_materials, log);
           })
         | std::ranges::to<std::vector>();
}

GltfResourceMap<gltf::Mesh, StagingModel::Mesh> CreateStagingMeshes(
    const vma::Allocator& allocator,
    const std::vector<gltf::UniqueMesh>& gltf_meshes,
    const GltfResourceMap<gltf::Material, StagingModel::Material>& staging_materials,
    Log& log) {
  return gltf_meshes  //
         | std::views::transform([&allocator, &staging_materials, &log](const auto& gltf_mesh) {
             assert(gltf_mesh != nullptr);  // guaranteed by glTF asset construction
             return std::pair{gltf_mesh.get(), CreateStagingMesh(allocator, *gltf_mesh, staging_materials, log)};
           })
         | std::ranges::to<std::unordered_map>();
}

// =====================================================================================================================
// Meshes
// =====================================================================================================================

using UniqueMesh = std::unique_ptr<const Mesh>;

BoundingBox Union(const BoundingBox& lhs, const BoundingBox& rhs) {
  return BoundingBox{.min = glm::min(lhs.min, rhs.min), .max = glm::max(lhs.max, rhs.max)};
}

UniqueMesh CreateMesh(const vma::Allocator& allocator,
                      const vk::CommandBuffer command_buffer,
                      const gltf::Mesh& gltf_mesh,
                      const StagingModel::Mesh& staging_mesh,
                      const GltfResourceMap<gltf::Material, UniqueMaterial>& materials) {
  assert(gltf_mesh.primitives.size() == staging_mesh.size());  // guaranteed by staging mesh construction

  std::vector<Primitive> primitives;
  primitives.reserve(gltf_mesh.primitives.size());

  BoundingBox bounding_box{.min = glm::vec3{std::numeric_limits<float>::max()},
                           .max = glm::vec3{std::numeric_limits<float>::min()}};

  for (const auto& [gltf_primitive, staging_primitive] : std::views::zip(gltf_mesh.primitives, staging_mesh)) {
    if (!staging_primitive.has_value()) continue;  // skip unsupported mesh primitive

    const auto& primitive_bounding_box = gltf_primitive.attributes.position.bounding_box;
    bounding_box = Union(bounding_box, primitive_bounding_box);

    const auto& material = Get(gltf_primitive.material, materials);
    assert(material != nullptr);  // guaranteed by staging primitive construction

    primitives.emplace_back(allocator,
                            command_buffer,
                            Primitive::CreateInfo{.staging_primitive = *staging_primitive,
                                                  .material_descriptor_set = material->descriptor_set()});
  }

  return primitives.empty() ? nullptr : std::make_unique<Mesh>(std::move(primitives), bounding_box);
}

GltfResourceMap<gltf::Mesh, UniqueMesh> CreateMeshes(
    const vma::Allocator& allocator,
    const vk::CommandBuffer command_buffer,
    const GltfResourceMap<gltf::Mesh, StagingModel::Mesh>& staging_meshes,
    const GltfResourceMap<gltf::Material, UniqueMaterial>& materials) {
  return staging_meshes  //
         | std::views::transform([&allocator, command_buffer, &materials](const auto& key_value_pair) {
             const auto& [gltf_mesh, staging_mesh] = key_value_pair;
             return std::pair{gltf_mesh, CreateMesh(allocator, command_buffer, *gltf_mesh, staging_mesh, materials)};
           })
         | std::ranges::to<std::unordered_map>();
}

// =====================================================================================================================
// Lights
// =====================================================================================================================

using Light = Model::Light;
using UniqueLight = std::unique_ptr<const Light>;

Light::Type GetLightType(const gltf::Light::Type& gltf_light_type) {
  switch (gltf_light_type) {
    using enum gltf::Light::Type;
    case kDirectional:
      return Light::Type::kDirectional;
    case kPoint:
      return Light::Type::kPoint;
    default:
      std::unreachable();
  }
}

GltfResourceMap<gltf::Light, UniqueLight> CreateLights(const std::vector<gltf::UniqueLight>& gltf_lights) {
  return gltf_lights  //
         | std::views::transform([](const auto& gltf_light) {
             assert(gltf_light != nullptr);  // guaranteed by glTF asset construction
             const auto& [_, color, type] = *gltf_light;
             return std::pair{gltf_light.get(), std::make_unique<const Light>(color, GetLightType(type))};
           })
         | std::ranges::to<std::unordered_map>();
}

// =====================================================================================================================
// Nodes
// =====================================================================================================================

using Node = Model::Node;
using UniqueNode = std::unique_ptr<Node>;

constexpr glm::mat4 kIdentityTransform{1.0f};

std::vector<Node*> GetChildren(const gltf::Node& gltf_parent_node,
                               const GltfResourceMap<gltf::Node, UniqueNode>& nodes) {
  return gltf_parent_node.children  //
         | std::views::transform([&nodes](const auto* const gltf_child_node) {
             assert(gltf_child_node != nullptr);  // guaranteed by glTF node construction
             const auto& child_node = Get(gltf_child_node, nodes);
             return child_node.get();
           })
         | std::ranges::to<std::vector>();
}

GltfResourceMap<gltf::Node, UniqueNode> CreateNodes(const std::vector<gltf::UniqueNode>& gltf_nodes,
                                                    const GltfResourceMap<gltf::Mesh, UniqueMesh>& meshes,
                                                    const GltfResourceMap<gltf::Light, UniqueLight>& lights) {
  auto nodes =
      gltf_nodes  //
      | std::views::transform([&meshes, &lights](const auto& gltf_node) {
          assert(gltf_node != nullptr);  // guaranteed by glTF asset construction
          const auto& [name, local_transform, gltf_mesh, gltf_cameras, gltf_light, children] = *gltf_node;
          const auto& mesh = Get(gltf_mesh, meshes);
          const auto& light = Get(gltf_light, lights);
          return std::pair{gltf_node.get(),
                           std::make_unique<Node>(local_transform, kIdentityTransform, mesh.get(), light.get())};
        })
      | std::ranges::to<std::unordered_map>();

  for (auto& [gltf_node, node] : nodes) {
    node->children = GetChildren(*gltf_node, nodes);  // assign children after all nodes have been created
  }

  return nodes;
}

const gltf::Scene& GetDefaultGltfScene(const gltf::Asset& gltf_asset) {
  if (const auto& default_scene = gltf_asset.default_scene; default_scene != nullptr) {
    return *default_scene;
  }
  if (const auto& gltf_scenes = gltf_asset.scenes; !gltf_scenes.empty()) {
    return *gltf_scenes.front();  // use first available scene when no default is specified
  }
  // TODO: glTF files not containing scene data should be treated as as library of individual entities
  throw std::runtime_error{std::format("Failed to get the default scene for glTF asset {}", gltf_asset.name)};
}

std::vector<Node*> GetRootNodes(const gltf::Scene& gltf_scene, const GltfResourceMap<gltf::Node, UniqueNode>& nodes) {
  return gltf_scene.root_nodes  //
         | std::views::transform([&nodes](const auto* const gltf_node) {
             assert(gltf_node != nullptr);  // guaranteed by glTF asset construction
             const auto& root_node = Get(gltf_node, nodes);
             return root_node.get();
           })
         | std::ranges::to<std::vector>();
}

// =====================================================================================================================
// Rendering
// =====================================================================================================================

void Render(const Mesh& mesh,
            const glm::mat4& model_transform,
            const vk::CommandBuffer command_buffer,
            const vk::PipelineLayout pipeline_layout,
            const ViewFrustum& view_frustum) {
  if (const auto& world_bounding_box = Transform(mesh.bounding_box(), model_transform);
      !view_frustum.Intersects(world_bounding_box)) {
    return;  // skip mesh primitive outside the view frustum
  }

  using ModelTransform = decltype(GraphicsPipeline::PushConstants::model_transform);
  command_buffer.pushConstants<ModelTransform>(pipeline_layout,
                                               vk::ShaderStageFlagBits::eVertex,
                                               offsetof(GraphicsPipeline::PushConstants, model_transform),
                                               model_transform);

  for (const auto& primitive : mesh.primitives()) {
    // TODO: avoid per-primitive material descriptor set binding
    command_buffer.bindDescriptorSets(vk::PipelineBindPoint::eGraphics,
                                      pipeline_layout,
                                      1,
                                      primitive.material_descriptor_set(),
                                      nullptr);
    primitive.Render(command_buffer);
  }
}

void Render(const Node& node,
            const vk::CommandBuffer command_buffer,
            const vk::PipelineLayout pipeline_layout,
            const ViewFrustum& view_frustum) {
  if (const auto* const mesh = node.mesh; mesh != nullptr) {
    Render(*mesh, node.global_transform, command_buffer, pipeline_layout, view_frustum);
  }

  for (const auto& child_node : node.children) {
    assert(child_node != nullptr);  // guaranteed by node construction
    Render(*child_node, command_buffer, pipeline_layout, view_frustum);
  }
}

}  // namespace

StagingModel::StagingModel(const vma::Allocator& allocator, const CreateInfo& create_info) {
  const auto& [gltf_asset, physical_device_features, log] = create_info;
  const auto ktx_texture_futures = CreateKtxTexturesAsync(gltf_asset.textures, physical_device_features, log);
  materials_ = CreateStagingMaterials(allocator, gltf_asset.materials, ktx_texture_futures, log);
  meshes_ = CreateStagingMeshes(allocator, gltf_asset.meshes, materials_, log);
}

Model::Model(const vma::Allocator& allocator, const vk::CommandBuffer command_buffer, const CreateInfo& create_info)
    : material_descriptor_pool_{CreateMaterialDescriptorPool(allocator.device(),
                                                             create_info.staging_model.materials(),
                                                             create_info.material_descriptor_set_layout)} {
  const auto& [gltf_asset, staging_model, material_descriptor_set_layout, sampler_anisotropy] = create_info;
  const auto& device = allocator.device();
  const auto& staging_materials = staging_model.materials();
  const auto& staging_meshes = staging_model.meshes();
  const auto& material_descriptor_sets = material_descriptor_pool_->descriptor_sets();

  auto samplers = CreateSamplers(device, gltf_asset.samplers, sampler_anisotropy);
  auto materials = CreateMaterials(allocator, command_buffer, staging_materials, samplers, material_descriptor_sets);
  auto meshes = CreateMeshes(allocator, command_buffer, staging_meshes, materials);
  auto lights = CreateLights(gltf_asset.lights);
  auto nodes = CreateNodes(gltf_asset.nodes, meshes, lights);

  const auto& gltf_scene = GetDefaultGltfScene(gltf_asset);
  root_nodes_ = GetRootNodes(gltf_scene, nodes);
  samplers_ = GetValues(std::move(samplers));
  materials_ = GetValues(std::move(materials));
  meshes_ = GetValues(std::move(meshes));
  lights_ = GetValues(std::move(lights));
  nodes_ = GetValues(std::move(nodes));
}

void Model::Render(const vk::CommandBuffer command_buffer,
                   const vk::PipelineLayout pipeline_layout,
                   const ViewFrustum& view_frustum) const {
  for (const auto* const root_node : root_nodes_) {
    assert(root_node != nullptr);  // guaranteed by root node construction
    vktf::Render(*root_node, command_buffer, pipeline_layout, view_frustum);
  }
}

}  // namespace vktf
