module;

#include <algorithm>
#include <array>
#include <cassert>
#include <concepts>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <format>
#include <future>
#include <iostream>
#include <limits>
#include <memory>
#include <optional>
#include <print>
#include <ranges>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_map>
#include <utility>
#include <vector>

#include <cgltf.h>
#include <vk_mem_alloc.h>
#include <glm/glm.hpp>
#include <glm/gtc/epsilon.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <vulkan/vulkan.hpp>

export module gltf_scene;

import allocator;
import buffer;
import camera;
import command_pool;
import data_view;
import descriptor_pool;
import image;
import material;
import mesh;
import shader_module;
import texture;

namespace vktf {

struct Light {
  glm::vec4 position{0.0f};
  glm::vec4 color{0.0f};  // alpha padding applied to conform to std140 layout requirements
};

struct Node {
  const Mesh* mesh = nullptr;
  const Light* light = nullptr;
  glm::mat4 transform{0.0f};
  std::vector<std::unique_ptr<const Node>> children;
};

export class GltfScene {
public:
  GltfScene(const std::filesystem::path& gltf_filepath,
            const vk::PhysicalDevice physical_device,
            const vk::Bool32 enable_sampler_anisotropy,
            const float max_sampler_anisotropy,
            const vk::Device device,
            const vk::Queue transfer_queue,
            const std::uint32_t transfer_queue_family_index,
            const vk::Extent2D viewport_extent,
            const vk::SampleCountFlagBits msaa_sample_count,
            const vk::RenderPass render_pass,
            const VmaAllocator allocator,
            const std::size_t max_render_frames);

  void Render(const Camera& camera, const std::size_t frame_index, const vk::CommandBuffer command_buffer) const;

private:
  std::vector<std::unique_ptr<Material>> materials_;
  std::vector<vk::UniqueSampler> samplers_;
  std::vector<std::unique_ptr<const Mesh>> meshes_;
  std::vector<std::unique_ptr<const Light>> lights_;
  std::unique_ptr<const Node> root_node_;
  std::vector<HostVisibleBuffer> camera_uniform_buffers_;
  std::vector<HostVisibleBuffer> lights_uniform_buffers_;
  std::optional<DescriptorPool> global_descriptor_pool_;
  std::optional<DescriptorPool> material_descriptor_pool_;
  vk::UniquePipelineLayout graphics_pipeline_layout_;
  vk::UniquePipeline graphics_pipeline_;
};

}  // namespace vktf

module :private;

#pragma region formatters

template <>
struct std::formatter<cgltf_result> : std::formatter<std::string_view> {
  [[nodiscard]] auto format(const cgltf_result gltf_result, auto& format_context) const {
    return std::formatter<std::string_view>::format(to_string(gltf_result), format_context);
  }

private:
  static std::string_view to_string(const cgltf_result gltf_result) noexcept {
    switch (gltf_result) {
      // clang-format off
#define CASE(kValue) case kValue: return #kValue;  // NOLINT(cppcoreguidelines-macro-usage)
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
      CASE(cgltf_result_max_enum)
#undef CASE
      // clang-format on
      default:
        std::unreachable();
    }
  }
};

template <>
struct std::formatter<cgltf_primitive_type> : std::formatter<std::string_view> {
  [[nodiscard]] auto format(const cgltf_primitive_type gltf_primitive_type, auto& format_context) const {
    return std::formatter<std::string_view>::format(to_string(gltf_primitive_type), format_context);
  }

private:
  static std::string_view to_string(const cgltf_primitive_type gltf_primitive_type) noexcept {
    switch (gltf_primitive_type) {
      // clang-format off
#define CASE(kValue) case kValue: return #kValue;  // NOLINT(cppcoreguidelines-macro-usage)
      CASE(cgltf_primitive_type_invalid)
      CASE(cgltf_primitive_type_points)
      CASE(cgltf_primitive_type_lines)
      CASE(cgltf_primitive_type_line_loop)
      CASE(cgltf_primitive_type_line_strip)
      CASE(cgltf_primitive_type_triangles)
      CASE(cgltf_primitive_type_triangle_strip)
      CASE(cgltf_primitive_type_triangle_fan)
      CASE(cgltf_primitive_type_max_enum)
#undef CASE
      // clang-format on
      default:
        std::unreachable();
    }
  }
};

template <>
struct std::formatter<cgltf_light_type> : std::formatter<std::string_view> {
  [[nodiscard]] auto format(const cgltf_light_type gltf_light_type, auto& format_context) const {
    return std::formatter<std::string_view>::format(to_string(gltf_light_type), format_context);
  }

private:
  static std::string_view to_string(const cgltf_light_type gltf_light_type) noexcept {
    switch (gltf_light_type) {
      // clang-format off
#define CASE(kValue) case kValue: return #kValue;  // NOLINT(cppcoreguidelines-macro-usage)
      CASE(cgltf_light_type_invalid)
      CASE(cgltf_light_type_directional)
      CASE(cgltf_light_type_point)
      CASE(cgltf_light_type_spot)
      CASE(cgltf_light_type_max_enum)
#undef CASE
      // clang-format on
      default:
        std::unreachable();
    }
  }
};

#pragma endregion

namespace {

#pragma region utilities

using UniqueGltfData = std::unique_ptr<cgltf_data, decltype(&cgltf_free)>;

UniqueGltfData Load(const std::string& gltf_filepath) {
  static constexpr cgltf_options kDefaultOptions{};
  UniqueGltfData gltf_data{nullptr, cgltf_free};

  if (const auto gltf_result = cgltf_parse_file(&kDefaultOptions, gltf_filepath.c_str(), std::out_ptr(gltf_data));
      gltf_result != cgltf_result_success) {
    throw std::runtime_error{std::format("Failed to parse {} with error {}", gltf_filepath, gltf_result)};
  }
#ifndef NDEBUG
  if (const auto gltf_result = cgltf_validate(gltf_data.get()); gltf_result != cgltf_result_success) {
    throw std::runtime_error{std::format("Failed to validate {} with error {}", gltf_filepath, gltf_result)};
  }
#endif
  if (const auto gltf_result = cgltf_load_buffers(&kDefaultOptions, gltf_data.get(), gltf_filepath.c_str());
      gltf_result != cgltf_result_success) {
    throw std::runtime_error{std::format("Failed to load buffers for {} with error {}", gltf_filepath, gltf_result)};
  }

  return gltf_data;
}

template <typename T>
  requires std::convertible_to<decltype(T::name), const char*>
std::string_view GetName(const T& gltf_element) {
  if (const auto* const name = gltf_element.name; name != nullptr) {
    if (const auto length = std::strlen(name); length > 0) {
      return std::string_view{name, length};
    }
  }
  return "undefined";
}

template <typename Key, typename Value>
const Value& Get(const Key* const key, const std::unordered_map<const Key*, Value>& map) {
  const auto iterator = map.find(key);
  assert(iterator != map.cend());
  return iterator->second;
}

template <typename T, glm::length_t N>
concept VecConstructible = std::constructible_from<glm::vec<N, T>>;

template <typename T, glm::length_t N>
  requires VecConstructible<T, N>
glm::vec<N, T> ToVec(const T (&src_array)[N]) {  // NOLINT(*-c-arrays): defines an explicit conversion from a c-array
  glm::vec<N, T> dst_vec{};
  static_assert(sizeof(src_array) == sizeof(dst_vec));
  std::ranges::copy(src_array, glm::value_ptr(dst_vec));
  return dst_vec;
}

#pragma endregion

#pragma region samplers

using Samplers = std::unordered_map<const cgltf_sampler*, vk::UniqueSampler>;

// filter and address mode values come from https://registry.khronos.org/glTF/specs/2.0/glTF-2.0.html#reference-sampler
enum class SamplerFilter : std::uint16_t {
  kUndefined = 0,
  kNearest = 9728,
  kLinear = 9729,
  kNearestMipmapNearest = 9984,
  kLinearMipmapNearest = 9985,
  kNearestMipmapLinear = 9986,
  kLinearMipmapLinear = 9987
};

enum class SamplerAddressMode : std::uint16_t { kClampToEdge = 33071, kMirroredRepeat = 33648, kRepeat = 10497 };

vk::Filter GetSamplerMagFilter(const cgltf_int gltf_mag_filter) {
  switch (static_cast<SamplerFilter>(gltf_mag_filter)) {
    using enum SamplerFilter;
    case kUndefined:
    case kNearest:
      return vk::Filter::eNearest;
    case kLinear:
      return vk::Filter::eLinear;
    default:
      std::unreachable();
  }
}

std::pair<vk::Filter, vk::SamplerMipmapMode> GetSamplerMinFilterAndMipmapMode(const cgltf_int gltf_min_filter) {
  switch (static_cast<SamplerFilter>(gltf_min_filter)) {
    using enum SamplerFilter;
    case kUndefined:
    case kNearest:
    case kNearestMipmapNearest:
      return std::pair{vk::Filter::eNearest, vk::SamplerMipmapMode::eNearest};
    case kLinear:
    case kLinearMipmapNearest:
      return std::pair{vk::Filter::eLinear, vk::SamplerMipmapMode::eNearest};
    case kNearestMipmapLinear:
      return std::pair{vk::Filter::eNearest, vk::SamplerMipmapMode::eLinear};
    case kLinearMipmapLinear:
      return std::pair{vk::Filter::eLinear, vk::SamplerMipmapMode::eLinear};
    default:
      std::unreachable();
  }
}

vk::SamplerAddressMode GetSamplerAddressMode(const cgltf_int gltf_wrap_mode) {
  switch (static_cast<SamplerAddressMode>(gltf_wrap_mode)) {
    using enum SamplerAddressMode;
    case kClampToEdge:
      return vk::SamplerAddressMode::eClampToEdge;
    case kMirroredRepeat:
      return vk::SamplerAddressMode::eMirroredRepeat;
    case kRepeat:
      return vk::SamplerAddressMode::eRepeat;
    default:
      std::unreachable();
  }
}

vk::UniqueSampler CreateSampler(const cgltf_sampler& gltf_sampler,
                                const vk::Device device,
                                const vk::Bool32 enable_anisotropy,
                                const float max_anisotropy) {
  const auto [min_filter, mipmap_mode] = GetSamplerMinFilterAndMipmapMode(gltf_sampler.min_filter);

  return device.createSamplerUnique(vk::SamplerCreateInfo{.magFilter = GetSamplerMagFilter(gltf_sampler.mag_filter),
                                                          .minFilter = min_filter,
                                                          .mipmapMode = mipmap_mode,
                                                          .addressModeU = GetSamplerAddressMode(gltf_sampler.wrap_s),
                                                          .addressModeV = GetSamplerAddressMode(gltf_sampler.wrap_t),
                                                          .anisotropyEnable = enable_anisotropy,
                                                          .maxAnisotropy = max_anisotropy,
                                                          .maxLod = vk::LodClampNone});
}

vk::UniqueSampler CreateDefaultSampler(const vk::Device device,
                                       const vk::Bool32 enable_anisotropy,
                                       const float max_anisotropy) {
  return device.createSamplerUnique(vk::SamplerCreateInfo{.magFilter = vk::Filter::eLinear,
                                                          .minFilter = vk::Filter::eLinear,
                                                          .mipmapMode = vk::SamplerMipmapMode::eLinear,
                                                          .addressModeU = vk::SamplerAddressMode::eRepeat,
                                                          .addressModeV = vk::SamplerAddressMode::eRepeat,
                                                          .anisotropyEnable = enable_anisotropy,
                                                          .maxAnisotropy = max_anisotropy,
                                                          .maxLod = vk::LodClampNone});
}

#pragma endregion

#pragma region materials

struct MaterialFutures {
  const cgltf_material* gltf_material = nullptr;
  std::future<std::optional<vktf::Texture>> base_color_texture_future;
  std::future<std::optional<vktf::Texture>> metallic_roughness_texture_future;
  std::future<std::optional<vktf::Texture>> normal_texture_future;
};

struct MaterialProperties {
  glm::vec4 base_color_factor{0.0f};
  glm::vec2 metallic_roughness_factor{0.0f};
  float normal_scale = 0.0f;
};

using Materials = std::unordered_map<const cgltf_material*, std::unique_ptr<vktf::Material>>;

std::optional<vktf::Texture> CreateTexture(const cgltf_texture_view& gltf_texture_view,
                                           const vktf::ColorSpace color_space,
                                           const std::filesystem::path& gltf_directory,
                                           const vk::PhysicalDevice physical_device,
                                           const Samplers& samplers) {
  const auto* const gltf_texture = gltf_texture_view.texture;
  if (gltf_texture == nullptr) return std::nullopt;

  const auto* const gltf_image = gltf_texture->has_basisu == 0 ? gltf_texture->image : gltf_texture->basisu_image;
  if (gltf_image == nullptr) {
    std::println(std::cerr, "No image source for texture {}", GetName(*gltf_texture));
    return std::nullopt;
  }

  const std::filesystem::path gltf_image_uri = gltf_image->uri == nullptr ? "" : gltf_image->uri;
  if (gltf_image_uri.empty()) {
    std::println(std::cerr, "No URI for image {}", GetName(*gltf_image));
    return std::nullopt;
  }

  const auto& sampler = Get(gltf_texture->sampler, samplers);
  return vktf::Texture{gltf_directory / gltf_image_uri, color_space, physical_device, *sampler};
}

MaterialFutures CreateMaterialFutures(const cgltf_material& gltf_material,
                                      const std::filesystem::path& gltf_directory,
                                      const vk::PhysicalDevice physical_device,
                                      const Samplers& samplers) {
  if (gltf_material.has_pbr_metallic_roughness == 0) {
    return {};  // TODO: add support for non PBR metallic-roughness materials
  }

  const auto& pbr_metallic_roughness = gltf_material.pbr_metallic_roughness;

  return MaterialFutures{
      .gltf_material = &gltf_material,
      .base_color_texture_future = std::async(std::launch::async,
                                              CreateTexture,
                                              std::cref(pbr_metallic_roughness.base_color_texture),
                                              vktf::ColorSpace::kSrgb,
                                              std::cref(gltf_directory),
                                              physical_device,
                                              std::cref(samplers)),
      .metallic_roughness_texture_future = std::async(std::launch::async,
                                                      CreateTexture,
                                                      std::cref(pbr_metallic_roughness.metallic_roughness_texture),
                                                      vktf::ColorSpace::kLinear,
                                                      std::cref(gltf_directory),
                                                      physical_device,
                                                      std::cref(samplers)),
      .normal_texture_future = std::async(std::launch::async,
                                          CreateTexture,
                                          std::cref(gltf_material.normal_texture),
                                          vktf::ColorSpace::kLinear,
                                          std::cref(gltf_directory),
                                          physical_device,
                                          std::cref(samplers))};
}

std::unique_ptr<vktf::Material> CreateMaterial(MaterialFutures& material_futures,
                                               const vk::Device device,
                                               const vk::CommandBuffer command_buffer,
                                               const VmaAllocator allocator,
                                               std::vector<vktf::HostVisibleBuffer>& staging_buffers) {
  auto& [gltf_material, base_color_texture_future, metallic_roughness_texture_future, normal_texture_future] =
      material_futures;

  auto maybe_base_color_texture = base_color_texture_future.get();
  auto maybe_metallic_roughness_texture = metallic_roughness_texture_future.get();
  auto maybe_normal_texture = normal_texture_future.get();

  if (std::ranges::any_of(
          std::array{&maybe_base_color_texture, &maybe_metallic_roughness_texture, &maybe_normal_texture},
          [](const auto* const maybe_ktx_texture) { return !maybe_ktx_texture->has_value(); })) {
    std::println(std::cerr,
                 "Failed to create material {} because it's missing required PBR metallic-roughness textures",
                 GetName(*gltf_material));
    return nullptr;  // TODO: add support for optional material textures
  }

  auto& base_color_texture = *maybe_base_color_texture;
  auto& metallic_roughness_texture = *maybe_metallic_roughness_texture;
  auto& normal_texture = *maybe_normal_texture;

  base_color_texture.CreateImage(device, command_buffer, allocator, staging_buffers);
  metallic_roughness_texture.CreateImage(device, command_buffer, allocator, staging_buffers);
  normal_texture.CreateImage(device, command_buffer, allocator, staging_buffers);

  const auto& pbr_metallic_roughness = gltf_material->pbr_metallic_roughness;
  const MaterialProperties material_properties{
      .base_color_factor = ToVec(pbr_metallic_roughness.base_color_factor),
      .metallic_roughness_factor =
          glm::vec2{pbr_metallic_roughness.metallic_factor, pbr_metallic_roughness.roughness_factor},
      .normal_scale = gltf_material->normal_texture.scale};

  const auto& properties_staging_buffer =
      staging_buffers.emplace_back(vktf::CreateStagingBuffer<MaterialProperties>(material_properties, allocator));

  return std::make_unique<vktf::Material>(
      vktf::CreateBuffer(properties_staging_buffer, vk::BufferUsageFlagBits::eUniformBuffer, command_buffer, allocator),
      std::move(base_color_texture),
      std::move(metallic_roughness_texture),
      std::move(normal_texture));
}

std::optional<vktf::DescriptorPool> CreateMaterialDescriptorPool(const vk::Device device,
                                                                 const std::uint32_t material_count) {
  static constexpr std::uint32_t kImagesPerMaterial = 3;
  const std::array descriptor_pool_sizes{
      vk::DescriptorPoolSize{.type = vk::DescriptorType::eUniformBuffer, .descriptorCount = material_count},
      vk::DescriptorPoolSize{.type = vk::DescriptorType::eCombinedImageSampler,
                             .descriptorCount = kImagesPerMaterial * material_count}};

  static constexpr std::array kDescriptorSetLayoutBindings{
      vk::DescriptorSetLayoutBinding{.binding = 0,  // material properties
                                     .descriptorType = vk::DescriptorType::eUniformBuffer,
                                     .descriptorCount = 1,
                                     .stageFlags = vk::ShaderStageFlagBits::eFragment},
      vk::DescriptorSetLayoutBinding{.binding = 1,  // base color, metallic-roughness, normal
                                     .descriptorType = vk::DescriptorType::eCombinedImageSampler,
                                     .descriptorCount = kImagesPerMaterial,
                                     .stageFlags = vk::ShaderStageFlagBits::eFragment}};

  return vktf::DescriptorPool{device, descriptor_pool_sizes, kDescriptorSetLayoutBindings, material_count};
}

void UpdateMaterialDescriptorSets(const vk::Device device,
                                  const vktf::DescriptorPool& material_descriptor_pool,
                                  Materials& materials) {
  const auto& descriptor_sets = material_descriptor_pool.descriptor_sets();
  assert(materials.size() == descriptor_sets.size());

  std::vector<vk::DescriptorBufferInfo> descriptor_buffer_infos;
  descriptor_buffer_infos.reserve(materials.size());

  std::vector<std::vector<vk::DescriptorImageInfo>> descriptor_image_infos;
  descriptor_image_infos.reserve(materials.size());

  static constexpr auto kDescriptorsPerMaterial = 2;
  std::vector<vk::WriteDescriptorSet> descriptor_set_writes;
  descriptor_set_writes.reserve(kDescriptorsPerMaterial * materials.size());

  for (const auto& [material, descriptor_set] : std::views::zip(materials | std::views::values, descriptor_sets)) {
    if (material == nullptr) continue;  // TODO: avoid creating descriptor set for unsupported material

    const auto& base_color_texture = material->base_color_texture;
    const auto& metallic_roughness_texture = material->metallic_roughness_texture;
    const auto& normal_texture = material->normal_texture;
    const auto& properties_buffer = material->properties_buffer;

    const auto& descriptor_buffer_info = descriptor_buffer_infos.emplace_back(
        vk::DescriptorBufferInfo{.buffer = *properties_buffer, .range = vk::WholeSize});

    descriptor_set_writes.push_back(vk::WriteDescriptorSet{.dstSet = descriptor_set,
                                                           .dstBinding = 0,
                                                           .dstArrayElement = 0,
                                                           .descriptorCount = 1,
                                                           .descriptorType = vk::DescriptorType::eUniformBuffer,
                                                           .pBufferInfo = &descriptor_buffer_info});

    const auto& descriptor_image_info = descriptor_image_infos.emplace_back(
        std::vector{vk::DescriptorImageInfo{.sampler = base_color_texture.sampler(),
                                            .imageView = base_color_texture.image_view(),
                                            .imageLayout = vk::ImageLayout::eShaderReadOnlyOptimal},
                    vk::DescriptorImageInfo{.sampler = metallic_roughness_texture.sampler(),
                                            .imageView = metallic_roughness_texture.image_view(),
                                            .imageLayout = vk::ImageLayout::eShaderReadOnlyOptimal},
                    vk::DescriptorImageInfo{.sampler = normal_texture.sampler(),
                                            .imageView = normal_texture.image_view(),
                                            .imageLayout = vk::ImageLayout::eShaderReadOnlyOptimal}});

    descriptor_set_writes.push_back(
        vk::WriteDescriptorSet{.dstSet = descriptor_set,
                               .dstBinding = 1,
                               .dstArrayElement = 0,
                               .descriptorCount = static_cast<uint32_t>(descriptor_image_info.size()),
                               .descriptorType = vk::DescriptorType::eCombinedImageSampler,
                               .pImageInfo = descriptor_image_info.data(),
                               .pBufferInfo = &descriptor_buffer_info});

    material->descriptor_set = descriptor_set;
  }

  device.updateDescriptorSets(descriptor_set_writes, nullptr);
}

#pragma endregion

#pragma region meshes

using Meshes = std::unordered_map<const cgltf_mesh*, std::unique_ptr<const vktf::Mesh>>;

template <typename T, glm::length_t N>
  requires VecConstructible<T, N>
struct VertexAttribute {
  using Data = std::vector<glm::vec<N, T>>;
  std::string_view name;  // guaranteed to reference a string literal with static storage duration
  std::optional<Data> maybe_data;
};

template <glm::length_t N>
  requires VecConstructible<float, N>
std::vector<glm::vec<N, float>> UnpackFloats(const cgltf_accessor& gltf_accessor) {
  if (const auto components_count = cgltf_num_components(gltf_accessor.type); components_count != N) {
    throw std::runtime_error{std::format(
        "The number of expected components {} does not match the number of actual components {} for accessor {}",
        N,
        components_count,
        GetName(gltf_accessor))};
  }
  std::vector<glm::vec<N, float>> floats(gltf_accessor.count);
  if (const auto float_count = N * gltf_accessor.count;
      cgltf_accessor_unpack_floats(&gltf_accessor, glm::value_ptr(floats.front()), float_count) == 0) {
    throw std::runtime_error{std::format("Failed to unpack floats for accessor {}", GetName(gltf_accessor))};
  }
  return floats;
}

template <glm::length_t N>
bool TryUnpackAttribute(const cgltf_attribute& gltf_attribute, VertexAttribute<float, N>& vertex_attribute) {
  if (auto& [name, maybe_data] = vertex_attribute; name == GetName(gltf_attribute)) {
    assert(!maybe_data.has_value());  // glTF primitives should not have duplicate attributes
    maybe_data = UnpackFloats<N>(*gltf_attribute.data);
    return true;
  }
  return false;
}

template <glm::length_t N>
void ValidateRequiredAttribute(const VertexAttribute<float, N>& vertex_attribute) {
  if (const auto& maybe_attribute_data = vertex_attribute.maybe_data;
      !maybe_attribute_data.has_value() || maybe_attribute_data->empty()) {
    throw std::runtime_error{std::format("Missing required vertex attribute {}", vertex_attribute.name)};
  }
}

void ValidateAttributes(const VertexAttribute<float, 3>& position_attribute,
                        const VertexAttribute<float, 3>& normal_attribute,
                        const VertexAttribute<float, 4>& tangent_attribute,
                        const VertexAttribute<float, 2>& texture_coordinates_0_attribute) {
  ValidateRequiredAttribute(position_attribute);
  ValidateRequiredAttribute(normal_attribute);   // TODO: derive from positions when undefined
  ValidateRequiredAttribute(tangent_attribute);  // TODO: derive from positions and texture coordinates when undefined
  ValidateRequiredAttribute(texture_coordinates_0_attribute);

#ifndef NDEBUG
  // the glTF specification requires all primitive attributes have the same count
  for (const auto& attribute_count : {normal_attribute.maybe_data->size(),
                                      tangent_attribute.maybe_data->size(),
                                      texture_coordinates_0_attribute.maybe_data->size()}) {
    assert(attribute_count == position_attribute.maybe_data->size());
  }
#endif
}

std::vector<vktf::Vertex> CreateVertices(const std::span<const cgltf_attribute> gltf_attributes) {
  VertexAttribute<float, 3> position_attribute{.name = "POSITION"};
  VertexAttribute<float, 3> normal_attribute{.name = "NORMAL"};
  VertexAttribute<float, 4> tangent_attribute{.name = "TANGENT"};
  VertexAttribute<float, 2> texture_coordinates_0_attribute{.name = "TEXCOORD_0"};

  for (const auto& gltf_attribute : gltf_attributes) {
    switch (gltf_attribute.type) {
      case cgltf_attribute_type_position:
        if (TryUnpackAttribute(gltf_attribute, position_attribute)) continue;
        break;
      case cgltf_attribute_type_normal:
        if (TryUnpackAttribute(gltf_attribute, normal_attribute)) continue;
        break;
      case cgltf_attribute_type_tangent:
        if (TryUnpackAttribute(gltf_attribute, tangent_attribute)) continue;
        break;
      case cgltf_attribute_type_texcoord:
        if (TryUnpackAttribute(gltf_attribute, texture_coordinates_0_attribute)) continue;
        break;
      default:
        break;
    }
    std::println(std::cerr, "Unsupported primitive attribute {}", GetName(gltf_attribute));
  }

  ValidateAttributes(position_attribute, normal_attribute, tangent_attribute, texture_coordinates_0_attribute);

  return std::views::zip_transform(
             [](const auto& position, const auto& normal, const auto& tangent, const auto& texture_coordinates_0) {
               // the glTF specification requires unit length normals and tangent vectors
               static constexpr auto kEpsilon = 1.0e-6f;
               assert(glm::epsilonEqual(glm::length(normal), 1.0f, kEpsilon));
               assert(glm::epsilonEqual(glm::length(glm::vec3{tangent}), 1.0f, kEpsilon));
               return vktf::Vertex{.position = position,
                                   .normal = normal,
                                   .tangent = tangent,
                                   .texture_coordinates_0 = texture_coordinates_0};
             },
             *position_attribute.maybe_data,
             *normal_attribute.maybe_data,
             *tangent_attribute.maybe_data,
             *texture_coordinates_0_attribute.maybe_data)
         | std::ranges::to<std::vector>();
}

template <vktf::IndexType T>
std::vector<T> UnpackIndices(const cgltf_accessor& gltf_accessor) {
  std::vector<T> indices(gltf_accessor.count);
  if (const auto index_size_bytes = sizeof(T);
      cgltf_accessor_unpack_indices(&gltf_accessor, indices.data(), index_size_bytes, indices.size()) == 0) {
    throw std::runtime_error{std::format("Failed to unpack indices for accessor {}", GetName(gltf_accessor))};
  }
  return indices;
}

vktf::StagingPrimitive CreateStagingPrimitive(const cgltf_primitive& gltf_primitive, const VmaAllocator allocator) {
  const std::span primitive_attributes{gltf_primitive.attributes, gltf_primitive.attributes_count};
  const auto vertices = CreateVertices(primitive_attributes);

  switch (const auto* const indices_accessor = gltf_primitive.indices;
          cgltf_component_size(indices_accessor->component_type)) {
    case 1:
      return vktf::StagingPrimitive{vertices, UnpackIndices<std::uint8_t>(*indices_accessor), allocator};
    case 2:
      return vktf::StagingPrimitive{vertices, UnpackIndices<std::uint16_t>(*indices_accessor), allocator};
    case 4:
      return vktf::StagingPrimitive{vertices, UnpackIndices<std::uint32_t>(*indices_accessor), allocator};
    default:
      std::unreachable();  // the glTF specification only supports 8, 16, and 32-bit indices
  }
}

using StagingPrimitive = std::pair<vktf::StagingPrimitive, const vktf::Material*>;
using StagingMesh = std::vector<StagingPrimitive>;

StagingMesh CreateStagingMesh(const cgltf_mesh& gltf_mesh, const VmaAllocator allocator, const Materials& materials) {
  std::vector<StagingPrimitive> staging_primitives;
  staging_primitives.reserve(gltf_mesh.primitives_count);

  for (const auto& [index, gltf_primitive] :
       std::span{gltf_mesh.primitives, gltf_mesh.primitives_count} | std::views::enumerate) {
    if (gltf_primitive.type != cgltf_primitive_type_triangles) {
      static constexpr auto* kMessageFormat = "Failed to create mesh primitive {}[{}] with type {}";
      std::println(std::cerr, kMessageFormat, GetName(gltf_mesh), index, gltf_primitive.type);
      continue;  // TODO: support alternative primitive types
    }
    if (gltf_primitive.indices == nullptr || gltf_primitive.indices->count == 0) {
      static constexpr auto* kMessageFormat = "Failed to create mesh primitive {}[{}] with invalid indices accessor";
      std::println(std::cerr, kMessageFormat, GetName(gltf_mesh), index);
      continue;  // TODO: support non-indexed triangle meshes
    }
    const auto& material = Get(gltf_primitive.material, materials);
    if (material == nullptr) {
      static constexpr auto* kMessageFormat = "Failed to create mesh primitive {}[{}] with unsupported material";
      std::println(std::cerr, kMessageFormat, GetName(gltf_mesh), index);
      continue;  // TODO: support dynamic material layouts
    }
    staging_primitives.emplace_back(CreateStagingPrimitive(gltf_primitive, allocator), material.get());
  }

  return staging_primitives;
}

std::unique_ptr<const vktf::Mesh> CreateMesh(const StagingMesh& staging_mesh,
                                             const vk::CommandBuffer command_buffer,
                                             const VmaAllocator allocator) {
  std::vector<vktf::Primitive> primitives;
  primitives.reserve(staging_mesh.size());

  for (const auto& [staging_primitive, material] : staging_mesh) {
    primitives.emplace_back(staging_primitive, material, command_buffer, allocator);
  }

  return std::make_unique<const vktf::Mesh>(std::move(primitives));
}

#pragma endregion

#pragma region scene

using Lights = std::unordered_map<const cgltf_light*, std::unique_ptr<const vktf::Light>>;

const cgltf_scene& GetDefaultScene(const cgltf_data& gltf_data) {
  if (const auto* const gltf_scene = gltf_data.scene; gltf_scene != nullptr) {
    return *gltf_scene;
  }
  if (const std::span gltf_scenes{gltf_data.scenes, gltf_data.scenes_count}; !gltf_scenes.empty()) {
    return gltf_scenes.front();  // return the first available scene when no default is specified
  }
  // TODO: glTF files not containing scene data should be treated as a library of individual entities
  throw std::runtime_error{"At least one glTF scene is required to render"};
}

glm::mat4 GetLocalTransform(const cgltf_node& gltf_node) {
  glm::mat4 transform{0.0f};
  cgltf_node_transform_local(&gltf_node, glm::value_ptr(transform));
  return transform;
}

std::vector<std::unique_ptr<const vktf::Node>> CreateNodes(const cgltf_node* const* const gltf_nodes,
                                                           const cgltf_size gltf_nodes_count,
                                                           const Meshes& meshes,
                                                           const Lights& lights) {
  return std::span{gltf_nodes, gltf_nodes_count}
         | std::views::transform([&meshes, &lights](const auto* const gltf_node) {
             return std::make_unique<const vktf::Node>(
                 gltf_node->mesh == nullptr ? nullptr : Get(gltf_node->mesh, meshes).get(),
                 gltf_node->light == nullptr ? nullptr : Get(gltf_node->light, lights).get(),
                 GetLocalTransform(*gltf_node),
                 CreateNodes(gltf_node->children, gltf_node->children_count, meshes, lights));
           })
         | std::ranges::to<std::vector>();
}

#pragma endregion

#pragma region uniform buffers

struct CameraTransforms {
  glm::mat4 view_transform{0.0f};
  glm::mat4 projection_transform{0.0f};
};

std::vector<vktf::HostVisibleBuffer> CreateUniformBuffers(const std::size_t buffer_count,
                                                          const std::size_t buffer_size_bytes,
                                                          const VmaAllocator allocator) {
  return std::views::iota(0uz, buffer_count)
         | std::views::transform([buffer_size_bytes, allocator](const auto /*index*/) {
             vktf::HostVisibleBuffer uniform_buffer{buffer_size_bytes,
                                                    vk::BufferUsageFlagBits::eUniformBuffer,
                                                    allocator};
             uniform_buffer.MapMemory();  // enable persistent mapping
             return uniform_buffer;
           })
         | std::ranges::to<std::vector>();
}

std::optional<vktf::DescriptorPool> CreateGlobalDescriptorPool(const vk::Device device,
                                                               const std::uint32_t max_render_frames) {
  static constexpr std::uint32_t kBuffersPerRenderFrame = 2;
  const std::array descriptor_pool_sizes{
      vk::DescriptorPoolSize{.type = vk::DescriptorType::eUniformBuffer,
                             .descriptorCount = kBuffersPerRenderFrame * max_render_frames}};

  static constexpr std::array kDescriptorSetLayoutBindings{
      vk::DescriptorSetLayoutBinding{.binding = 0,  // camera buffer
                                     .descriptorType = vk::DescriptorType::eUniformBuffer,
                                     .descriptorCount = 1,
                                     .stageFlags = vk::ShaderStageFlagBits::eVertex},
      vk::DescriptorSetLayoutBinding{.binding = 1,  // lights buffer
                                     .descriptorType = vk::DescriptorType::eUniformBuffer,
                                     .descriptorCount = 1,
                                     .stageFlags = vk::ShaderStageFlagBits::eFragment}};

  return vktf::DescriptorPool{device, descriptor_pool_sizes, kDescriptorSetLayoutBindings, max_render_frames};
}

void UpdateGlobalDescriptorSets(const vk::Device device,
                                const vktf::DescriptorPool& global_descriptor_pool,
                                const std::vector<vktf::HostVisibleBuffer>& camera_uniform_buffers,
                                const std::vector<vktf::HostVisibleBuffer>& lights_uniform_buffers) {
  const auto& descriptor_sets = global_descriptor_pool.descriptor_sets();
  assert(camera_uniform_buffers.size() == descriptor_sets.size());
  assert(lights_uniform_buffers.size() == descriptor_sets.size());

  const auto buffer_count = camera_uniform_buffers.size() + lights_uniform_buffers.size();
  std::vector<vk::DescriptorBufferInfo> descriptor_buffer_infos;
  descriptor_buffer_infos.reserve(buffer_count);

  std::vector<vk::WriteDescriptorSet> descriptor_set_writes;
  descriptor_set_writes.reserve(buffer_count);

  for (const auto& [camera_uniform_buffer, lights_uniform_buffer, descriptor_set] :
       std::views::zip(camera_uniform_buffers, lights_uniform_buffers, descriptor_sets)) {
    const auto& camera_descriptor_buffer_info = descriptor_buffer_infos.emplace_back(
        vk::DescriptorBufferInfo{.buffer = *camera_uniform_buffer, .range = vk::WholeSize});

    descriptor_set_writes.push_back(vk::WriteDescriptorSet{.dstSet = descriptor_set,
                                                           .dstBinding = 0,
                                                           .dstArrayElement = 0,
                                                           .descriptorCount = 1,
                                                           .descriptorType = vk::DescriptorType::eUniformBuffer,
                                                           .pBufferInfo = &camera_descriptor_buffer_info});

    const auto& lights_descriptor_buffer_info = descriptor_buffer_infos.emplace_back(
        vk::DescriptorBufferInfo{.buffer = *lights_uniform_buffer, .range = vk::WholeSize});

    descriptor_set_writes.push_back(vk::WriteDescriptorSet{.dstSet = descriptor_set,
                                                           .dstBinding = 1,
                                                           .dstArrayElement = 0,
                                                           .descriptorCount = 1,
                                                           .descriptorType = vk::DescriptorType::eUniformBuffer,
                                                           .pBufferInfo = &lights_descriptor_buffer_info});
  }

  device.updateDescriptorSets(descriptor_set_writes, nullptr);
}

#pragma endregion

#pragma region graphics pipeline

struct PushConstants {
  glm::mat4 model_transform{0.0f};
  glm::vec3 view_position{0.0f};
};

template <std::size_t N>
vk::UniquePipelineLayout CreateGraphicsPipelineLayout(
    const vk::Device device,
    const std::array<vk::DescriptorSetLayout, N>& descriptor_set_layouts) {
  static constexpr std::array kPushConstantRanges{
      vk::PushConstantRange{.stageFlags = vk::ShaderStageFlagBits::eVertex,
                            .offset = offsetof(PushConstants, model_transform),
                            .size = sizeof(PushConstants::model_transform)},
      vk::PushConstantRange{.stageFlags = vk::ShaderStageFlagBits::eFragment,
                            .offset = offsetof(PushConstants, view_position),
                            .size = sizeof(PushConstants::view_position)}};

  return device.createPipelineLayoutUnique(
      vk::PipelineLayoutCreateInfo{.setLayoutCount = static_cast<std::uint32_t>(descriptor_set_layouts.size()),
                                   .pSetLayouts = descriptor_set_layouts.data(),
                                   .pushConstantRangeCount = static_cast<std::uint32_t>(kPushConstantRanges.size()),
                                   .pPushConstantRanges = kPushConstantRanges.data()});
}

vk::UniquePipeline CreateGraphicsPipeline(const vk::Device device,
                                          const vk::PipelineLayout graphics_pipeline_layout,
                                          const vk::Extent2D viewport_extent,
                                          const vk::SampleCountFlagBits msaa_sample_count,
                                          const vk::RenderPass render_pass,
                                          const std::uint32_t light_count) {
  const std::filesystem::path vertex_shader_filepath{"shaders/vertex.glsl.spv"};
  const vktf::ShaderModule vertex_shader_module{device, vertex_shader_filepath, vk::ShaderStageFlagBits::eVertex};

  const std::filesystem::path fragment_shader_filepath{"shaders/fragment.glsl.spv"};
  const vktf::ShaderModule fragment_shader_module{device, fragment_shader_filepath, vk::ShaderStageFlagBits::eFragment};

  static constexpr auto kLightCountSize = sizeof(decltype(light_count));
  static constexpr vk::SpecializationMapEntry kSpecializationMapEntry{.constantID = 0,
                                                                      .offset = 0,
                                                                      .size = kLightCountSize};

  const vk::SpecializationInfo specialization_info{.mapEntryCount = 1,
                                                   .pMapEntries = &kSpecializationMapEntry,
                                                   .dataSize = kLightCountSize,
                                                   .pData = &light_count};

  const std::array shader_stage_create_info{
      vk::PipelineShaderStageCreateInfo{.stage = vk::ShaderStageFlagBits::eVertex,
                                        .module = *vertex_shader_module,
                                        .pName = "main"},
      vk::PipelineShaderStageCreateInfo{.stage = vk::ShaderStageFlagBits::eFragment,
                                        .module = *fragment_shader_module,
                                        .pName = "main",
                                        .pSpecializationInfo = &specialization_info}};

  static constexpr vk::VertexInputBindingDescription kVertexInputBindingDescription{
      .binding = 0,
      .stride = sizeof(vktf::Vertex),
      .inputRate = vk::VertexInputRate::eVertex};

  static constexpr std::array kVertexAttributeDescriptions{
      vk::VertexInputAttributeDescription{.location = 0,
                                          .binding = 0,
                                          .format = vk::Format::eR32G32B32Sfloat,
                                          .offset = offsetof(vktf::Vertex, position)},
      vk::VertexInputAttributeDescription{.location = 1,
                                          .binding = 0,
                                          .format = vk::Format::eR32G32B32Sfloat,
                                          .offset = offsetof(vktf::Vertex, normal)},
      vk::VertexInputAttributeDescription{.location = 2,
                                          .binding = 0,
                                          .format = vk::Format::eR32G32B32A32Sfloat,
                                          .offset = offsetof(vktf::Vertex, tangent)},
      vk::VertexInputAttributeDescription{.location = 3,
                                          .binding = 0,
                                          .format = vk::Format::eR32G32Sfloat,
                                          .offset = offsetof(vktf::Vertex, texture_coordinates_0)}};

  static constexpr vk::PipelineVertexInputStateCreateInfo kVertexInputStateCreateInfo{
      .vertexBindingDescriptionCount = 1,
      .pVertexBindingDescriptions = &kVertexInputBindingDescription,
      .vertexAttributeDescriptionCount = static_cast<std::uint32_t>(kVertexAttributeDescriptions.size()),
      .pVertexAttributeDescriptions = kVertexAttributeDescriptions.data()};

  static constexpr vk::PipelineInputAssemblyStateCreateInfo kInputAssemblyStateCreateInfo{
      .topology = vk::PrimitiveTopology::eTriangleList};

  // TODO: use dynamic viewport and scissor pipeline state when window resizing is implemented
  const vk::Viewport viewport{.x = 0.0f,
                              .y = 0.0f,
                              .width = static_cast<float>(viewport_extent.width),
                              .height = static_cast<float>(viewport_extent.height),
                              .minDepth = 0.0f,
                              .maxDepth = 1.0f};
  const vk::Rect2D scissor{.offset = vk::Offset2D{.x = 0, .y = 0}, .extent = viewport_extent};

  const vk::PipelineViewportStateCreateInfo viewport_state_create_info{.viewportCount = 1,
                                                                       .pViewports = &viewport,
                                                                       .scissorCount = 1,
                                                                       .pScissors = &scissor};

  static constexpr vk::PipelineRasterizationStateCreateInfo kRasterizationStateCreateInfo{
      .polygonMode = vk::PolygonMode::eFill,
      .cullMode = vk::CullModeFlagBits::eBack,
      .frontFace = vk::FrontFace::eCounterClockwise,
      .lineWidth = 1.0f};

  static constexpr vk::PipelineDepthStencilStateCreateInfo kDepthStencilStateCreateInfo{
      .depthTestEnable = vk::True,
      .depthWriteEnable = vk::True,
      .depthCompareOp = vk::CompareOp::eLess};

  const vk::PipelineMultisampleStateCreateInfo multisample_state_create_info{.rasterizationSamples = msaa_sample_count};

  using enum vk::ColorComponentFlagBits;
  static constexpr vk::PipelineColorBlendAttachmentState kColorBlendAttachmentState{
      .blendEnable = vk::True,
      .srcColorBlendFactor = vk::BlendFactor::eSrcAlpha,
      .dstColorBlendFactor = vk::BlendFactor::eOneMinusSrcAlpha,
      .colorBlendOp = vk::BlendOp::eAdd,
      .srcAlphaBlendFactor = vk::BlendFactor::eOne,
      .dstAlphaBlendFactor = vk::BlendFactor::eZero,
      .alphaBlendOp = vk::BlendOp::eAdd,
      .colorWriteMask = eR | eG | eB | eA};

  static constexpr vk::PipelineColorBlendStateCreateInfo kColorBlendStateCreateInfo{
      .attachmentCount = 1,
      .pAttachments = &kColorBlendAttachmentState,
      .blendConstants = std::array{0.0f, 0.0f, 0.0f, 0.0f}};

  auto [result, graphics_pipeline] = device.createGraphicsPipelineUnique(
      nullptr,
      vk::GraphicsPipelineCreateInfo{.stageCount = static_cast<std::uint32_t>(shader_stage_create_info.size()),
                                     .pStages = shader_stage_create_info.data(),
                                     .pVertexInputState = &kVertexInputStateCreateInfo,
                                     .pInputAssemblyState = &kInputAssemblyStateCreateInfo,
                                     .pViewportState = &viewport_state_create_info,
                                     .pRasterizationState = &kRasterizationStateCreateInfo,
                                     .pMultisampleState = &multisample_state_create_info,
                                     .pDepthStencilState = &kDepthStencilStateCreateInfo,
                                     .pColorBlendState = &kColorBlendStateCreateInfo,
                                     .layout = graphics_pipeline_layout,
                                     .renderPass = render_pass,
                                     .subpass = 0});
  vk::detail::resultCheck(result, "Graphics pipeline creation failed");

  return std::move(graphics_pipeline);  // return value optimization not available here
}

#pragma endregion

#pragma region rendering

struct WorldPrimitive {
  const vktf::Primitive* primitive = nullptr;
  glm::mat4 world_transform{0.0f};
};

using WorldPrimitivesByMaterial = std::unordered_map<const vktf::Material*, std::vector<WorldPrimitive>>;

void UpdateWorldTransforms(const vktf::Node& node,
                           const glm::mat4& parent_transform,
                           std::vector<vktf::Light>& world_lights,
                           WorldPrimitivesByMaterial& world_primitives_by_material) {
  const auto world_transform = parent_transform * node.transform;

  if (const auto* const light = node.light; light != nullptr) {
    if (light->position.w == 0.0f) {
      const auto& light_direction = world_transform[2];  // light direction inherited from node world-orientation z-axis
      world_lights.emplace_back(glm::normalize(light_direction), light->color);
    } else {
      assert(light->position.w == 1.0f);
      const auto& light_position = world_transform[3];  // light position inherited from node world-position
      world_lights.emplace_back(light_position, light->color);
    }
  }

  if (const auto* const mesh = node.mesh; mesh != nullptr) {
    for (const auto& primitive : *mesh) {
      world_primitives_by_material[primitive.material()].emplace_back(&primitive, world_transform);
    }
  }

  for (const auto& child_node : node.children) {
    UpdateWorldTransforms(*child_node, world_transform, world_lights, world_primitives_by_material);
  }
}

void Render(const WorldPrimitivesByMaterial& world_primitives_by_material,
            const vk::PipelineLayout graphics_pipeline_layout,
            const vk::CommandBuffer command_buffer) {
  for (const auto& [material, world_primitives] : world_primitives_by_material) {
    assert(material != nullptr);
    command_buffer.bindDescriptorSets(vk::PipelineBindPoint::eGraphics,
                                      graphics_pipeline_layout,
                                      1,
                                      material->descriptor_set,
                                      nullptr);

    for (const auto& [primitive, world_transform] : world_primitives) {
      using ModelTransform = decltype(PushConstants::model_transform);
      command_buffer.pushConstants<ModelTransform>(graphics_pipeline_layout,
                                                   vk::ShaderStageFlagBits::eVertex,
                                                   offsetof(PushConstants, model_transform),
                                                   world_transform);
      assert(primitive != nullptr);
      primitive->Render(command_buffer);
    }
  }
}

#pragma endregion

}  // namespace

namespace vktf {

GltfScene::GltfScene(const std::filesystem::path& gltf_filepath,
                     const vk::PhysicalDevice physical_device,
                     const vk::Bool32 enable_sampler_anisotropy,
                     const float max_sampler_anisotropy,
                     const vk::Device device,
                     const vk::Queue transfer_queue,
                     const std::uint32_t transfer_queue_family_index,
                     const vk::Extent2D viewport_extent,
                     const vk::SampleCountFlagBits msaa_sample_count,
                     const vk::RenderPass render_pass,
                     const VmaAllocator allocator,
                     const std::size_t max_render_frames) {
  const auto gltf_directory = gltf_filepath.parent_path();
  const auto gltf_data = Load(gltf_filepath.string());

  const CommandPool copy_command_pool{device,
                                      vk::CommandPoolCreateFlagBits::eTransient,
                                      transfer_queue_family_index,
                                      1};
  const auto command_buffer = *copy_command_pool.command_buffers().front();
  command_buffer.begin(vk::CommandBufferBeginInfo{.flags = vk::CommandBufferUsageFlagBits::eOneTimeSubmit});

  const auto staging_buffer_count = gltf_data->buffers_count + gltf_data->images_count;
  std::vector<HostVisibleBuffer> staging_buffers;
  staging_buffers.reserve(staging_buffer_count);

  auto samplers =
      std::span{gltf_data->samplers, gltf_data->samplers_count}
      | std::views::transform([device, enable_sampler_anisotropy, max_sampler_anisotropy](const auto& gltf_sampler) {
          return std::pair{&gltf_sampler,
                           CreateSampler(gltf_sampler, device, enable_sampler_anisotropy, max_sampler_anisotropy)};
        })
      | std::ranges::to<std::unordered_map>();

  if (samplers.empty()) {
    samplers.emplace(nullptr, CreateDefaultSampler(device, enable_sampler_anisotropy, max_sampler_anisotropy));
  }

  auto materials =
      std::span{gltf_data->materials, gltf_data->materials_count}
      | std::views::transform([&gltf_directory, physical_device, &samplers](const auto& gltf_material) {
          return CreateMaterialFutures(gltf_material, gltf_directory, physical_device, samplers);
        })
      | std::ranges::to<std::vector>()  // force evaluation to allow all threads to begin execution before proceeding
      | std::views::transform([device, command_buffer, allocator, &staging_buffers](MaterialFutures& material_futures) {
          return std::pair{material_futures.gltf_material,
                           CreateMaterial(material_futures, device, command_buffer, allocator, staging_buffers)};
        })
      | std::ranges::to<std::unordered_map>();

  const auto staging_meshes = std::span{gltf_data->meshes, gltf_data->meshes_count}
                              | std::views::transform([allocator, &materials](const auto& gltf_mesh) {
                                  return std::pair{&gltf_mesh, CreateStagingMesh(gltf_mesh, allocator, materials)};
                                })
                              | std::ranges::to<std::vector>();

  auto meshes = staging_meshes  //
                | std::views::transform([command_buffer, allocator](const auto& mesh_pair) {
                    const auto& [gltf_mesh, staging_mesh] = mesh_pair;
                    return std::pair{gltf_mesh, CreateMesh(staging_mesh, command_buffer, allocator)};
                  })
                | std::ranges::to<std::unordered_map>();

  command_buffer.end();
  const auto copy_fence = device.createFenceUnique(vk::FenceCreateInfo{});
  transfer_queue.submit(vk::SubmitInfo{.commandBufferCount = 1, .pCommandBuffers = &command_buffer}, *copy_fence);

  auto lights =
      std::span{gltf_data->lights, gltf_data->lights_count}  //
      | std::views::filter([](const auto& gltf_light) {
          switch (gltf_light.type) {
            case cgltf_light_type_directional:
            case cgltf_light_type_point:
              return true;
            default:
              std::println(std::cerr, "Unsupported light {} with type {}", GetName(gltf_light), gltf_light.type);
              return false;  // TODO: add cgltf_light_type_spot support
          }
        })
      | std::views::transform([](const auto& gltf_light) {
          // light position (or direction) is determined during rendering based on its associated node transform
          const glm::vec4 position{glm::vec3{0.0f}, static_cast<float>(gltf_light.type == cgltf_light_type_point)};
          const glm::vec4 color{ToVec(gltf_light.color), 1.0f};
          return std::pair{&gltf_light, std::make_unique<const Light>(position, color)};
        })
      | std::ranges::to<std::unordered_map>();

  const auto& gltf_scene = GetDefaultScene(*gltf_data);
  root_node_ = std::make_unique<const Node>(nullptr,
                                            nullptr,
                                            glm::mat4{1.0f},
                                            CreateNodes(gltf_scene.nodes, gltf_scene.nodes_count, meshes, lights));

  camera_uniform_buffers_ = CreateUniformBuffers(max_render_frames, sizeof(CameraTransforms), allocator);
  lights_uniform_buffers_ = CreateUniformBuffers(max_render_frames, sizeof(Light) * lights.size(), allocator);
  global_descriptor_pool_ = CreateGlobalDescriptorPool(device, static_cast<std::uint32_t>(max_render_frames));
  UpdateGlobalDescriptorSets(device, *global_descriptor_pool_, camera_uniform_buffers_, lights_uniform_buffers_);

  material_descriptor_pool_ = CreateMaterialDescriptorPool(device, static_cast<std::uint32_t>(materials.size()));
  UpdateMaterialDescriptorSets(device, *material_descriptor_pool_, materials);

  graphics_pipeline_layout_ = CreateGraphicsPipelineLayout(
      device,
      std::array{global_descriptor_pool_->descriptor_set_layout(), material_descriptor_pool_->descriptor_set_layout()});
  graphics_pipeline_ = CreateGraphicsPipeline(device,
                                              *graphics_pipeline_layout_,
                                              viewport_extent,
                                              msaa_sample_count,
                                              render_pass,
                                              static_cast<std::uint32_t>(lights.size()));

  materials_ = materials | std::views::values | std::views::as_rvalue | std::ranges::to<std::vector>();
  samplers_ = samplers | std::views::values | std::views::as_rvalue | std::ranges::to<std::vector>();
  meshes_ = meshes | std::views::values | std::views::as_rvalue | std::ranges::to<std::vector>();
  lights_ = lights | std::views::values | std::views::as_rvalue | std::ranges::to<std::vector>();

  static constexpr auto kMaxTimeout = std::numeric_limits<std::uint64_t>::max();
  const auto result = device.waitForFences(*copy_fence, vk::True, kMaxTimeout);
  vk::detail::resultCheck(result, "Copy fence failed to enter a signaled state");
}

void GltfScene::Render(const Camera& camera,
                       const std::size_t frame_index,
                       const vk::CommandBuffer command_buffer) const {
  std::vector<Light> world_lights;
  world_lights.reserve(lights_.size());

  WorldPrimitivesByMaterial world_primitives_by_material;
  world_primitives_by_material.reserve(materials_.size());

  UpdateWorldTransforms(*root_node_, glm::mat4{1.0f}, world_lights, world_primitives_by_material);

  command_buffer.bindPipeline(vk::PipelineBindPoint::eGraphics, *graphics_pipeline_);

  const auto& global_descriptor_sets = global_descriptor_pool_->descriptor_sets();
  command_buffer.bindDescriptorSets(vk::PipelineBindPoint::eGraphics,
                                    *graphics_pipeline_layout_,
                                    0,
                                    global_descriptor_sets[frame_index],
                                    nullptr);

  camera_uniform_buffers_[frame_index].Copy<CameraTransforms>(
      CameraTransforms{.view_transform = camera.view_transform(),
                       .projection_transform = camera.projection_transform()});

  lights_uniform_buffers_[frame_index].Copy<Light>(world_lights);

  using ViewPosition = decltype(PushConstants::view_position);
  command_buffer.pushConstants<ViewPosition>(*graphics_pipeline_layout_,
                                             vk::ShaderStageFlagBits::eFragment,
                                             offsetof(PushConstants, view_position),
                                             camera.GetPosition());

  ::Render(world_primitives_by_material, *graphics_pipeline_layout_, command_buffer);
}

}  // namespace vktf
