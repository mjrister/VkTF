#include "graphics/model.h"

#include <algorithm>
#include <array>
#include <cassert>
#include <concepts>
#include <cstring>
#include <format>
#include <future>
#include <iostream>
#include <limits>
#include <optional>
#include <print>
#include <ranges>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_set>
#include <utility>

#include <cgltf.h>
#include <ktx.h>
#include <stb_image.h>
#include <glm/glm.hpp>
#include <glm/gtc/epsilon.hpp>
#include <glm/gtc/type_ptr.hpp>

#include "graphics/buffer.h"
#include "graphics/camera.h"
#include "graphics/image.h"
#include "graphics/mesh.h"
#include "graphics/shader_module.h"

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
      CASE(cgltf_primitive_type_points)
      CASE(cgltf_primitive_type_lines)
      CASE(cgltf_primitive_type_line_loop)
      CASE(cgltf_primitive_type_line_strip)
      CASE(cgltf_primitive_type_triangles)
      CASE(cgltf_primitive_type_triangle_strip)
      CASE(cgltf_primitive_type_triangle_fan)
#undef CASE
      // clang-format on
      default:
        std::unreachable();
    }
  }
};

namespace {

struct PushConstants {
  glm::mat4 model_view_transform{1.0f};
  glm::mat4 projection_transform{1.0f};
};

struct Vertex {
  glm::vec3 position{0.0f};
  glm::vec3 normal{0.0f};
  glm::vec4 tangent{0.0f};  // w-component indicates the signed handedness of the tangent basis
  glm::vec2 texture_coordinates0{0.0f};
};

template <typename T, glm::length_t N>
struct VertexAttribute {
  std::string_view name;
  std::optional<std::vector<glm::vec<N, T>>> maybe_data;
};

struct TranscodeFormat {
  vk::Format srgb_format = vk::Format::eUndefined;
  vk::Format unorm_format = vk::Format::eUndefined;
  ktx_transcode_fmt_e ktx_transcode_format = KTX_TTF_NOSELECTION;
};

using UniqueGltfData = std::unique_ptr<cgltf_data, decltype(&cgltf_free)>;

// NOLINTNEXTLINE(cppcoreguidelines-pro-type-cstyle-cast)
void DestroyKtxTexture2(ktxTexture2* const ktx_texture2) noexcept { ktxTexture_Destroy(ktxTexture(ktx_texture2)); }
using UniqueKtxTexture2 = std::unique_ptr<ktxTexture2, decltype(&DestroyKtxTexture2)>;

constexpr TranscodeFormat kBc1TranscodeFormat{.srgb_format = vk::Format::eBc1RgbSrgbBlock,
                                              .unorm_format = vk::Format::eBc1RgbUnormBlock,
                                              .ktx_transcode_format = KTX_TTF_BC1_RGB};
constexpr TranscodeFormat kBc3TranscodeFormat{.srgb_format = vk::Format::eBc3SrgbBlock,
                                              .unorm_format = vk::Format::eBc3UnormBlock,
                                              .ktx_transcode_format = KTX_TTF_BC3_RGBA};
constexpr TranscodeFormat kBc7TranscodeFormat{.srgb_format = vk::Format::eBc7SrgbBlock,
                                              .unorm_format = vk::Format::eBc7UnormBlock,
                                              .ktx_transcode_format = KTX_TTF_BC7_RGBA};
constexpr TranscodeFormat kEtc1TranscodeFormat{.srgb_format = vk::Format::eEtc2R8G8B8SrgbBlock,
                                               .unorm_format = vk::Format::eEtc2R8G8B8UnormBlock,
                                               .ktx_transcode_format = KTX_TTF_ETC1_RGB};
constexpr TranscodeFormat kEtc2TranscodeFormat{.srgb_format = vk::Format::eEtc2R8G8B8A8SrgbBlock,
                                               .unorm_format = vk::Format::eEtc2R8G8B8A8UnormBlock,
                                               .ktx_transcode_format = KTX_TTF_ETC2_RGBA};
constexpr TranscodeFormat kAstc4x4TranscodeFormat{.srgb_format = vk::Format::eAstc4x4SrgbBlock,
                                                  .unorm_format = vk::Format::eAstc4x4UnormBlock,
                                                  .ktx_transcode_format = KTX_TTF_ASTC_4x4_RGBA};
constexpr TranscodeFormat kRgba32TranscodeFormat{.srgb_format = vk::Format::eR8G8B8A8Srgb,
                                                 .unorm_format = vk::Format::eR8G8B8A8Unorm,
                                                 .ktx_transcode_format = KTX_TTF_RGBA32};

UniqueGltfData ReadFile(const std::filesystem::path& gltf_filepath) {
  static constexpr cgltf_options kGltfOptions{};
  UniqueGltfData gltf_data{nullptr, nullptr};
  const auto gltf_filepath_string = gltf_filepath.string();

  if (const auto gltf_result =
          cgltf_parse_file(&kGltfOptions, gltf_filepath_string.c_str(), std::out_ptr(gltf_data, cgltf_free));
      gltf_result != cgltf_result_success) {
    throw std::runtime_error{std::format("Failed to parse {} with error {}", gltf_filepath_string, gltf_result)};
  }
  if (const auto gltf_result = cgltf_load_buffers(&kGltfOptions, gltf_data.get(), gltf_filepath_string.c_str());
      gltf_result != cgltf_result_success) {
    throw std::runtime_error{
        std::format("Failed to load buffers for {} with error {}", gltf_filepath_string, gltf_result)};
  }
#ifndef NDEBUG
  if (const auto gltf_result = cgltf_validate(gltf_data.get()); gltf_result != cgltf_result_success) {
    throw std::runtime_error{std::format("Failed to validate {} with error {}", gltf_filepath_string, gltf_result)};
  }
#endif

  return gltf_data;
}

const cgltf_scene& GetDefaultScene(const cgltf_data& gltf_data, const std::filesystem::path& gltf_filepath) {
  if (const auto* const gltf_scene = gltf_data.scene; gltf_scene != nullptr) {
    return *gltf_scene;
  }
  if (const std::span gltf_scenes{gltf_data.scenes, gltf_data.scenes_count}; !gltf_scenes.empty()) {
    return gltf_scenes.front();
  }
  throw std::runtime_error{std::format("No scene data found for {}", gltf_filepath.string())};
}

template <typename T>
  requires requires(T gltf_element) {
    { gltf_element.name } -> std::same_as<char*&>;
  }
std::string_view GetName(const T& gltf_element) {
  const auto name_size = gltf_element.name == nullptr ? 0 : std::strlen(gltf_element.name);
  return name_size == 0 ? "unknown" : std::string_view{gltf_element.name, name_size};
}

template <glm::length_t N>
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

template <typename T>
  requires std::same_as<T, std::uint16_t> || std::same_as<T, std::uint32_t>
std::vector<T> UnpackIndices(const cgltf_accessor& gltf_accessor) {
  std::vector<T> indices(gltf_accessor.count);
  if (const auto index_size_bytes = sizeof(T);
      cgltf_accessor_unpack_indices(&gltf_accessor, indices.data(), index_size_bytes, indices.size()) == 0) {
    throw std::runtime_error{std::format("Failed to unpack indices for accessor {}", GetName(gltf_accessor))};
  }
  return indices;
}

std::vector<Vertex> GetVertices(const cgltf_primitive& gltf_primitive) {
  VertexAttribute<float, 3> positions_attribute{.name = "POSITION"};
  VertexAttribute<float, 3> normals_attribute{.name = "NORMAL"};
  VertexAttribute<float, 4> tangents_attribute{.name = "TANGENT"};
  VertexAttribute<float, 2> texture_coordinates0_attribute{.name = "TEXCOORD_0"};

  static constexpr auto kUnpackAttributeData = []<glm::length_t N>(const auto* const gltf_accessor,
                                                                   VertexAttribute<float, N>& vertex_attribute) {
    assert(gltf_accessor != nullptr);                  // valid glTF primitives should not have null attribute data
    assert(!vertex_attribute.maybe_data.has_value());  // valid glTF primitives should not have duplicate attributes
    vertex_attribute.maybe_data = UnpackFloats<N>(*gltf_accessor);
  };

  for (const auto& gltf_attribute : std::span{gltf_primitive.attributes, gltf_primitive.attributes_count}) {
    switch (const auto* const gltf_accessor = gltf_attribute.data; gltf_attribute.type) {
      case cgltf_attribute_type_position:
        kUnpackAttributeData(gltf_accessor, positions_attribute);
        break;
      case cgltf_attribute_type_normal:
        kUnpackAttributeData(gltf_accessor, normals_attribute);
        break;
      case cgltf_attribute_type_tangent:
        kUnpackAttributeData(gltf_accessor, tangents_attribute);
        break;
      case cgltf_attribute_type_texcoord:
        if (texture_coordinates0_attribute.name == GetName(gltf_attribute)) {
          kUnpackAttributeData(gltf_accessor, texture_coordinates0_attribute);
          break;
        }
        [[fallthrough]];
      default:
        std::println(std::cerr, "Unsupported primitive attribute {}", GetName(gltf_attribute));
        break;
    }
  }

  const auto& [positions_attribute_name, maybe_positions] = positions_attribute;
  const auto& [normals_attribute_name, maybe_normals] = normals_attribute;
  const auto& [tangents_attribute_name, maybe_tangents] = tangents_attribute;
  const auto& [texture_coordinates0_attribute_name, maybe_texture_coordinates0] = texture_coordinates0_attribute;

  for (const auto [attribute_name, has_attribute_data] :
       std::array{std::pair{positions_attribute_name, maybe_positions.has_value()},
                  std::pair{normals_attribute_name, maybe_normals.has_value()},
                  std::pair{tangents_attribute_name, maybe_tangents.has_value()},
                  std::pair{texture_coordinates0_attribute_name, maybe_texture_coordinates0.has_value()}}) {
    if (!has_attribute_data) {
      throw std::runtime_error{std::format("Missing required vertex attribute {}", attribute_name)};
    }
  }

  // primitives are expected to represent an indexed triangle mesh with a matching number of vertex attributes
  for (const auto positions_count = maybe_positions->size();
       const auto [attribute_name, attribute_count] :
       std::array{std::pair{normals_attribute_name, maybe_normals->size()},
                  std::pair{tangents_attribute_name, maybe_tangents->size()},
                  std::pair{texture_coordinates0_attribute_name, maybe_texture_coordinates0->size()}}) {
    if (attribute_count != positions_count) {
      throw std::runtime_error{
          std::format("The number of {} attributes {} does not match the number of {} attributes {}",
                      positions_attribute_name,
                      positions_count,
                      attribute_name,
                      attribute_count)};
    }
  }

  return std::views::zip_transform(
             [](const auto& position, const auto& normal, const auto& tangent, const auto& texture_coordinates0) {
#ifndef NDEBUG
               // valid glTF primitives should have unit length normal and tangent vectors
               static constexpr auto kEpsilon = 1.0e-6f;
               assert(glm::epsilonEqual(glm::length(normal), 1.0f, kEpsilon));
               assert(glm::epsilonEqual(glm::length(glm::vec3{tangent}), 1.0f, kEpsilon));
#endif
               return Vertex{.position = position,
                             .normal = normal,
                             .tangent = tangent,
                             .texture_coordinates0 = texture_coordinates0};
             },
             *maybe_positions,
             *maybe_normals,
             *maybe_tangents,
             *maybe_texture_coordinates0)
         | std::ranges::to<std::vector>();
}

template <typename T>
gfx::Buffer CreateBuffer(const std::vector<T>& buffer_data,
                         const vk::BufferUsageFlags buffer_usage_flags,
                         const vk::CommandBuffer command_buffer,
                         const VmaAllocator allocator,
                         std::vector<gfx::Buffer>& staging_buffers) {
  const auto buffer_size_bytes = sizeof(T) * buffer_data.size();

  static constexpr VmaAllocationCreateInfo kStagingBufferAllocationCreateInfo{
      .flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT,
      .usage = VMA_MEMORY_USAGE_AUTO};
  auto& staging_buffer = staging_buffers.emplace_back(buffer_size_bytes,
                                                      vk::BufferUsageFlagBits::eTransferSrc,
                                                      allocator,
                                                      kStagingBufferAllocationCreateInfo);
  staging_buffer.template CopyOnce<T>(buffer_data);

  static constexpr VmaAllocationCreateInfo kBufferAllocationCreateInfo{.usage = VMA_MEMORY_USAGE_AUTO};
  gfx::Buffer buffer{buffer_size_bytes,
                     buffer_usage_flags | vk::BufferUsageFlagBits::eTransferDst,
                     allocator,
                     kBufferAllocationCreateInfo};
  command_buffer.copyBuffer(*staging_buffer, *buffer, vk::BufferCopy{.size = buffer_size_bytes});

  return buffer;
}

gfx::IndexBuffer CreateIndexBuffer(const cgltf_accessor& gltf_accessor,
                                   const vk::CommandBuffer command_buffer,
                                   const VmaAllocator allocator,
                                   std::vector<gfx::Buffer>& staging_buffers) {
  switch (const auto component_size = cgltf_component_size(gltf_accessor.component_type)) {
    using enum vk::BufferUsageFlagBits;
    case 2: {
      const auto indices = UnpackIndices<std::uint16_t>(gltf_accessor);
      return gfx::IndexBuffer{
          .index_count = static_cast<std::uint32_t>(indices.size()),
          .index_type = vk::IndexType::eUint16,
          .buffer = CreateBuffer(indices, eIndexBuffer, command_buffer, allocator, staging_buffers)};
    }
    case 4: {
      const auto indices = UnpackIndices<std::uint32_t>(gltf_accessor);
      return gfx::IndexBuffer{
          .index_count = static_cast<std::uint32_t>(indices.size()),
          .index_type = vk::IndexType::eUint32,
          .buffer = CreateBuffer(indices, eIndexBuffer, command_buffer, allocator, staging_buffers)};
    }
    default: {
      static constexpr auto kBitsPerByte = 8;
      throw std::runtime_error{std::format("Unsupported {}-bit index type", component_size * kBitsPerByte)};
    }
  }
}

std::vector<gfx::Mesh> CreateMeshes(const cgltf_mesh& gltf_mesh,
                                    const vk::CommandBuffer command_buffer,
                                    const VmaAllocator allocator,
                                    const std::unordered_map<const cgltf_material*, vk::DescriptorSet>& descriptor_sets,
                                    std::vector<gfx::Buffer>& staging_buffers) {
  std::vector<gfx::Mesh> meshes;
  meshes.reserve(gltf_mesh.primitives_count);

  for (const auto& gltf_primitive : std::span{gltf_mesh.primitives, gltf_mesh.primitives_count}) {
    if (gltf_primitive.type != cgltf_primitive_type_triangles) {
      std::println(std::cerr, "Unsupported primitive {}", GetName(gltf_mesh), gltf_primitive.type);
      continue;  // TODO(matthew-rister): add support for other primitive types
    }
    if (gltf_primitive.indices == nullptr || gltf_primitive.indices->count == 0) {
      // TODO(matthew-rister): add support for non-indexed triangle meshes
      throw std::runtime_error{"Primitive must represent a valid indexed triangle mesh"};
    }
    const auto iterator = descriptor_sets.find(gltf_primitive.material);
    if (iterator == descriptor_sets.cend()) continue;  // exclude primitive with unsupported material

    const auto vertices = GetVertices(gltf_primitive);
    meshes.emplace_back(
        CreateBuffer(vertices, vk::BufferUsageFlagBits::eVertexBuffer, command_buffer, allocator, staging_buffers),
        CreateIndexBuffer(*gltf_primitive.indices, command_buffer, allocator, staging_buffers),
        iterator->second);
  }

  return meshes;
}

std::vector<gfx::Mesh> GetMeshes(const cgltf_node& gltf_node,
                                 std::unordered_map<const cgltf_mesh*, std::vector<gfx::Mesh>>& meshes) {
  if (gltf_node.mesh == nullptr) return {};
  const auto iterator = meshes.find(gltf_node.mesh);
  assert(iterator != meshes.cend());  // all glTF meshes should have been processed by here
  return std::move(iterator->second);
}

glm::mat4 GetTransform(const cgltf_node& gltf_node) {
  glm::mat4 transform{1.0f};
  if (gltf_node.has_matrix != 0 || gltf_node.has_translation != 0 || gltf_node.has_rotation != 0
      || gltf_node.has_scale != 0) {
    cgltf_node_transform_local(&gltf_node, glm::value_ptr(transform));
  }
  return transform;
}

std::unordered_set<vk::Format> GetSupportedTranscodeFormats(const vk::PhysicalDevice& physical_device) {
  static constexpr std::array kTargetTranscodeFormats{
      // clang-format off
    kBc1TranscodeFormat.srgb_format, kBc1TranscodeFormat.unorm_format,
    kBc3TranscodeFormat.srgb_format, kBc3TranscodeFormat.unorm_format,
    kBc7TranscodeFormat.srgb_format, kBc7TranscodeFormat.unorm_format,
    kEtc1TranscodeFormat.srgb_format, kEtc1TranscodeFormat.unorm_format,
    kEtc2TranscodeFormat.srgb_format, kEtc2TranscodeFormat.unorm_format,
    kAstc4x4TranscodeFormat.srgb_format, kAstc4x4TranscodeFormat.unorm_format,
    kRgba32TranscodeFormat.srgb_format, kRgba32TranscodeFormat.unorm_format
      // clang-format on
  };
  return kTargetTranscodeFormats  //
         | std::views::filter([physical_device](const auto transcode_format) {
             const auto format_properties = physical_device.getFormatProperties(transcode_format);
             return static_cast<bool>(format_properties.optimalTilingFeatures
                                      & vk::FormatFeatureFlagBits::eSampledImage);
           })
         | std::ranges::to<std::unordered_set>();
}

ktx_transcode_fmt_e GetKtxTranscodeFormatForColorModel(
    const std::span<const TranscodeFormat> color_model_transcode_formats,
    const bool has_unorm_format,
    const std::unordered_set<vk::Format>& supported_transcode_formats) {
  for (const auto [srgb_format, unorm_format, ktx_transcode_format] : color_model_transcode_formats) {
    if (const auto color_model_transcode_format = has_unorm_format ? unorm_format : srgb_format;
        supported_transcode_formats.contains(color_model_transcode_format)) {
      return ktx_transcode_format;
    }
  }
  const auto [srgb_format, unorm_format, ktx_transcode_format] = kRgba32TranscodeFormat;
  if (const auto rgba32_transcode_format = has_unorm_format ? unorm_format : srgb_format;
      supported_transcode_formats.contains(rgba32_transcode_format)) {
#ifndef NDEBUG
    std::println(std::clog,
                 "No supported texture compression format could be found. Decompressing to {}",
                 ktxTranscodeFormatString(ktx_transcode_format));
#endif
    return ktx_transcode_format;
  }
  throw std::runtime_error{"No supported KTX transcode formats could be found"};
}

ktx_transcode_fmt_e GetKtxTranscodeFormat(ktxTexture2& ktx_texture2,
                                          const bool has_unorm_format,
                                          const std::unordered_set<vk::Format>& supported_transcode_formats) {
  // format selection based on https://github.com/KhronosGroup/3D-Formats-Guidelines/blob/main/KTXDeveloperGuide.md
  switch (const auto has_alpha = ktxTexture2_GetNumComponents(&ktx_texture2) == 4;
          ktxTexture2_GetColorModel_e(&ktx_texture2)) {
    case KHR_DF_MODEL_ETC1S: {
      static constexpr std::array kEtc1sRgbTranscodeFormats{kEtc1TranscodeFormat,
                                                            kBc7TranscodeFormat,
                                                            kBc1TranscodeFormat};
      static constexpr std::array kEtc1sRgbaTranscodeFormats{kEtc2TranscodeFormat,
                                                             kBc7TranscodeFormat,
                                                             kBc3TranscodeFormat};
      const auto& etc1s_transcode_formats = has_alpha ? kEtc1sRgbaTranscodeFormats : kEtc1sRgbTranscodeFormats;
      return GetKtxTranscodeFormatForColorModel(etc1s_transcode_formats, has_unorm_format, supported_transcode_formats);
    }
    case KHR_DF_MODEL_UASTC: {
      static constexpr std::array kUastcRgbTranscodeFormats{kAstc4x4TranscodeFormat,
                                                            kBc7TranscodeFormat,
                                                            kEtc1TranscodeFormat,
                                                            kBc1TranscodeFormat};
      static constexpr std::array kUastcRgbaTranscodeFormats{kAstc4x4TranscodeFormat,
                                                             kBc7TranscodeFormat,
                                                             kEtc2TranscodeFormat,
                                                             kBc3TranscodeFormat};
      const auto& uastc_transcode_formats = has_alpha ? kUastcRgbaTranscodeFormats : kUastcRgbTranscodeFormats;
      return GetKtxTranscodeFormatForColorModel(uastc_transcode_formats, has_unorm_format, supported_transcode_formats);
    }
    default:
      std::unreachable();  // basis universal only supports UASTC/ETC1S transmission formats
  }
}

UniqueKtxTexture2 CreateKtxTexture2FromKtxFile(const std::filesystem::path& ktx_filepath,
                                               const bool has_unorm_format,
                                               const std::unordered_set<vk::Format>& supported_transcode_formats) {
  UniqueKtxTexture2 ktx_texture2{nullptr, nullptr};
  if (const auto ktx_error_code = ktxTexture2_CreateFromNamedFile(ktx_filepath.string().c_str(),
                                                                  KTX_TEXTURE_CREATE_CHECK_GLTF_BASISU_BIT,
                                                                  std::out_ptr(ktx_texture2, DestroyKtxTexture2));
      ktx_error_code != KTX_SUCCESS) {
    throw std::runtime_error{std::format("Failed to create KTX texture for {} with error {}",
                                         ktx_filepath.string(),
                                         ktxErrorString(ktx_error_code))};
  }

  if (ktxTexture2_NeedsTranscoding(ktx_texture2.get())) {
    const auto ktx_transcode_format =
        GetKtxTranscodeFormat(*ktx_texture2, has_unorm_format, supported_transcode_formats);
    if (const auto ktx_error_code = ktxTexture2_TranscodeBasis(ktx_texture2.get(), ktx_transcode_format, 0);
        ktx_error_code != KTX_SUCCESS) {
      throw std::runtime_error{std::format("Failed to transcode {} to {} with error {}",
                                           ktx_filepath.string(),
                                           ktxTranscodeFormatString(ktx_transcode_format),
                                           ktxErrorString(ktx_error_code))};
    }
  }

  return ktx_texture2;
}

UniqueKtxTexture2 CreateKtxTexture2FromImageFile(const std::filesystem::path& image_filepath,
                                                 const bool has_unorm_format) {
  static constexpr auto kRequiredImageChannels = 4;  // require RGBA for improved device compatibility
  const auto image_filepath_string = image_filepath.string();
  int image_width = 0;
  int image_height = 0;
  int image_channels = 0;

  const std::unique_ptr<stbi_uc, decltype(&stbi_image_free)> image_data{
      stbi_load(image_filepath_string.c_str(), &image_width, &image_height, &image_channels, kRequiredImageChannels),
      stbi_image_free};
  if (image_data == nullptr) {
    throw std::runtime_error{
        std::format("Failed to load {} with error {}", image_filepath_string, stbi_failure_reason())};
  }

  // R8G8B8A8 format support for images with VK_FORMAT_FEATURE_SAMPLED_IMAGE_BIT is mandated by the Vulkan specification
  const auto image_format = has_unorm_format ? vk::Format::eR8G8B8A8Unorm : vk::Format::eR8G8B8A8Srgb;
  ktxTextureCreateInfo ktx_texture_create_info{.vkFormat = static_cast<ktx_uint32_t>(image_format),
                                               .baseWidth = static_cast<ktx_uint32_t>(image_width),
                                               .baseHeight = static_cast<ktx_uint32_t>(image_height),
                                               .baseDepth = 1,
                                               .numDimensions = 2,
                                               .numLevels = 1,
                                               .numLayers = 1,
                                               .numFaces = 1};

  UniqueKtxTexture2 ktx_texture2{nullptr, nullptr};
  if (const auto result = ktxTexture2_Create(&ktx_texture_create_info,
                                             KTX_TEXTURE_CREATE_ALLOC_STORAGE,
                                             std::out_ptr(ktx_texture2, DestroyKtxTexture2));
      result != KTX_SUCCESS) {
    throw std::runtime_error{std::format("Failed to create KTX texture for {} with error {}",
                                         image_filepath_string,
                                         ktxErrorString(result))};
  }

  // TODO(matthew-rister): implement runtime mipmap generation for raw images
  const auto image_size_bytes = static_cast<ktx_size_t>(image_width) * image_height * kRequiredImageChannels;
  if (const auto result = ktxTexture_SetImageFromMemory(
          ktxTexture(ktx_texture2.get()),  // NOLINT(cppcoreguidelines-pro-type-cstyle-cast)
          0,
          0,
          KTX_FACESLICE_WHOLE_LEVEL,
          image_data.get(),  // image data is copied so ownership does not need to be transferred
          image_size_bytes);
      result != KTX_SUCCESS) {
    throw std::runtime_error{std::format("Failed to set KTX texture image for {} with error {}",
                                         image_filepath_string,
                                         ktxErrorString(result))};
  }

  return ktx_texture2;
}

UniqueKtxTexture2 CreateKtxTexture2(const cgltf_texture& gltf_texture,
                                    const std::filesystem::path& gltf_parent_filepath,
                                    const bool has_unorm_format,
                                    const std::unordered_set<vk::Format>& supported_transcode_formats) {
  const auto* const image = gltf_texture.has_basisu == 0 ? gltf_texture.image : gltf_texture.basisu_image;
  if (image == nullptr) throw std::runtime_error{std::format("No image source for texture {}", GetName(gltf_texture))};

  const auto texture_filepath = gltf_parent_filepath / image->uri;
  return texture_filepath.extension() == ".ktx2"
             ? CreateKtxTexture2FromKtxFile(texture_filepath, has_unorm_format, supported_transcode_formats)
             : CreateKtxTexture2FromImageFile(texture_filepath, has_unorm_format);
}

UniqueKtxTexture2 CreateBaseColorKtxTexture(const cgltf_material& gltf_material,
                                            const std::filesystem::path& gltf_parent_filepath,
                                            const std::unordered_set<vk::Format>& supported_transcode_formats) {
  if (gltf_material.has_pbr_metallic_roughness != 0) {
    if (const auto* const base_color_gltf_texture = gltf_material.pbr_metallic_roughness.base_color_texture.texture) {
      return CreateKtxTexture2(*base_color_gltf_texture, gltf_parent_filepath, false, supported_transcode_formats);
    }
  }
  return UniqueKtxTexture2{nullptr, nullptr};
}

gfx::Image CreateImage(const vk::Device device,
                       const vk::CommandBuffer command_buffer,
                       const VmaAllocator allocator,
                       ktxTexture2& ktx_texture2,
                       std::vector<gfx::Buffer>& staging_buffers) {
  static constexpr VmaAllocationCreateInfo kStagingBufferAllocationCreateInfo{
      .flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT,
      .usage = VMA_MEMORY_USAGE_AUTO};
  auto& staging_buffer = staging_buffers.emplace_back(ktx_texture2.dataSize,
                                                      vk::BufferUsageFlagBits::eTransferSrc,
                                                      allocator,
                                                      kStagingBufferAllocationCreateInfo);
  staging_buffer.CopyOnce<ktx_uint8_t>(std::span{ktx_texture2.pData, ktx_texture2.dataSize});

  static constexpr VmaAllocationCreateInfo kImageAllocationCreateInfo{.usage = VMA_MEMORY_USAGE_AUTO};
  gfx::Image image{device,
                   static_cast<vk::Format>(ktx_texture2.vkFormat),
                   vk::Extent2D{ktx_texture2.baseWidth, ktx_texture2.baseHeight},
                   ktx_texture2.numLevels,
                   vk::SampleCountFlagBits::e1,
                   vk::ImageUsageFlagBits::eSampled | vk::ImageUsageFlagBits::eTransferDst,
                   vk::ImageAspectFlagBits::eColor,
                   allocator,
                   kImageAllocationCreateInfo};

  const auto buffer_image_copies =
      std::views::iota(0u, ktx_texture2.numLevels)  //
      | std::views::transform([&ktx_texture2](const auto mip_level) {
          ktx_size_t image_offset = 0;
          if (const auto ktx_error_code =  // NOLINTNEXTLINE(cppcoreguidelines-pro-type-cstyle-cast)
              ktxTexture_GetImageOffset(ktxTexture(&ktx_texture2), mip_level, 0, 0, &image_offset);
              ktx_error_code != KTX_SUCCESS) {
            throw std::runtime_error{std::format("Failed to get image offset for mip level {} with error {}",
                                                 mip_level,
                                                 ktxErrorString(ktx_error_code))};
          }
          return vk::BufferImageCopy{
              .bufferOffset = image_offset,
              .imageSubresource = vk::ImageSubresourceLayers{.aspectMask = vk::ImageAspectFlagBits::eColor,
                                                             .mipLevel = mip_level,
                                                             .layerCount = 1},
              .imageExtent = vk::Extent3D{.width = ktx_texture2.baseWidth >> mip_level,
                                          .height = ktx_texture2.baseHeight >> mip_level,
                                          .depth = 1}};
        })
      | std::ranges::to<std::vector>();
  image.Copy(*staging_buffer, command_buffer, buffer_image_copies);

  return image;
}

vk::UniqueDescriptorPool CreateDescriptorPool(const vk::Device device, const std::uint32_t max_descriptor_sets) {
  static constexpr vk::DescriptorPoolSize kDescriptorPoolSize{.type = vk::DescriptorType::eCombinedImageSampler,
                                                              .descriptorCount = 1};

  return device.createDescriptorPoolUnique(vk::DescriptorPoolCreateInfo{.maxSets = max_descriptor_sets,
                                                                        .poolSizeCount = 1,
                                                                        .pPoolSizes = &kDescriptorPoolSize});
}

vk::UniqueDescriptorSetLayout CreateDescriptorSetLayout(const vk::Device device) {
  static constexpr vk::DescriptorSetLayoutBinding kDescriptorSetLayoutBinding{
      .descriptorType = vk::DescriptorType::eCombinedImageSampler,
      .descriptorCount = 1,
      .stageFlags = vk::ShaderStageFlagBits::eFragment};

  return device.createDescriptorSetLayoutUnique(
      vk::DescriptorSetLayoutCreateInfo{.bindingCount = 1, .pBindings = &kDescriptorSetLayoutBinding});
}

std::vector<vk::DescriptorSet> AllocateDescriptorSets(const vk::Device device,
                                                      const vk::DescriptorPool descriptor_pool,
                                                      const vk::DescriptorSetLayout& descriptor_set_layout,
                                                      const std::uint32_t descriptor_set_count) {
  const std::vector descriptor_set_layouts(descriptor_set_count, descriptor_set_layout);

  return device.allocateDescriptorSets(vk::DescriptorSetAllocateInfo{.descriptorPool = descriptor_pool,
                                                                     .descriptorSetCount = descriptor_set_count,
                                                                     .pSetLayouts = descriptor_set_layouts.data()});
}

void UpdateDescriptorSet(const vk::Device device,
                         const vk::ImageView image_view,
                         const vk::Sampler sampler,
                         const vk::DescriptorSet descriptor_set) {
  const vk::DescriptorImageInfo descriptor_image_info{.sampler = sampler,
                                                      .imageView = image_view,
                                                      .imageLayout = vk::ImageLayout::eShaderReadOnlyOptimal};

  device.updateDescriptorSets(vk::WriteDescriptorSet{.dstSet = descriptor_set,
                                                     .dstBinding = 0,
                                                     .dstArrayElement = 0,
                                                     .descriptorCount = 1,
                                                     .descriptorType = vk::DescriptorType::eCombinedImageSampler,
                                                     .pImageInfo = &descriptor_image_info},
                              nullptr);
}

vk::UniquePipelineLayout CreatePipelineLayout(const vk::Device device,
                                              const vk::DescriptorSetLayout& descriptor_set_layout) {
  static constexpr vk::PushConstantRange kPushConstantRange{.stageFlags = vk::ShaderStageFlagBits::eVertex,
                                                            .offset = 0,
                                                            .size = sizeof(PushConstants)};

  return device.createPipelineLayoutUnique(vk::PipelineLayoutCreateInfo{.setLayoutCount = 1,
                                                                        .pSetLayouts = &descriptor_set_layout,
                                                                        .pushConstantRangeCount = 1,
                                                                        .pPushConstantRanges = &kPushConstantRange});
}

vk::UniquePipeline CreatePipeline(const vk::Device device,
                                  const vk::Extent2D viewport_extent,
                                  const vk::SampleCountFlagBits msaa_sample_count,
                                  const vk::RenderPass render_pass,
                                  const vk::PipelineLayout pipeline_layout) {
  const std::filesystem::path vertex_shader_filepath{"assets/shaders/mesh.vert"};
  const gfx::ShaderModule vertex_shader_module{device, vk::ShaderStageFlagBits::eVertex, vertex_shader_filepath};

  const std::filesystem::path fragment_shader_filepath{"assets/shaders/mesh.frag"};
  const gfx::ShaderModule fragment_shader_module{device, vk::ShaderStageFlagBits::eFragment, fragment_shader_filepath};

  const std::array shader_stage_create_info{
      vk::PipelineShaderStageCreateInfo{.stage = vk::ShaderStageFlagBits::eVertex,
                                        .module = *vertex_shader_module,
                                        .pName = "main"},
      vk::PipelineShaderStageCreateInfo{.stage = vk::ShaderStageFlagBits::eFragment,
                                        .module = *fragment_shader_module,
                                        .pName = "main"}};

  static constexpr vk::VertexInputBindingDescription kVertexInputBindingDescription{
      .binding = 0,
      .stride = sizeof(Vertex),
      .inputRate = vk::VertexInputRate::eVertex};

  static constexpr std::array kVertexAttributeDescriptions{
      vk::VertexInputAttributeDescription{.location = 0,
                                          .binding = 0,
                                          .format = vk::Format::eR32G32B32Sfloat,
                                          .offset = offsetof(Vertex, position)},
      vk::VertexInputAttributeDescription{.location = 1,
                                          .binding = 0,
                                          .format = vk::Format::eR32G32B32Sfloat,
                                          .offset = offsetof(Vertex, normal)},
      vk::VertexInputAttributeDescription{.location = 2,
                                          .binding = 0,
                                          .format = vk::Format::eR32G32B32A32Sfloat,
                                          .offset = offsetof(Vertex, tangent)},
      vk::VertexInputAttributeDescription{.location = 3,
                                          .binding = 0,
                                          .format = vk::Format::eR32G32Sfloat,
                                          .offset = offsetof(Vertex, texture_coordinates0)}};

  static constexpr vk::PipelineVertexInputStateCreateInfo kVertexInputStateCreateInfo{
      .vertexBindingDescriptionCount = 1,
      .pVertexBindingDescriptions = &kVertexInputBindingDescription,
      .vertexAttributeDescriptionCount = static_cast<std::uint32_t>(kVertexAttributeDescriptions.size()),
      .pVertexAttributeDescriptions = kVertexAttributeDescriptions.data()};

  static constexpr vk::PipelineInputAssemblyStateCreateInfo kInputAssemblyStateCreateInfo{
      .topology = vk::PrimitiveTopology::eTriangleList};

  const vk::Viewport viewport{.x = 0.0f,
                              .y = 0.0f,
                              .width = static_cast<float>(viewport_extent.width),
                              .height = static_cast<float>(viewport_extent.height),
                              .minDepth = 0.0f,
                              .maxDepth = 1.0f};
  const vk::Rect2D scissor{.offset = vk::Offset2D{0, 0}, .extent = viewport_extent};

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

  auto [result, pipeline] = device.createGraphicsPipelineUnique(
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
                                     .layout = pipeline_layout,
                                     .renderPass = render_pass,
                                     .subpass = 0});
  vk::resultCheck(result, "Graphics pipeline creation failed");

  return std::move(pipeline);
}

vk::Sampler CreateSampler(const vk::Device device,
                          const std::uint32_t mip_levels,
                          const vk::PhysicalDeviceFeatures& physical_device_features,
                          const vk::PhysicalDeviceLimits& physical_device_limits,
                          std::unordered_map<vk::SamplerCreateInfo, vk::UniqueSampler>& unique_samplers) {
  // TODO(matthew-rister): apply glTF sampler attributes
  const vk::SamplerCreateInfo sampler_create_info{.magFilter = vk::Filter::eLinear,
                                                  .minFilter = vk::Filter::eLinear,
                                                  .mipmapMode = VULKAN_HPP_NAMESPACE::SamplerMipmapMode::eLinear,
                                                  .anisotropyEnable = physical_device_features.samplerAnisotropy,
                                                  .maxAnisotropy = physical_device_limits.maxSamplerAnisotropy,
                                                  .maxLod = static_cast<float>(mip_levels)};
  auto iterator = unique_samplers.find(sampler_create_info);
  if (iterator == unique_samplers.cend()) {
    std::tie(iterator, std::ignore) =
        unique_samplers.emplace(sampler_create_info, device.createSamplerUnique(sampler_create_info));
  }
  return *iterator->second;
}

}  // namespace

namespace gfx {

class Model::Node {
public:
  Node(const cgltf_scene& gltf_scene, std::unordered_map<const cgltf_mesh*, std::vector<Mesh>>& meshes);
  Node(const cgltf_node& gltf_node, std::unordered_map<const cgltf_mesh*, std::vector<Mesh>>& meshes);

  void Render(const glm::mat4& model_transform,
              const glm::mat4& view_transform,
              const glm::mat4& projection_transform,
              vk::PipelineLayout pipeline_layout,
              vk::CommandBuffer command_buffer) const;

private:
  std::vector<Mesh> meshes_;
  std::vector<std::unique_ptr<const Node>> children_;
  glm::mat4 transform_{1.0f};
};

struct Model::Material {
  const cgltf_material* gltf_material = nullptr;
  Image base_color_image;
  vk::Sampler base_color_sampler;
};

Model::Model(const std::filesystem::path& gltf_filepath,
             const vk::PhysicalDeviceFeatures& physical_device_features,
             const vk::PhysicalDeviceLimits& physical_device_limits,
             const vk::PhysicalDevice physical_device,
             const vk::Device device,
             const vk::Queue queue,
             const std::uint32_t queue_family_index,
             const vk::Extent2D viewport_extent,
             const vk::SampleCountFlagBits msaa_sample_count,
             const vk::RenderPass render_pass,
             const VmaAllocator allocator) {
  const auto gltf_data = ReadFile(gltf_filepath);
  const auto gltf_parent_filepath = gltf_filepath.parent_path();

  const auto supported_transcode_formats = GetSupportedTranscodeFormats(physical_device);
  auto material_futures =
      std::span{gltf_data->materials, gltf_data->materials_count}
      | std::views::transform([&gltf_parent_filepath, &supported_transcode_formats](const auto& gltf_material) {
          return std::async(std::launch::async, [&gltf_material, &gltf_parent_filepath, &supported_transcode_formats] {
            return std::pair{
                &gltf_material,
                CreateBaseColorKtxTexture(gltf_material, gltf_parent_filepath, supported_transcode_formats)};
          });
        })
      | std::ranges::to<std::vector>();

  const auto material_count = static_cast<std::uint32_t>(material_futures.size());
  descriptor_pool_ = CreateDescriptorPool(device, material_count);
  descriptor_set_layout_ = CreateDescriptorSetLayout(device);
  pipeline_layout_ = CreatePipelineLayout(device, *descriptor_set_layout_);
  pipeline_ = CreatePipeline(device, viewport_extent, msaa_sample_count, render_pass, *pipeline_layout_);

  const auto command_pool =
      device.createCommandPoolUnique(vk::CommandPoolCreateInfo{.flags = vk::CommandPoolCreateFlagBits::eTransient,
                                                               .queueFamilyIndex = queue_family_index});
  const auto command_buffers =
      device.allocateCommandBuffersUnique(vk::CommandBufferAllocateInfo{.commandPool = *command_pool,
                                                                        .level = vk::CommandBufferLevel::ePrimary,
                                                                        .commandBufferCount = 1});
  const auto command_buffer = *command_buffers.front();
  command_buffer.begin(vk::CommandBufferBeginInfo{.flags = vk::CommandBufferUsageFlagBits::eOneTimeSubmit});

  std::vector<Buffer> staging_buffers;  // staging buffers must remain in scope until copy commands complete
  staging_buffers.reserve(gltf_data->materials_count + gltf_data->meshes_count);

  materials_.reserve(material_count);
  for (auto& material_future : material_futures) {
    const auto [gltf_material, base_color_ktx_texture] = material_future.get();
    if (base_color_ktx_texture == nullptr) {
      std::println(std::cerr, "Unsupported material {}", GetName(*gltf_material));
      continue;  // TODO(matthew-rister): add support for other material types
    }
    materials_.emplace_back(gltf_material,
                            CreateImage(device, command_buffer, allocator, *base_color_ktx_texture, staging_buffers),
                            CreateSampler(device,
                                          base_color_ktx_texture->numLevels,
                                          physical_device_features,
                                          physical_device_limits,
                                          samplers_));
  }

  const auto descriptor_sets =
      std::views::zip_transform(
          [device](const auto& material, const auto& descriptor_set) {
            const auto& [gltf_material, base_color_image, base_color_sampler] = material;
            UpdateDescriptorSet(device, base_color_image.image_view(), base_color_sampler, descriptor_set);
            return std::pair{gltf_material, descriptor_set};
          },
          materials_,
          AllocateDescriptorSets(device, *descriptor_pool_, *descriptor_set_layout_, material_count))
      | std::ranges::to<std::unordered_map>();

  auto meshes =
      std::span{gltf_data->meshes, gltf_data->meshes_count}
      | std::views::transform([command_buffer, allocator, &staging_buffers, &descriptor_sets](const auto& gltf_mesh) {
          return std::pair{&gltf_mesh,
                           CreateMeshes(gltf_mesh, command_buffer, allocator, descriptor_sets, staging_buffers)};
        })
      | std::ranges::to<std::unordered_map>();

  command_buffer.end();

  const auto fence = device.createFenceUnique(vk::FenceCreateInfo{});
  queue.submit(vk::SubmitInfo{.commandBufferCount = 1, .pCommandBuffers = &command_buffer}, *fence);

  const auto gltf_scene = GetDefaultScene(*gltf_data, gltf_filepath);
  root_node_ = std::make_unique<const Node>(gltf_scene, meshes);

  static constexpr auto kMaxTimeout = std::numeric_limits<std::uint64_t>::max();
  const auto result = device.waitForFences(*fence, vk::True, kMaxTimeout);
  vk::resultCheck(result, "Fence failed to enter a signaled state");
}

Model::~Model() noexcept = default;  // this is necessary to enable forward declaring Model::Node with std::unique_ptr

void Model::Render(const Camera& camera, const vk::CommandBuffer command_buffer) const {
  static constexpr glm::mat4 kModelTransform{1.0f};
  command_buffer.bindPipeline(vk::PipelineBindPoint::eGraphics, *pipeline_);
  root_node_->Render(kModelTransform,
                     camera.GetViewTransform(),
                     camera.GetProjectionTransform(),
                     *pipeline_layout_,
                     command_buffer);
}

Model::Node::Node(const cgltf_scene& gltf_scene, std::unordered_map<const cgltf_mesh*, std::vector<Mesh>>& meshes)
    : children_{std::span{gltf_scene.nodes, gltf_scene.nodes_count}
                | std::views::transform([&meshes](const auto* const gltf_node) {
                    assert(gltf_node != nullptr);  // valid glTF scenes should not contain null scene nodes
                    return std::make_unique<const Node>(*gltf_node, meshes);
                  })
                | std::ranges::to<std::vector>()} {}

Model::Node::Node(const cgltf_node& gltf_node, std::unordered_map<const cgltf_mesh*, std::vector<Mesh>>& meshes)
    : meshes_{GetMeshes(gltf_node, meshes)},
      children_{std::span{gltf_node.children, gltf_node.children_count}
                | std::views::transform([&meshes](const auto* const child_gltf_node) {
                    assert(child_gltf_node != nullptr);  // valid glTF nodes should not contain null child nodes
                    return std::make_unique<const Node>(*child_gltf_node, meshes);
                  })
                | std::ranges::to<std::vector>()},
      transform_{GetTransform(gltf_node)} {}

void Model::Node::Render(const glm::mat4& model_transform,
                         const glm::mat4& view_transform,
                         const glm::mat4& projection_transform,
                         const vk::PipelineLayout pipeline_layout,
                         const vk::CommandBuffer command_buffer) const {
  const auto node_transform = model_transform * transform_;
  command_buffer.pushConstants<PushConstants>(pipeline_layout,
                                              vk::ShaderStageFlagBits::eVertex,
                                              0,
                                              PushConstants{.model_view_transform = view_transform * node_transform,
                                                            .projection_transform = projection_transform});

  for (const auto& mesh : meshes_) {
    mesh.Render(pipeline_layout, command_buffer);
  }

  for (const auto& child_node : children_) {
    child_node->Render(node_transform, view_transform, projection_transform, pipeline_layout, command_buffer);
  }
}

}  // namespace gfx
