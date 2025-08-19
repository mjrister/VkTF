module;

#include <cassert>
#include <concepts>
#include <cstdint>
#include <filesystem>
#include <format>
#include <memory>
#include <optional>
#include <ranges>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <type_traits>
#include <unordered_map>
#include <utility>
#include <variant>
#include <vector>

#include <cgltf.h>
#include <glm/glm.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <vulkan/vulkan.hpp>

export module gltf_asset;

import bounding_box;
import log;

namespace vktf::gltf {

/** @brief A structure representing glTF sampler properties. */
export struct [[nodiscard]] Sampler {
  /** @brief The user-defined name for the sampler. */
  std::optional<std::string> name;

  /** @brief The magnification filter for sampling nearby textures. */
  vk::Filter mag_filter;

  /** @brief The minification filter for sampling distant textures. */
  vk::Filter min_filter;

  /** @brief The mipmap mode for sampling texture mipmaps. */
  vk::SamplerMipmapMode mipmap_mode;

  /** @brief The address mode for the u texture coordinate. */
  vk::SamplerAddressMode address_mode_u;

  /** @brief The address mode for the v texture coordinate. */
  vk::SamplerAddressMode address_mode_v;
};

/** @brief A type alias for a @c unique_ptr that manages the lifetime of a @ref Sampler. */
export using UniqueSampler = std::unique_ptr<const Sampler>;

/** @brief A structure representing glTF texture properties. */
export struct [[nodiscard]] Texture {
  /** @brief The user-defined name for the texture. */
  std::optional<std::string> name;

  /** @brief The filepath to the texture image data. */
  std::optional<std::filesystem::path> filepath;

  /** @brief A non-owning pointer to the texture sampler. */
  const Sampler* sampler = nullptr;
};

/** @brief A type alias for a @c unique_ptr that manages the lifetime of a @ref Texture. */
export using UniqueTexture = std::unique_ptr<const Texture>;

/** @brief A structure representing glTF PBR metallic-roughness properties. */
export struct [[nodiscard]] PbrMetallicRoughness {
  /**
   * @brief The factors for the base color of the material.
   * @details If @ref base_color_texture is specified, this property acts as a linear multiplier with the sampled texel;
   *          otherwise, it defines the absolute base color value for the material.
   */
  glm::vec4 base_color_factor{0.0f};

  /** @brief A non-owning pointer to the base color texture. */
  const Texture* base_color_texture = nullptr;

  /**
   * @brief The factor for the metalness of the material.
   * @details If @ref metallic_roughness_texture is specified, this value acts as a linear multiplier with the blue (B)
   *          channel of the sampled texel; otherwise, it defines the absolute metallic value for the material.
   */
  float metallic_factor = 0.0f;

  /**
   * @brief The factor for the roughness of the material.
   * @details If @ref metallic_roughness_texture is specified, this value acts as a linear multiplier with the green (G)
   *          channel of the sampled texel; otherwise, it defines the absolute roughness value for the material.
   */
  float roughness_factor = 0.0f;

  /** @brief A non-owning pointer to the metallic-roughness texture. */
  const Texture* metallic_roughness_texture = nullptr;
};

/** @brief A structure representing glTF material properties. */
export struct [[nodiscard]] Material {
  /** @brief The user-defined name for the material. */
  std::optional<std::string> name;

  /** @brief The PBR metallic-roughness properties for the material. */
  std::optional<PbrMetallicRoughness> pbr_metallic_roughness;

  /** @brief The amount to scale sampled normals in the x/y directions. */
  float normal_scale = 0.0f;

  /** @brief A non-owning pointer to the normal texture. */
  const Texture* normal_texture = nullptr;
};

/** @brief A type alias for a @c unique_ptr that manages the lifetime of a @ref Material. */
export using UniqueMaterial = std::unique_ptr<const Material>;

/** @brief A structure representing glTF attributes for a mesh primitive. */
export struct [[nodiscard]] Attributes {
  /** @brief A structure representing the position glTF attribute. */
  struct Position {
    /** @brief The name for the position glTF attribute. */
    static constexpr std::string_view kName = "POSITION";

    /** @brief A type alias for a vector of 3D positions. */
    using Data = std::vector<glm::vec3>;

    /** @brief The position attribute data. */
    Data data;

    /** @brief The bounding box that encloses all positions in this attribute. */
    BoundingBox bounding_box;
  };

  /** @brief A structure representing the normal glTF attribute. */
  struct Normal {
    /** @brief The name for the normal glTF attribute. */
    static constexpr std::string_view kName = "NORMAL";

    /** @brief A type alias for a vector of 3D normals. */
    using Data = std::vector<glm::vec3>;

    /** @brief The normal attribute data. */
    std::optional<Data> data;
  };

  /** @brief A structure representing the tangent glTF attribute. */
  struct Tangent {
    /** @brief The name for the tangent glTF attribute. */
    static constexpr std::string_view kName = "TANGENT";

    /**
     * @brief A type alias for a vector of 4D tangents.
     * @details The w-component indicates the signed handedness of the tangent basis vector.
     */
    using Data = std::vector<glm::vec4>;

    /** @brief The tangent attribute data. */
    std::optional<Data> data;
  };

  /** @brief A structure representing the first texture coordinate set glTF attribute. */
  struct TexCoord0 {
    /** @brief The name for the first texture coordinates set glTF attribute. */
    static constexpr std::string_view kName = "TEXCOORD_0";

    /** @brief A type alias for a vector of 2D texture coordinates. */
    using Data = std::vector<glm::vec2>;

    /** @brief The first texture coordinates set attribute data. */
    std::optional<Data> data;
  };

  /** @brief The position glTF attribute. */
  Position position;

  /** @brief The normal glTF attribute. */
  Normal normal;

  /** @brief The tangent glTF attribute. */
  Tangent tangent;

  /** @brief The first texture coordinates set glTF attribute. */
  TexCoord0 texcoord_0;
};

/** @brief A structure representing glTF mesh primitive properties. */
export struct [[nodiscard]] Primitive {
  /** @brief A type alias for a @c std::variant that represents variable-length vertex indices. */
  using Indices = std::variant<std::vector<std::uint8_t>, std::vector<std::uint16_t>, std::vector<std::uint32_t>>;

  /**
   * @brief The primitive topology.
   * @details The primitive topology defines how vertices are interpreted to form geometric primitives (e.g., points,
   *          lines, triangles). When @ref indices is specified, sequential index values are interpreted based on the
   *          chosen topology; otherwise, sequential vertices in @ref attributes are used instead.
   */
  static constexpr auto kTopology = vk::PrimitiveTopology::eTriangleList;  // TODO: add support for other topologies

  /** @brief The vertex attributes (e.g., position, normal, tangent). */
  Attributes attributes;

  /** @brief The variable-length vertex indices. */
  std::optional<Indices> indices;

  /** @brief A non-owning pointer to the primitive material. */
  const Material* material = nullptr;
};

/** @brief A structure representing glTF mesh properties. */
export struct [[nodiscard]] Mesh {
  /** @brief The user-defined name for the mesh. */
  std::optional<std::string> name;

  /** @brief The glTF primitives that compose the mesh. */
  std::vector<Primitive> primitives;
};

/** @brief A type alias for a @c unique_ptr that manages the lifetime of a @ref Mesh. */
export using UniqueMesh = std::unique_ptr<const Mesh>;

/** @brief A structure representing glTF light properties. */
export struct [[nodiscard]] Light {
  /** @brief The glTF light type. */
  enum class Type : std::uint8_t { kDirectional, kPoint };  // TODO: add support for glTF spot lights

  /** @brief The user-defined name for the light. */
  std::optional<std::string> name;

  /** @brief The linear-space light color. */
  glm::vec3 color{0.0f};

  /** @brief The light type. */
  Type type = Type::kDirectional;
};

/** @brief A type alias for a @c unique_ptr that manages the lifetime of a @ref Light. */
export using UniqueLight = std::unique_ptr<const Light>;

/** @brief A structure representing glTF node properties. */
export struct [[nodiscard]] Node {
  /** @brief The user-defined name for the node. */
  std::optional<std::string> name;

  /** @brief The local transform representing the node position and orientation relative to its parent. */
  glm::mat4 local_transform{0.0f};

  /** @brief The non-owning pointer to the node mesh. */
  const Mesh* mesh = nullptr;

  /** @brief The non-owning pointer to the node light. */
  const Light* light = nullptr;

  /** @brief A list of non-owning pointers to the node children. */
  std::vector<const Node*> children;
};

/** @brief A type alias for a @c unique_ptr that manages the lifetime of a @ref Node. */
export using UniqueNode = std::unique_ptr<const Node>;

/** @brief A structure representing glTF scene properties. */
export struct [[nodiscard]] Scene {
  /** @brief The user-defined name for the scene. */
  std::optional<std::string> name;

  /** @brief A list of non-owning pointers to the root nodes in the scene. */
  std::vector<const Node*> root_nodes;
};

/** @brief A type alias for a @c unique_ptr that manages the lifetime of a @ref Scene. */
export using UniqueScene = std::unique_ptr<const Scene>;

/** @brief A structure representing a top-level container for a glTF asset. */
export struct [[nodiscard]] Asset {
  /** @brief The filename of the glTF asset. */
  std::string name;

  /** @brief A list of glTF samplers. */
  std::vector<UniqueSampler> samplers;

  /** @brief A list of glTF textures. */
  std::vector<UniqueTexture> textures;

  /** @brief A list of glTF materials. */
  std::vector<UniqueMaterial> materials;

  /** @brief A list of glTF meshes. */
  std::vector<UniqueMesh> meshes;

  /** @brief A list of glTF lights. */
  std::vector<UniqueLight> lights;

  /** @brief A list of glTF nodes. */
  std::vector<UniqueNode> nodes;

  /** @brief A list of glTF scenes. */
  std::vector<UniqueScene> scenes;

  /** @brief A non-owning pointer to the default glTF scene. */
  const Scene* default_scene = nullptr;
};

/**
 * @brief Loads a glTF file.
 * @details This function parses a glTF 2.0 file to create an in-memory representation of a glTF asset.
 * @param gltf_filepath The filepath of the glTF asset to load.
 * @param log The log for writing messages when loading a glTF file.
 * @return An in-memory representation of a glTF asset loaded from @ref gltf_filepath.
 * @throws std::runtime_error Thrown if the glTF asset at @ref gltf_filepath is invalid or unsupported.
 * @see https://registry.khronos.org/glTF/specs/2.0/glTF-2.0.html glTF 2.0 Specification
 */
export [[nodiscard]] Asset Load(const std::filesystem::path& gltf_filepath, Log& log);

}  // namespace vktf::gltf

module :private;

// =====================================================================================================================
// Formatters
// =====================================================================================================================

template <>
struct std::formatter<cgltf_result> : std::formatter<std::string_view> {
  [[nodiscard]] auto format(const cgltf_result cgltf_result, auto& format_context) const {
    return std::formatter<std::string_view>::format(to_string(cgltf_result), format_context);
  }

private:
  static std::string_view to_string(const cgltf_result cgltf_result) noexcept {
    switch (cgltf_result) {
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
  [[nodiscard]] auto format(const cgltf_primitive_type cgltf_primitive_type, auto& format_context) const {
    return std::formatter<std::string_view>::format(to_string(cgltf_primitive_type), format_context);
  }

private:
  static std::string_view to_string(const cgltf_primitive_type cgltf_primitive_type) noexcept {
    switch (cgltf_primitive_type) {
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
  [[nodiscard]] auto format(const cgltf_light_type cgltf_light_type, auto& format_context) const {
    return std::formatter<std::string_view>::format(to_string(cgltf_light_type), format_context);
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

namespace vktf::gltf {

namespace {

// =====================================================================================================================
// Utilities
// =====================================================================================================================

using Severity = Log::Severity;

template <typename Key, typename Value>
using CgltfResourceMap = std::unordered_map<const Key*, std::unique_ptr<Value>>;

template <typename Key, typename Value>
const Value* Get(const Key* const key, const CgltfResourceMap<Key, Value>& cgltf_resource_map) {
  if (key == nullptr) return nullptr;
  const auto iterator = cgltf_resource_map.find(key);
  assert(iterator != cgltf_resource_map.cend());  // resource maps are constructed in advance with all known glTF keys
  return iterator->second.get();
}

template <typename Key, typename Value>
std::vector<std::unique_ptr<Value>> GetValues(CgltfResourceMap<Key, Value>&& cgltf_resource_map) {
  // clang-format off
  auto values = cgltf_resource_map
                | std::views::values
                | std::views::filter([](auto& value) { return value != nullptr; })  // skip unsupported values
                | std::views::as_rvalue
                | std::ranges::to<std::vector>();
  // clang-format on
  cgltf_resource_map.clear();  // empty resource map after its values have been moved
  return values;
}

template <typename T>
  requires std::convertible_to<decltype(T::name), const char*>
std::optional<std::string> GetName(const T& cgltf_element) {
  return cgltf_element.name == nullptr ? std::nullopt : std::optional<std::string>{cgltf_element.name};
}

template <typename T>
std::string GetNameOrDefault(const T& cgltf_element) {
  static constexpr std::string_view kDefaultName = "undefined";
  return GetName(cgltf_element).value_or(std::string{kDefaultName});
}

// =====================================================================================================================
// glTF File
// =====================================================================================================================

using UniqueCgltfData = std::unique_ptr<cgltf_data, decltype(&cgltf_free)>;

UniqueCgltfData LoadGltfFile(const std::string& gltf_filepath) {
  static constexpr cgltf_options kDefaultOptions{};
  UniqueCgltfData cgltf_data{nullptr, cgltf_free};

  if (const auto cgltf_result = cgltf_parse_file(&kDefaultOptions, gltf_filepath.c_str(), std::out_ptr(cgltf_data));
      cgltf_result != cgltf_result_success) {
    throw std::runtime_error{std::format("Failed to parse {} with error {}", gltf_filepath, cgltf_result)};
  }
#ifndef NDEBUG
  if (const auto cgltf_result = cgltf_validate(cgltf_data.get()); cgltf_result != cgltf_result_success) {
    throw std::runtime_error{std::format("Failed to validate {} with error {}", gltf_filepath, cgltf_result)};
  }
#endif
  if (const auto cgltf_result = cgltf_load_buffers(&kDefaultOptions, cgltf_data.get(), gltf_filepath.c_str());
      cgltf_result != cgltf_result_success) {
    throw std::runtime_error{std::format("Failed to load buffers for {} with error {}", gltf_filepath, cgltf_result)};
  }

  return cgltf_data;
}

// =====================================================================================================================
// Samplers
// =====================================================================================================================

// filter and address mode values come from https://registry.khronos.org/glTF/specs/2.0/glTF-2.0.html#reference-sampler
enum class SamplerFilter : std::uint16_t {
  kDefault = 0,  // the glTF specification allows client implementation to set default filter values when undefined
  kNearest = 9728,
  kLinear = 9729,
  kNearestMipmapNearest = 9984,
  kLinearMipmapNearest = 9985,
  kNearestMipmapLinear = 9986,
  kLinearMipmapLinear = 9987
};

enum class SamplerAddressMode : std::uint16_t {
  kClampToEdge = 33071,
  kMirroredRepeat = 33648,
  kRepeat = 10497,
  kDefault = kRepeat
};

constexpr vk::Filter GetSamplerMagFilter(const cgltf_int cgltf_mag_filter) {
  switch (static_cast<SamplerFilter>(cgltf_mag_filter)) {
    using enum SamplerFilter;
    case kNearest:
      return vk::Filter::eNearest;
    case kDefault:
    case kLinear:
      return vk::Filter::eLinear;
    default:
      throw std::runtime_error{std::format("Invalid glTF sampler with bad magnification filter {}", cgltf_mag_filter)};
  }
}

constexpr std::pair<vk::Filter, vk::SamplerMipmapMode> GetSamplerMinFilter(const cgltf_int cgltf_min_filter) {
  switch (static_cast<SamplerFilter>(cgltf_min_filter)) {
    using enum SamplerFilter;
    case kNearest:
    case kNearestMipmapNearest:
      return std::pair{vk::Filter::eNearest, vk::SamplerMipmapMode::eNearest};
    case kNearestMipmapLinear:
      return std::pair{vk::Filter::eNearest, vk::SamplerMipmapMode::eLinear};
    case kLinear:
    case kLinearMipmapNearest:
      return std::pair{vk::Filter::eLinear, vk::SamplerMipmapMode::eNearest};
    case kDefault:
    case kLinearMipmapLinear:
      return std::pair{vk::Filter::eLinear, vk::SamplerMipmapMode::eLinear};
    default:
      throw std::runtime_error{std::format("Invalid glTF sampler with bad minification filter {}", cgltf_min_filter)};
  }
}

constexpr vk::SamplerAddressMode GetSamplerAddressMode(const cgltf_int cgltf_wrap_mode) {
  switch (static_cast<SamplerAddressMode>(cgltf_wrap_mode)) {
    using enum SamplerAddressMode;
    case kClampToEdge:
      return vk::SamplerAddressMode::eClampToEdge;
    case kMirroredRepeat:
      return vk::SamplerAddressMode::eMirroredRepeat;
    case kRepeat:
      return vk::SamplerAddressMode::eRepeat;
    default:
      throw std::runtime_error{std::format("Invalid glTF sampler with bad wrap mode {}", cgltf_wrap_mode)};
  }
}

constexpr Sampler CreateDefaultSampler() {
  const auto [min_filter, mipmap_mode] = GetSamplerMinFilter(std::to_underlying(SamplerFilter::kDefault));
  const auto address_mode = GetSamplerAddressMode(std::to_underlying(SamplerAddressMode::kDefault));

  return Sampler{.mag_filter = GetSamplerMagFilter(std::to_underlying(SamplerFilter::kDefault)),
                 .min_filter = min_filter,
                 .mipmap_mode = mipmap_mode,
                 .address_mode_u = address_mode,
                 .address_mode_v = address_mode};
}

UniqueSampler CreateSampler(const cgltf_sampler& cgltf_sampler) {
  const auto [min_filter, mipmap_mode] = GetSamplerMinFilter(cgltf_sampler.min_filter);

  return std::make_unique<const Sampler>(GetName(cgltf_sampler),
                                         GetSamplerMagFilter(cgltf_sampler.mag_filter),
                                         min_filter,
                                         mipmap_mode,
                                         GetSamplerAddressMode(cgltf_sampler.wrap_s),
                                         GetSamplerAddressMode(cgltf_sampler.wrap_t));
}

CgltfResourceMap<cgltf_sampler, const Sampler> CreateSamplers(const std::span<const cgltf_sampler> cgltf_samplers) {
  return cgltf_samplers  //
         | std::views::transform(
             [](const auto& cgltf_sampler) { return std::pair{&cgltf_sampler, CreateSampler(cgltf_sampler)}; })
         | std::ranges::to<std::unordered_map>();
}

// =====================================================================================================================
// Textures
// =====================================================================================================================

std::optional<std::filesystem::path> GetImageUri(const cgltf_texture& cgltf_texture,
                                                 const std::filesystem::path& gltf_directory) {
  const auto* const cgltf_image = cgltf_texture.has_basisu == 0 ? cgltf_texture.image : cgltf_texture.basisu_image;
  if (cgltf_image == nullptr) return std::nullopt;

  const std::filesystem::path cgltf_image_uri = cgltf_image->uri == nullptr ? "" : cgltf_image->uri;
  if (cgltf_image_uri.empty()) return std::nullopt;

  return gltf_directory / cgltf_image_uri;
}

UniqueTexture CreateTexture(const cgltf_texture& cgltf_texture,
                            const std::filesystem::path& gltf_directory,
                            const CgltfResourceMap<cgltf_sampler, const Sampler>& samplers) {
  const auto* sampler = Get(cgltf_texture.sampler, samplers);
  if (sampler == nullptr) {
    static constexpr auto kDefaultSampler = CreateDefaultSampler();
    sampler = &kDefaultSampler;
  }

  return std::make_unique<const Texture>(GetName(cgltf_texture), GetImageUri(cgltf_texture, gltf_directory), sampler);
}

CgltfResourceMap<cgltf_texture, const Texture> CreateTextures(
    const std::span<const cgltf_texture> cgltf_textures,
    const std::filesystem::path& gltf_directory,
    const CgltfResourceMap<cgltf_sampler, const Sampler>& samplers) {
  return cgltf_textures  //
         | std::views::transform([&gltf_directory, &samplers](const auto& cgltf_texture) {
             return std::pair{&cgltf_texture, CreateTexture(cgltf_texture, gltf_directory, samplers)};
           })
         | std::ranges::to<std::unordered_map>();
}

// =====================================================================================================================
// Materials
// =====================================================================================================================

std::optional<PbrMetallicRoughness> GetPbrMetallicRoughness(
    const cgltf_material& cgltf_material,
    const CgltfResourceMap<cgltf_texture, const Texture>& textures) {
  if (cgltf_material.has_pbr_metallic_roughness == 0) return std::nullopt;

  const auto& [base_color_texture_view,
               metallic_roughness_texture_view,
               base_color_factor,
               metallic_factor,
               roughness_factor] = cgltf_material.pbr_metallic_roughness;

  return PbrMetallicRoughness{.base_color_factor = glm::make_vec4(base_color_factor),
                              .base_color_texture = Get(base_color_texture_view.texture, textures),
                              .metallic_factor = metallic_factor,
                              .roughness_factor = roughness_factor,
                              .metallic_roughness_texture = Get(metallic_roughness_texture_view.texture, textures)};
}

UniqueMaterial CreateMaterial(const cgltf_material& cgltf_material,
                              const CgltfResourceMap<cgltf_texture, const Texture>& textures) {
  const auto normal_texture_view = cgltf_material.normal_texture;

  return std::make_unique<const Material>(GetName(cgltf_material),
                                          GetPbrMetallicRoughness(cgltf_material, textures),
                                          normal_texture_view.scale,
                                          Get(normal_texture_view.texture, textures));
}

CgltfResourceMap<cgltf_material, const Material> CreateMaterials(
    const std::span<const cgltf_material>& cgltf_materials,
    const CgltfResourceMap<cgltf_texture, const Texture>& textures) {
  return cgltf_materials  //
         | std::views::transform([&textures](const auto& cgltf_material) {
             return std::pair{&cgltf_material, CreateMaterial(cgltf_material, textures)};
           })
         | std::ranges::to<std::unordered_map>();
}

// =====================================================================================================================
// Meshes
// =====================================================================================================================

template <typename T, glm::length_t N>
  requires std::constructible_from<glm::vec<N, T>>
using AttributeData = std::vector<glm::vec<N, T>>;

template <glm::length_t N>
AttributeData<float, N> UnpackFloats(const cgltf_accessor& cgltf_accessor) {
  if (const auto component_count = cgltf_num_components(cgltf_accessor.type); component_count != N) {
    throw std::runtime_error{std::format("Invalid glTF primitive attribute {} with bad component count {}",
                                         GetNameOrDefault(cgltf_accessor),
                                         component_count)};
  }
  AttributeData<float, N> attribute_data(cgltf_accessor.count);
  if (const auto float_count = N * cgltf_accessor.count;
      cgltf_accessor_unpack_floats(&cgltf_accessor, glm::value_ptr(attribute_data.front()), float_count) == 0) {
    throw std::runtime_error{std::format("Failed to unpack floats for accessor {}", GetNameOrDefault(cgltf_accessor))};
  }
  return attribute_data;
}

template <glm::length_t N>
bool TryUnpackFloats(const cgltf_attribute& cgltf_attribute,
                     const std::string_view attribute_name,
                     std::optional<AttributeData<float, N>>& attribute_data) {
  if (const auto cgltf_attribute_name = GetName(cgltf_attribute);
      !cgltf_attribute_name.has_value() || attribute_name != *cgltf_attribute_name) {
    // attribute sets share the same attribute type so their name must be checked to ensure data is unpacked correctly
    return false;
  }
  if (attribute_data.has_value()) {
    // the glTF specification uses attribute names as JSON keys which must be unique for a glTF primitive
    throw std::runtime_error{std::format("Duplicate glTF primitive attribute {}", attribute_name)};
  }
  assert(cgltf_attribute.data != nullptr);  // assume valid cgltf accessor pointer
  attribute_data = UnpackFloats<N>(*cgltf_attribute.data);
  return true;
}

template <typename AttributeData>
void ValidateOptionalAttribute(const std::size_t position_count, const std::optional<AttributeData>& attribute_data) {
  if (attribute_data.has_value() && position_count != attribute_data->size()) {
    // the glTF specification requires all primitive attributes to have the same accessor count
    throw std::runtime_error{
        std::format("Invalid glTF primitive attribute with bad accessor count {}", attribute_data->size())};
  }
}

template <typename... AttributeData>
void ValidateOptionalAttributes(const std::size_t position_count,
                                const std::optional<AttributeData>&... attribute_data) {
  (ValidateOptionalAttribute(position_count, attribute_data), ...);
}

std::optional<Attributes> CreateAttributes(const std::span<const cgltf_attribute> cgltf_attributes, Log& log) {
  using Position = Attributes::Position;
  std::optional<Position::Data> position_data;
  BoundingBox bounding_box;

  using Normal = Attributes::Normal;
  std::optional<Normal::Data> normal_data;

  using Tangent = Attributes::Tangent;
  std::optional<Tangent::Data> tangent_data;

  using TexCoord0 = Attributes::TexCoord0;
  std::optional<TexCoord0::Data> texcoord_0_data;

  for (const auto& cgltf_attribute : cgltf_attributes) {
    switch (cgltf_attribute.type) {
      case cgltf_attribute_type_position:
        if (TryUnpackFloats(cgltf_attribute, Position::kName, position_data)) {
          // the glTF specification requires the min/max properties to be defined for the position attribute accessor
          const auto& position_accessor = *cgltf_attribute.data;
          bounding_box.min = glm::make_vec3(position_accessor.min);
          bounding_box.max = glm::make_vec3(position_accessor.max);
          continue;
        }
        break;
      case cgltf_attribute_type_normal:
        if (TryUnpackFloats(cgltf_attribute, Normal::kName, normal_data)) {
          continue;
        }
        break;
      case cgltf_attribute_type_tangent:
        if (TryUnpackFloats(cgltf_attribute, Tangent::kName, tangent_data)) {
          continue;
        }
        break;
      case cgltf_attribute_type_texcoord:
        if (TryUnpackFloats(cgltf_attribute, TexCoord0::kName, texcoord_0_data)) {
          continue;
        }
        break;
      default:
        break;
    }
    // TODO: add support for at least two texture coordinate sets, one vertex color, and one joints/weights set
    log(Severity::kError) << std::format("Unsupported primitive attribute {}", GetNameOrDefault(cgltf_attribute));
  }

  return position_data.transform(
      [&bounding_box, &normal_data, &tangent_data, &texcoord_0_data](auto& position_data_value) {
        const auto position_count = position_data_value.size();
        ValidateOptionalAttributes(position_count, normal_data, tangent_data, texcoord_0_data);

        return Attributes{.position = Position{.data = std::move(position_data_value), .bounding_box = bounding_box},
                          .normal = Normal{.data = std::move(normal_data)},
                          .tangent = Tangent{.data = std::move(tangent_data)},
                          .texcoord_0 = TexCoord0{.data = std::move(texcoord_0_data)}};
      });
}

template <typename T>
  requires std::same_as<T, std::uint8_t> || std::same_as<T, std::uint16_t> || std::same_as<T, std::uint32_t>
std::vector<T> UnpackIndices(const cgltf_accessor& cgltf_accessor) {
  std::vector<T> indices(cgltf_accessor.count);
  if (const auto index_size_bytes = sizeof(T);
      cgltf_accessor_unpack_indices(&cgltf_accessor, indices.data(), index_size_bytes, indices.size()) == 0) {
    throw std::runtime_error{std::format("Failed to unpack indices for accessor {}", GetNameOrDefault(cgltf_accessor))};
  }
  return indices;
}

std::optional<Primitive::Indices> CreateIndices(const cgltf_accessor* const cgltf_accessor) {
  if (cgltf_accessor == nullptr) return std::nullopt;

  switch (cgltf_component_size(cgltf_accessor->component_type)) {
    case 1:
      return UnpackIndices<std::uint8_t>(*cgltf_accessor);
    case 2:
      return UnpackIndices<std::uint16_t>(*cgltf_accessor);
    case 4:
      return UnpackIndices<std::uint32_t>(*cgltf_accessor);
    default:
      // the glTF specification only supports 8, 16, and 32-bit unsigned indices
      throw std::runtime_error{std::format("Invalid glTF primitive indices with bad component size {}",
                                           cgltf_component_size(cgltf_accessor->component_type))};
  }
}

UniqueMesh CreateMesh(const cgltf_mesh& cgltf_mesh,
                      const CgltfResourceMap<cgltf_material, const Material>& materials,
                      Log& log) {
  std::vector<Primitive> primitives;
  primitives.reserve(cgltf_mesh.primitives_count);

  for (const auto& [index, cgltf_primitive] :
       std::span{cgltf_mesh.primitives, cgltf_mesh.primitives_count} | std::views::enumerate) {
    if (cgltf_primitive.type != cgltf_primitive_type_triangles) {
      log(Severity::kError) << std::format("Failed to create mesh primitive {}[{}] with unsupported type {}",
                                           GetNameOrDefault(cgltf_mesh),
                                           index,
                                           cgltf_primitive.type);
      continue;  // TODO: add support for other primitive types
    }

    auto attributes = CreateAttributes(std::span{cgltf_primitive.attributes, cgltf_primitive.attributes_count}, log);
    if (!attributes.has_value()) {
      log(Severity::kError) << std::format("Failed to create mesh primitive {}[{}] with missing position attribute",
                                           GetNameOrDefault(cgltf_mesh),
                                           index);
      continue;  // skip mesh primitive with missing position attribute
    }

    primitives.emplace_back(
        std::move(*attributes),
        CreateIndices(cgltf_primitive.indices),  // TODO: validate index count for the primitive topology
        Get(cgltf_primitive.material, materials));
  }

  return primitives.empty() ? nullptr : std::make_unique<const Mesh>(GetName(cgltf_mesh), std::move(primitives));
}

CgltfResourceMap<cgltf_mesh, const Mesh> CreateMeshes(const std::span<const cgltf_mesh>& cgltf_meshes,
                                                      const CgltfResourceMap<cgltf_material, const Material>& materials,
                                                      Log& log) {
  return cgltf_meshes  //
         | std::views::transform([&materials, &log](const auto& cgltf_mesh) {
             return std::pair{&cgltf_mesh, CreateMesh(cgltf_mesh, materials, log)};
           })
         | std::ranges::to<std::unordered_map>();
}

// =====================================================================================================================
// Lights
// =====================================================================================================================

std::optional<Light::Type> GetLightType(const cgltf_light& cgltf_light, Log& log) {
  switch (cgltf_light.type) {
    case cgltf_light_type_directional:
      return Light::Type::kDirectional;
    case cgltf_light_type_point:
      return Light::Type::kPoint;
    default:
      log(Severity::kError) << std::format("Failed to create light {} with unsupported type {}",
                                           GetNameOrDefault(cgltf_light),
                                           cgltf_light.type);
      return std::nullopt;  // TODO: add support for other light types
  }
}

UniqueLight CreateLight(const cgltf_light& cgltf_light, Log& log) {
  if (const auto light_type = GetLightType(cgltf_light, log); light_type.has_value()) {
    return std::make_unique<const Light>(GetName(cgltf_light), glm::make_vec4(cgltf_light.color), *light_type);
  }
  return nullptr;
}

CgltfResourceMap<cgltf_light, const Light> CreateLights(const std::span<const cgltf_light>& cgltf_lights, Log& log) {
  return cgltf_lights  //
         | std::views::transform(
             [&log](const auto& cgltf_light) { return std::pair{&cgltf_light, CreateLight(cgltf_light, log)}; })
         | std::ranges::to<std::unordered_map>();
}

// =====================================================================================================================
// Nodes
// =====================================================================================================================

glm::mat4 GetLocalTransform(const cgltf_node& cgltf_node) {
  glm::mat4 local_transform{0.0f};
  cgltf_node_transform_local(&cgltf_node, glm::value_ptr(local_transform));
  return local_transform;
}

std::vector<const Node*> GetChildren(const cgltf_node& cgltf_parent_node,
                                     const CgltfResourceMap<cgltf_node, Node>& mutable_nodes) {
  return std::span{cgltf_parent_node.children, cgltf_parent_node.children_count}
         | std::views::transform([&mutable_nodes](const auto* const cgltf_child_node) {
             assert(cgltf_child_node != nullptr);  // assume valid cgltf node pointer
             return Get(cgltf_child_node, mutable_nodes);
           })
         | std::ranges::to<std::vector>();
}

CgltfResourceMap<cgltf_node, const Node> CreateNodes(const std::span<const cgltf_node> cgltf_nodes,
                                                     const CgltfResourceMap<cgltf_mesh, const Mesh>& meshes,
                                                     const CgltfResourceMap<cgltf_light, const Light>& lights) {
  // create nodes without establishing parent-child relationships in the node hierarchy
  auto nodes = cgltf_nodes  //
               | std::views::transform([&meshes, &lights](const auto& cgltf_node) {
                   return std::pair{&cgltf_node,
                                    std::make_unique<Node>(GetName(cgltf_node),
                                                           GetLocalTransform(cgltf_node),
                                                           Get(cgltf_node.mesh, meshes),
                                                           Get(cgltf_node.light, lights))};
                 })
               | std::ranges::to<std::unordered_map>();

  // assign child pointers after all nodes have been created
  for (auto& [cgltf_node, node] : nodes) {
    node->children = GetChildren(*cgltf_node, nodes);
  }

  return nodes  // convert to const pointers after all nodes have been completely initialized
         | std::views::transform([](auto& key_value_pair) {
             auto& [cgltf_node, mutable_node] = key_value_pair;
             return std::pair{cgltf_node, std::unique_ptr<const Node>(std::move(mutable_node))};
           })
         | std::ranges::to<std::unordered_map>();
}

// =====================================================================================================================
// Scenes
// =====================================================================================================================

std::vector<const Node*> GetRootNodes(const cgltf_scene& cgltf_scene,
                                      const CgltfResourceMap<cgltf_node, const Node>& nodes) {
  return std::span{cgltf_scene.nodes, cgltf_scene.nodes_count}
         | std::views::transform([&nodes](const auto* const cgltf_root_node) {
             assert(cgltf_root_node != nullptr);  // assume valid cgltf node pointer
             return Get(cgltf_root_node, nodes);
           })
         | std::ranges::to<std::vector>();
}

CgltfResourceMap<cgltf_scene, const Scene> CreateScenes(const std::span<const cgltf_scene>& cgltf_scenes,
                                                        const CgltfResourceMap<cgltf_node, const Node>& nodes) {
  return cgltf_scenes  //
         | std::views::transform([&nodes](const auto& cgltf_scene) {
             return std::pair{&cgltf_scene,
                              std::make_unique<const Scene>(GetName(cgltf_scene), GetRootNodes(cgltf_scene, nodes))};
           })
         | std::ranges::to<std::unordered_map>();
}

}  // namespace

Asset Load(const std::filesystem::path& gltf_filepath, Log& log) {
  const auto cgltf_data = LoadGltfFile(gltf_filepath.string());

  const std::span cgltf_samplers{cgltf_data->samplers, cgltf_data->samplers_count};
  auto samplers = CreateSamplers(cgltf_samplers);

  const std::span cgltf_textures{cgltf_data->textures, cgltf_data->textures_count};
  auto textures = CreateTextures(cgltf_textures, gltf_filepath.parent_path(), samplers);

  const std::span cgltf_materials{cgltf_data->materials, cgltf_data->materials_count};
  auto materials = CreateMaterials(cgltf_materials, textures);

  const std::span cgltf_meshes{cgltf_data->meshes, cgltf_data->meshes_count};
  auto meshes = CreateMeshes(cgltf_meshes, materials, log);

  const std::span cgltf_lights{cgltf_data->lights, cgltf_data->lights_count};
  auto lights = CreateLights(cgltf_lights, log);

  const std::span cgltf_nodes{cgltf_data->nodes, cgltf_data->nodes_count};
  auto nodes = CreateNodes(cgltf_nodes, meshes, lights);

  const std::span cgltf_scenes{cgltf_data->scenes, cgltf_data->scenes_count};
  auto scenes = CreateScenes(cgltf_scenes, nodes);

  const auto* const default_scene = Get(cgltf_data->scene, scenes);

  return Asset{.name = gltf_filepath.filename().string(),
               .samplers = GetValues(std::move(samplers)),
               .textures = GetValues(std::move(textures)),
               .materials = GetValues(std::move(materials)),
               .meshes = GetValues(std::move(meshes)),
               .lights = GetValues(std::move(lights)),
               .nodes = GetValues(std::move(nodes)),
               .scenes = GetValues(std::move(scenes)),
               .default_scene = default_scene};
}

}  // namespace vktf::gltf
