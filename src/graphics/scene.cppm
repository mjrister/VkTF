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
#include <variant>
#include <vector>

#include <cgltf.h>
#include <ktx.h>
#include <vk_mem_alloc.h>
#include <glm/glm.hpp>
#include <glm/gtc/epsilon.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <vulkan/vulkan.hpp>
#include <vulkan/vulkan_hash.hpp>

export module scene;

import buffer;
import camera;
import data_view;
import image;
import ktx_texture;
import shader_module;

namespace {

struct Texture {
  std::variant<gfx::KtxTexture, gfx::Image> image;  // first loaded as a KTX texture then converted to a device image
  vk::Sampler sampler;
};

struct Material {
  std::optional<Texture> maybe_base_color_texture;
  std::optional<Texture> maybe_metallic_roughness_texture;
  glm::vec4 base_color_factor{1.0f};
  float metallic_factor = 1.0f;
  float roughness_factor = 1.0f;
  vk::UniqueDescriptorSet descriptor_set;
};

struct IndexBuffer {
  std::uint32_t index_count = 0;
  vk::IndexType index_type = vk::IndexType::eUint16;
  gfx::Buffer buffer;
};

struct Primitive {
  gfx::Buffer vertex_buffer;
  IndexBuffer index_buffer;
  vk::DescriptorSet descriptor_set;
};

struct Mesh {
  std::vector<Primitive> primitives;
};

struct Node {
  const Mesh* mesh = nullptr;  // pointer lifetime is managed by the scene
  glm::mat4 transform{1.0f};
  std::vector<std::unique_ptr<const Node>> children;
};

}  // namespace

namespace gfx {

export struct SubmitCopyCommandsOptions {
  vk::Device device;
  vk::Queue transfer_queue;
  std::uint32_t transfer_queue_family_index = 0;
  VmaAllocator allocator = nullptr;
};

export struct CreateTextureOptions {
  vk::PhysicalDevice physical_device;
  vk::Bool32 enable_sampler_anisotropy = 0;
  float max_sampler_anisotropy = 0.0f;
};

export struct CreateGraphicsPipelineOptions {
  vk::Extent2D viewport_extent;
  vk::SampleCountFlagBits msaa_sample_count = vk::SampleCountFlagBits::e1;
  vk::RenderPass render_pass;
};

export class Scene {
public:
  Scene(const std::filesystem::path& gltf_filepath,
        const SubmitCopyCommandsOptions& submit_copy_commands_options,
        const CreateTextureOptions& create_texture_options,
        const CreateGraphicsPipelineOptions& create_graphics_pipeline_options);

  void Render(const Camera& camera, vk::CommandBuffer command_buffer) const;

private:
  vk::UniqueDescriptorPool descriptor_pool_;
  vk::UniqueDescriptorSetLayout descriptor_set_layout_;
  vk::UniquePipelineLayout graphics_pipeline_layout_;
  vk::UniquePipeline graphics_pipeline_;
  std::unique_ptr<const Node> root_node_;
  std::vector<std::unique_ptr<const Material>> materials_;
  std::vector<std::unique_ptr<const Mesh>> meshes_;
  std::vector<vk::UniqueSampler> samplers_;
};

}  // namespace gfx

module :private;

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

struct CopyBufferOptions {
  vk::UniqueCommandPool command_pool;
  vk::UniqueCommandBuffer command_buffer;
  std::vector<gfx::Buffer> staging_buffers;  // staging buffers must remain in scope until copy commands complete
  VmaAllocator allocator = nullptr;
};

CopyBufferOptions CreateCopyBufferOptions(const vk::Device device,
                                          const std::uint32_t transfer_queue_family_index,
                                          const std::size_t staging_buffers_size,
                                          const VmaAllocator allocator) {
  auto copy_command_pool =
      device.createCommandPoolUnique(vk::CommandPoolCreateInfo{.flags = vk::CommandPoolCreateFlagBits::eTransient,
                                                               .queueFamilyIndex = transfer_queue_family_index});
  auto copy_command_buffers =
      device.allocateCommandBuffersUnique(vk::CommandBufferAllocateInfo{.commandPool = *copy_command_pool,
                                                                        .level = vk::CommandBufferLevel::ePrimary,
                                                                        .commandBufferCount = 1});

  std::vector<gfx::Buffer> staging_buffers;
  staging_buffers.reserve(staging_buffers_size);

  return CopyBufferOptions{.command_pool = std::move(copy_command_pool),
                           .command_buffer = std::move(copy_command_buffers.front()),
                           .staging_buffers = std::move(staging_buffers),
                           .allocator = allocator};
}

template <typename Key, typename Value>
using UnorderedPtrMap = std::unordered_map<const Key*, std::unique_ptr<const Value>>;

template <typename Key, typename Value>
const Value* Find(const Key* const key, const UnorderedPtrMap<Key, Value>& map) {
  if (key == nullptr) return nullptr;
  const auto iterator = map.find(key);
  assert(iterator != map.cend());  // map should be initialized with all known key types before this function is called
  return iterator->second.get();
}

// =================================================== glTF Loading ====================================================

using CgltfData = std::unique_ptr<cgltf_data, decltype(&cgltf_free)>;

CgltfData Load(const std::string& gltf_filepath) {
  static constexpr cgltf_options kDefaultOptions{};
  CgltfData cgltf_data{nullptr, cgltf_free};

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

// ================================================= Sampler Creation ==================================================

// filter and address mode values come from https://registry.khronos.org/glTF/specs/2.0/glTF-2.0.html#reference-sampler
enum class SamplerFilter {
  kNearest = 9728,
  kLinear = 9729,
  kNearestMipmapNearest = 9984,
  kLinearMipmapNearest = 9985,
  kNearestMipmapLinear = 9986,
  kLinearMipmapLinear = 9987
};

enum class SamplerAddressMode { kClampToEdge = 33071, kMirroredRepeat = 33648, kRepeat = 10497 };

vk::Filter GetSamplerMagFilter(const int cgltf_mag_filter) {
  switch (static_cast<SamplerFilter>(cgltf_mag_filter)) {
    using enum SamplerFilter;
    case kNearest:
      return vk::Filter::eNearest;
    case kLinear:
      return vk::Filter::eLinear;
    default:
      std::unreachable();
  }
}

std::pair<vk::Filter, vk::SamplerMipmapMode> GetSamplerMinFilterAndMipmapMode(const int cgltf_min_filter) {
  switch (static_cast<SamplerFilter>(cgltf_min_filter)) {
    using enum SamplerFilter;
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

vk::SamplerAddressMode GetSamplerAddressMode(const int cgltf_wrap_mode) {
  switch (static_cast<SamplerAddressMode>(cgltf_wrap_mode)) {
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

vk::Sampler CreateSampler(const cgltf_sampler* const cgltf_sampler,
                          const vk::Bool32 enable_anisotropy,
                          const float max_anisotropy,
                          const std::uint32_t mip_levels,
                          const vk::Device device,
                          std::unordered_map<vk::SamplerCreateInfo, vk::UniqueSampler>& samplers) {
  vk::SamplerCreateInfo sampler_create_info{.anisotropyEnable = enable_anisotropy,
                                            .maxAnisotropy = max_anisotropy,
                                            .maxLod = static_cast<float>(mip_levels)};

  if (cgltf_sampler != nullptr) {
    const auto [min_filter, mipmap_mode] = GetSamplerMinFilterAndMipmapMode(cgltf_sampler->min_filter);
    sampler_create_info.magFilter = GetSamplerMagFilter(cgltf_sampler->mag_filter);
    sampler_create_info.minFilter = min_filter;
    sampler_create_info.mipmapMode = mipmap_mode;
    sampler_create_info.addressModeU = GetSamplerAddressMode(cgltf_sampler->wrap_s);
    sampler_create_info.addressModeV = GetSamplerAddressMode(cgltf_sampler->wrap_t);
  }

  auto iterator = samplers.find(sampler_create_info);
  if (iterator == samplers.cend()) {
    iterator = samplers.emplace(sampler_create_info, device.createSamplerUnique(sampler_create_info)).first;
  }

  return *iterator->second;
}

// ================================================= Texture Creation ==================================================

template <typename T>
  requires requires(T cgltf_element) {
    { cgltf_element.name } -> std::same_as<char*&>;
  }
std::string_view GetName(const T& cgltf_element) {
  if (const auto* const name = cgltf_element.name; name != nullptr) {
    if (const auto length = std::strlen(name); length > 0) {
      return std::string_view{name, length};
    }
  }
  return "unknown";
}

std::optional<gfx::KtxTexture> CreateKtxTexture(const cgltf_texture_view& cgltf_texture_view,
                                                const gfx::ColorSpace color_space,
                                                const vk::PhysicalDevice physical_device,
                                                const std::filesystem::path& gltf_directory) {
  const auto* const cgltf_texture = cgltf_texture_view.texture;
  if (cgltf_texture == nullptr) return std::nullopt;

  const auto* const cgltf_image = cgltf_texture->has_basisu == 0 ? cgltf_texture->image : cgltf_texture->basisu_image;
  if (cgltf_image == nullptr) {
    std::println(std::cerr, "No image source for texture {}", GetName(*cgltf_texture));
    return std::nullopt;
  }

  const std::filesystem::path cgltf_image_filepath = cgltf_image->uri == nullptr ? "" : cgltf_image->uri;
  if (cgltf_image_filepath.empty()) {
    std::println(std::cerr, "No URI for image {}", GetName(*cgltf_image));
    return std::nullopt;
  }

  return gfx::KtxTexture{gltf_directory / cgltf_image_filepath, color_space, physical_device};
}

std::optional<Texture> CreateTexture(const cgltf_texture_view& cgltf_texture_view,
                                     const gfx::ColorSpace color_space,
                                     const gfx::CreateTextureOptions& create_texture_options,
                                     const std::filesystem::path& gltf_directory,
                                     const vk::Device device,
                                     std::unordered_map<vk::SamplerCreateInfo, vk::UniqueSampler>& samplers) {
  const auto& [physical_device, enable_anisotropy, max_anisotropy] = create_texture_options;

  return CreateKtxTexture(cgltf_texture_view, color_space, physical_device, gltf_directory)
      .transform([cgltf_texture = cgltf_texture_view.texture, enable_anisotropy, max_anisotropy, device, &samplers](
                     auto&& ktx_texture) {
        const auto mip_levels = ktx_texture->numLevels;
        return Texture{
            .image = std::move(ktx_texture),
            .sampler =
                CreateSampler(cgltf_texture->sampler, enable_anisotropy, max_anisotropy, mip_levels, device, samplers)};
      });
}

// ================================================= Material Creation =================================================

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

std::unique_ptr<Material> CreateMaterial(const cgltf_material& cgltf_material,
                                         const gfx::CreateTextureOptions& create_texture_options,
                                         const std::filesystem::path& gltf_directory,
                                         const vk::Device device,
                                         std::unordered_map<vk::SamplerCreateInfo, vk::UniqueSampler>& samplers) {
  // TODO(matthew-rister): avoid requiring PBR metallic-roughness properties
  if (!cgltf_material.has_pbr_metallic_roughness) {
    throw std::runtime_error{
        std::format("The material {} is unsupported because it does not have PBR metallic-roughness properties",
                    GetName(cgltf_material))};
  }

  const auto& [base_color_texture, metallic_roughness_texture, base_color_factor, metallic_factor, roughness_factor] =
      cgltf_material.pbr_metallic_roughness;

  using enum gfx::ColorSpace;
  return std::make_unique<Material>(
      CreateTexture(base_color_texture, kSrgb, create_texture_options, gltf_directory, device, samplers),
      CreateTexture(metallic_roughness_texture, kLinear, create_texture_options, gltf_directory, device, samplers),
      ToVec(base_color_factor),
      metallic_factor,
      roughness_factor);
}

// ================================================== Image Creation ===================================================

template <typename T>
const gfx::Buffer& CreateStagingBuffer(const gfx::DataView<const T> data_view,
                                       const VmaAllocator allocator,
                                       std::vector<gfx::Buffer>& staging_buffers) {
  static constexpr VmaAllocationCreateInfo kHostVisibleAllocationCreateInfo{
      .flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT,
      .usage = VMA_MEMORY_USAGE_AUTO};
  auto& staging_buffer = staging_buffers.emplace_back(data_view.size_bytes(),
                                                      vk::BufferUsageFlagBits::eTransferSrc,
                                                      allocator,
                                                      kHostVisibleAllocationCreateInfo);
  staging_buffer.CopyOnce(data_view);
  return staging_buffer;
}

std::vector<vk::BufferImageCopy> GetBufferImageCopies(const ktxTexture2& ktx_texture2) {
  return std::views::iota(0u, ktx_texture2.numLevels)
         | std::views::transform([ktx_texture = ktxTexture(&ktx_texture2)](const auto mip_level) {
             ktx_size_t image_offset = 0;
             if (const auto ktx_error_code = ktxTexture_GetImageOffset(ktx_texture, mip_level, 0, 0, &image_offset);
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
                 .imageExtent = vk::Extent3D{.width = ktx_texture->baseWidth >> mip_level,
                                             .height = ktx_texture->baseHeight >> mip_level,
                                             .depth = 1}};
           })
         | std::ranges::to<std::vector>();
}

gfx::Image CreateImage(const ktxTexture2& ktx_texture2,
                       const vk::Device device,
                       CopyBufferOptions& copy_buffer_options) {
  auto& [_, command_buffer, staging_buffers, allocator] = copy_buffer_options;
  const auto& staging_buffer =
      CreateStagingBuffer(gfx::DataView<const ktx_uint8_t>{ktx_texture2.pData, ktx_texture2.dataSize},
                          allocator,
                          staging_buffers);

  gfx::Image image{device,
                   static_cast<vk::Format>(ktx_texture2.vkFormat),
                   vk::Extent2D{ktx_texture2.baseWidth, ktx_texture2.baseHeight},
                   ktx_texture2.numLevels,
                   vk::SampleCountFlagBits::e1,
                   vk::ImageUsageFlagBits::eSampled | vk::ImageUsageFlagBits::eTransferDst,
                   vk::ImageAspectFlagBits::eColor,
                   allocator};

  const auto buffer_image_copies = GetBufferImageCopies(ktx_texture2);
  image.Copy(*staging_buffer, buffer_image_copies, *command_buffer);

  return image;
}

std::unique_ptr<const Material> CreateImages(std::unique_ptr<Material>&& material,
                                             vk::UniqueDescriptorSet&& descriptor_set,
                                             const vk::Device device,
                                             CopyBufferOptions& copy_buffer_options) {
  auto& maybe_base_color_texture = material->maybe_base_color_texture;
  auto& maybe_metallic_roughness_texture = material->maybe_metallic_roughness_texture;
  if (!maybe_base_color_texture.has_value() || !maybe_metallic_roughness_texture.has_value()) {
    return nullptr;  // TODO(matthew-rister): avoid requiring base color and metallic roughness textures
  }

  auto& base_color_texture = *maybe_base_color_texture;
  const auto& base_color_ktx_texture = std::get<gfx::KtxTexture>(base_color_texture.image);
  base_color_texture.image = CreateImage(*base_color_ktx_texture, device, copy_buffer_options);
  const auto& base_color_image = std::get<gfx::Image>(base_color_texture.image);

  auto& metallic_roughness_texture = *maybe_metallic_roughness_texture;
  const auto& metallic_roughness_color_ktx_texture = std::get<gfx::KtxTexture>(metallic_roughness_texture.image);
  metallic_roughness_texture.image = CreateImage(*metallic_roughness_color_ktx_texture, device, copy_buffer_options);
  const auto& metallic_roughness_image = std::get<gfx::Image>(metallic_roughness_texture.image);

  const std::array descriptor_image_info{
      vk::DescriptorImageInfo{.sampler = base_color_texture.sampler,
                              .imageView = base_color_image.image_view(),
                              .imageLayout = vk::ImageLayout::eShaderReadOnlyOptimal},
      vk::DescriptorImageInfo{.sampler = metallic_roughness_texture.sampler,
                              .imageView = metallic_roughness_image.image_view(),
                              .imageLayout = vk::ImageLayout::eShaderReadOnlyOptimal}};

  device.updateDescriptorSets(
      vk::WriteDescriptorSet{.dstSet = *descriptor_set,
                             .dstBinding = 0,
                             .dstArrayElement = 0,
                             .descriptorCount = static_cast<uint32_t>(descriptor_image_info.size()),
                             .descriptorType = vk::DescriptorType::eCombinedImageSampler,
                             .pImageInfo = descriptor_image_info.data()},
      nullptr);

  material->descriptor_set = std::move(descriptor_set);
  return std::unique_ptr<const Material>{std::move(material)};  // convert to a const pointer after updates are complete
}

// ================================================== Vertex Creation ==================================================

template <typename T, glm::length_t N>
  requires VecConstructible<T, N>
struct VertexAttribute {
  std::string_view name;  // guaranteed to reference a string literal with static storage duration
  std::optional<std::vector<glm::vec<N, T>>> maybe_data;
};

struct Vertex {
  glm::vec3 position{0.0f};
  glm::vec3 normal{0.0f};
  glm::vec4 tangent{0.0f};  // w-component indicates the signed handedness of the tangent basis
  glm::vec2 texture_coordinates_0{0.0f};
};

template <glm::length_t N>
  requires VecConstructible<float, N>
std::vector<glm::vec<N, float>> UnpackFloats(const cgltf_accessor& cgltf_accessor) {
  if (const auto components_count = cgltf_num_components(cgltf_accessor.type); components_count != N) {
    throw std::runtime_error{std::format(
        "The number of expected components {} does not match the number of actual components {} for accessor {}",
        N,
        components_count,
        GetName(cgltf_accessor))};
  }
  std::vector<glm::vec<N, float>> floats(cgltf_accessor.count);
  if (const auto float_count = N * cgltf_accessor.count;
      cgltf_accessor_unpack_floats(&cgltf_accessor, glm::value_ptr(floats.front()), float_count) == 0) {
    throw std::runtime_error{std::format("Failed to unpack floats for accessor {}", GetName(cgltf_accessor))};
  }
  return floats;
}

template <glm::length_t N>
void ValidateRequiredAttributeData(const VertexAttribute<float, N>& vertex_attribute) {
  if (!vertex_attribute.maybe_data.has_value()) {
    throw std::runtime_error{std::format("Missing required vertex attribute {}", vertex_attribute.name)};
  }
}

template <glm::length_t N>
void ValidateRequiredAttributeSize(const VertexAttribute<float, 3>& vertex_attribute_0,
                                   const VertexAttribute<float, N>& vertex_attribute_n) {
  const auto& [attribute_0_name, maybe_attribute_0_data] = vertex_attribute_0;
  const auto& [attribute_n_name, maybe_attribute_n_data] = vertex_attribute_n;

  // NOLINTBEGIN(bugprone-unchecked-optional-access): attribute data is validated before this function is called
  const auto attribute_0_size = maybe_attribute_0_data->size();
  const auto attribute_n_size = maybe_attribute_n_data->size();
  // NOLINTEND(bugprone-unchecked-optional-access)

  // the glTF specification requires all attribute accessors for a primitive must have the same count
  if (attribute_0_size != attribute_n_size) {
    throw std::runtime_error{std::format("The number of {} attributes {} does not match the number of {} attributes {}",
                                         attribute_0_name,
                                         attribute_0_size,
                                         attribute_n_name,
                                         attribute_n_size)};
  }
}

// validation requires variadic templates because vertex attribute data types are not homogeneous
template <typename... VertexAttributes>
void ValidateRequiredAttributes(const VertexAttribute<float, 3>& vertex_attribute_0,
                                const VertexAttributes&... vertex_attributes) {
  ValidateRequiredAttributeData(vertex_attribute_0);
  (ValidateRequiredAttributeData(vertex_attributes), ...);
  (ValidateRequiredAttributeSize(vertex_attribute_0, vertex_attributes), ...);
}

std::vector<Vertex> CreateVertices(const cgltf_primitive& cgltf_primitive) {
  VertexAttribute<float, 3> position_attribute{.name = "POSITION"};
  VertexAttribute<float, 3> normal_attribute{.name = "NORMAL"};
  VertexAttribute<float, 4> tangent_attribute{.name = "TANGENT"};
  VertexAttribute<float, 2> texture_coordinates_0_attribute{.name = "TEXCOORD_0"};

  static constexpr auto kUnpackAttributeData = []<glm::length_t N>(const auto& cgltf_attribute,
                                                                   VertexAttribute<float, N>& vertex_attribute) {
    assert(!vertex_attribute.maybe_data.has_value());  // glTF primitives should not have duplicate attributes
    vertex_attribute.maybe_data = UnpackFloats<N>(*cgltf_attribute.data);
  };

  for (const auto& cgltf_attribute : std::span{cgltf_primitive.attributes, cgltf_primitive.attributes_count}) {
    switch (cgltf_attribute.type) {
      case cgltf_attribute_type_position:
        kUnpackAttributeData(cgltf_attribute, position_attribute);
        break;
      case cgltf_attribute_type_normal:
        kUnpackAttributeData(cgltf_attribute, normal_attribute);
        break;
      case cgltf_attribute_type_tangent:
        kUnpackAttributeData(cgltf_attribute, tangent_attribute);
        break;
      case cgltf_attribute_type_texcoord:
        if (texture_coordinates_0_attribute.name == GetName(cgltf_attribute)) {
          kUnpackAttributeData(cgltf_attribute, texture_coordinates_0_attribute);
          break;
        }
        [[fallthrough]];
      default:
        std::println(std::cerr, "Unsupported primitive attribute {}", GetName(cgltf_attribute));
        break;
    }
  }

  // TODO(matthew-rister): add support for optional vertex attributes
  ValidateRequiredAttributes(position_attribute, normal_attribute, tangent_attribute, texture_coordinates_0_attribute);

  return std::views::zip_transform(
             [](const auto& position, const auto& normal, const auto& tangent, const auto& texture_coordinates_0) {
#ifndef NDEBUG
               // the glTF specification requires unit length normals and tangent vectors
               static constexpr auto kEpsilon = 1.0e-6f;
               assert(glm::epsilonEqual(glm::length(normal), 1.0f, kEpsilon));
               assert(glm::epsilonEqual(glm::length(glm::vec3{tangent}), 1.0f, kEpsilon));
#endif
               return Vertex{.position = position,
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

// =================================================== Mesh Creation ===================================================

template <typename T>
gfx::Buffer CreateBuffer(const std::vector<T>& data,
                         const vk::BufferUsageFlags usage_flags,
                         CopyBufferOptions& copy_buffer_options) {
  const gfx::DataView<const T> data_view{data};
  auto& [_, command_buffer, staging_buffers, allocator] = copy_buffer_options;
  const auto& staging_buffer = CreateStagingBuffer(data_view, allocator, staging_buffers);

  gfx::Buffer buffer{data_view.size_bytes(), usage_flags | vk::BufferUsageFlagBits::eTransferDst, allocator};
  command_buffer->copyBuffer(*staging_buffer, *buffer, vk::BufferCopy{.size = data_view.size_bytes()});

  return buffer;
}

template <typename T>
  requires std::same_as<T, std::uint16_t> || std::same_as<T, std::uint32_t>
std::vector<T> UnpackIndices(const cgltf_accessor& cgltf_accessor) {
  std::vector<T> indices(cgltf_accessor.count);
  if (const auto index_size_bytes = sizeof(T);
      cgltf_accessor_unpack_indices(&cgltf_accessor, indices.data(), index_size_bytes, indices.size()) == 0) {
    throw std::runtime_error{std::format("Failed to unpack indices for accessor {}", GetName(cgltf_accessor))};
  }
  return indices;
}

IndexBuffer CreateIndexBuffer(const cgltf_accessor& cgltf_accessor, CopyBufferOptions& copy_buffer_options) {
  switch (const auto component_size = cgltf_component_size(cgltf_accessor.component_type)) {
    case 2: {
      const auto indices = UnpackIndices<std::uint16_t>(cgltf_accessor);
      return IndexBuffer{.index_count = static_cast<std::uint32_t>(indices.size()),
                         .index_type = vk::IndexType::eUint16,
                         .buffer = CreateBuffer(indices, vk::BufferUsageFlagBits::eIndexBuffer, copy_buffer_options)};
    }
    case 4: {
      const auto indices = UnpackIndices<std::uint32_t>(cgltf_accessor);
      return IndexBuffer{.index_count = static_cast<std::uint32_t>(indices.size()),
                         .index_type = vk::IndexType::eUint32,
                         .buffer = CreateBuffer(indices, vk::BufferUsageFlagBits::eIndexBuffer, copy_buffer_options)};
    }
    default: {
      static constexpr auto kBitsPerByte = 8;
      throw std::runtime_error{std::format("Unsupported {}-bit index type", component_size * kBitsPerByte)};
    }
  }
}

std::unique_ptr<const Mesh> CreateMesh(const cgltf_mesh& cgltf_mesh,
                                       const UnorderedPtrMap<cgltf_material, Material>& materials,
                                       CopyBufferOptions& copy_buffer_options) {
  std::vector<Primitive> primitives;
  primitives.reserve(cgltf_mesh.primitives_count);

  for (auto index = 0; const auto& cgltf_primitive : std::span{cgltf_mesh.primitives, cgltf_mesh.primitives_count}) {
    if (cgltf_primitive.type != cgltf_primitive_type_triangles) {
      static constexpr auto kMessageFormat = "Mesh {} primitive {} with type {} is unsupported";
      std::println(std::cerr, kMessageFormat, GetName(cgltf_mesh), index++, cgltf_primitive.type);
      continue;  // TODO(matthew-rister): add support for other primitive types
    }
    if (cgltf_primitive.indices == nullptr || cgltf_primitive.indices->count == 0) {
      static constexpr auto kMessageFormat = "Mesh {} primitive {} without an indices accessor is unsupported";
      std::println(std::cerr, kMessageFormat, GetName(cgltf_mesh), index++);
      continue;  // TODO(matthew-rister): add support for non-indexed triangle meshes
    }
    const auto material = Find(cgltf_primitive.material, materials);
    if (material == nullptr) {
      static constexpr auto kMessageFormat = "Mesh {} primitive {} with material {} is unsupported";
      std::println(std::cerr, kMessageFormat, GetName(cgltf_mesh), index++, GetName(*cgltf_primitive.material));
      continue;  // TODO(matthew-rister): add default material support
    }
    const auto vertices = CreateVertices(cgltf_primitive);
    primitives.emplace_back(CreateBuffer(vertices, vk::BufferUsageFlagBits::eVertexBuffer, copy_buffer_options),
                            CreateIndexBuffer(*cgltf_primitive.indices, copy_buffer_options),
                            *material->descriptor_set);
    ++index;
  }

  return std::make_unique<const Mesh>(std::move(primitives));
}

// =================================================== Node Creation ===================================================

glm::mat4 GetTransform(const cgltf_node& cgltf_node) {
  glm::mat4 transform{1.0f};
  cgltf_node_transform_local(&cgltf_node, glm::value_ptr(transform));
  return transform;
}

std::vector<std::unique_ptr<const Node>> CreateNodes(const cgltf_node* const* const cgltf_nodes,
                                                     const cgltf_size cgltf_nodes_count,
                                                     const UnorderedPtrMap<cgltf_mesh, Mesh>& meshes) {
  return std::span{cgltf_nodes, cgltf_nodes_count}  //
         | std::views::transform([&meshes](const auto* const cgltf_node) {
             return std::make_unique<const Node>(Find(cgltf_node->mesh, meshes),
                                                 GetTransform(*cgltf_node),
                                                 CreateNodes(cgltf_node->children, cgltf_node->children_count, meshes));
           })
         | std::ranges::to<std::vector>();
}

const cgltf_scene& GetDefaultScene(const cgltf_data& cgltf_data) {
  if (const auto* const cgltf_scene = cgltf_data.scene; cgltf_scene != nullptr) {
    return *cgltf_scene;
  }
  if (const std::span cgltf_scenes{cgltf_data.scenes, cgltf_data.scenes_count}; !cgltf_scenes.empty()) {
    return cgltf_scenes.front();
  }
  // TODO(matthew-rister): glTF files not containing scene data should be treated as a library of individual entities
  throw std::runtime_error{"At least one glTF scene is required to render"};
}

std::unique_ptr<const Node> CreateRootNode(const cgltf_data& cgltf_data,
                                           const UnorderedPtrMap<cgltf_mesh, Mesh>& meshes) {
  const auto& cgltf_scene = GetDefaultScene(cgltf_data);
  return std::make_unique<const Node>(nullptr,
                                      glm::mat4{1.0f},
                                      CreateNodes(cgltf_scene.nodes, cgltf_scene.nodes_count, meshes));
}

// ============================================= Descriptor Sets Creation ==============================================

vk::UniqueDescriptorPool CreateDescriptorPool(const vk::Device device, const std::uint32_t max_descriptor_sets) {
  static constexpr vk::DescriptorPoolSize kDescriptorPoolSize{.type = vk::DescriptorType::eCombinedImageSampler,
                                                              .descriptorCount = 2};

  return device.createDescriptorPoolUnique(
      vk::DescriptorPoolCreateInfo{.flags = vk::DescriptorPoolCreateFlagBits::eFreeDescriptorSet,
                                   .maxSets = max_descriptor_sets,
                                   .poolSizeCount = 1,
                                   .pPoolSizes = &kDescriptorPoolSize});
}

vk::UniqueDescriptorSetLayout CreateDescriptorSetLayout(const vk::Device device) {
  static constexpr vk::DescriptorSetLayoutBinding kDescriptorSetLayoutBinding{
      .descriptorType = vk::DescriptorType::eCombinedImageSampler,
      .descriptorCount = 2,
      .stageFlags = vk::ShaderStageFlagBits::eFragment};

  return device.createDescriptorSetLayoutUnique(
      vk::DescriptorSetLayoutCreateInfo{.bindingCount = 1, .pBindings = &kDescriptorSetLayoutBinding});
}

std::vector<vk::UniqueDescriptorSet> AllocateDescriptorSets(const vk::Device device,
                                                            const vk::DescriptorPool descriptor_pool,
                                                            const vk::DescriptorSetLayout& descriptor_set_layout,
                                                            const std::uint32_t descriptor_set_count) {
  const std::vector descriptor_set_layouts(descriptor_set_count, descriptor_set_layout);

  return device.allocateDescriptorSetsUnique(
      vk::DescriptorSetAllocateInfo{.descriptorPool = descriptor_pool,
                                    .descriptorSetCount = descriptor_set_count,
                                    .pSetLayouts = descriptor_set_layouts.data()});
}

// ============================================ Graphics Pipeline Creation =============================================

struct PushConstants {
  glm::mat4 model_view_transform{1.0f};
  glm::mat4 projection_transform{1.0f};
};

vk::UniquePipelineLayout CreateGraphicsPipelineLayout(const vk::Device device,
                                                      const vk::DescriptorSetLayout& descriptor_set_layout) {
  static constexpr vk::PushConstantRange kPushConstantRange{.stageFlags = vk::ShaderStageFlagBits::eVertex,
                                                            .offset = 0,
                                                            .size = sizeof(PushConstants)};
  return device.createPipelineLayoutUnique(vk::PipelineLayoutCreateInfo{.setLayoutCount = 1,
                                                                        .pSetLayouts = &descriptor_set_layout,
                                                                        .pushConstantRangeCount = 1,
                                                                        .pPushConstantRanges = &kPushConstantRange});
}

vk::UniquePipeline CreateGraphicsPipeline(const vk::Device device,
                                          const vk::PipelineLayout graphics_pipeline_layout,
                                          const gfx::CreateGraphicsPipelineOptions& create_graphics_pipeline_options) {
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
                                          .offset = offsetof(Vertex, texture_coordinates_0)}};

  static constexpr vk::PipelineVertexInputStateCreateInfo kVertexInputStateCreateInfo{
      .vertexBindingDescriptionCount = 1,
      .pVertexBindingDescriptions = &kVertexInputBindingDescription,
      .vertexAttributeDescriptionCount = static_cast<std::uint32_t>(kVertexAttributeDescriptions.size()),
      .pVertexAttributeDescriptions = kVertexAttributeDescriptions.data()};

  static constexpr vk::PipelineInputAssemblyStateCreateInfo kInputAssemblyStateCreateInfo{
      .topology = vk::PrimitiveTopology::eTriangleList};

  // TODO(matthew-rister): use dynamic viewport and scissor pipeline state when window resizing is implemented
  const auto& [viewport_extent, msaa_sample_count, render_pass] = create_graphics_pipeline_options;
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

// ==================================================== Rendering ======================================================

void Render(const Mesh& mesh,
            const glm::mat4& model_view_transform,
            const vk::PipelineLayout graphics_pipeline_layout,
            const vk::CommandBuffer command_buffer) {
  using ModelViewTransform = decltype(PushConstants::model_view_transform);
  command_buffer.pushConstants<ModelViewTransform>(graphics_pipeline_layout,
                                                   vk::ShaderStageFlagBits::eVertex,
                                                   offsetof(PushConstants, model_view_transform),
                                                   model_view_transform);

  for (const auto& [vertex_buffer, index_buffer, descriptor_set] : mesh.primitives) {
    command_buffer.bindDescriptorSets(vk::PipelineBindPoint::eGraphics,
                                      graphics_pipeline_layout,
                                      0,
                                      descriptor_set,
                                      nullptr);
    command_buffer.bindVertexBuffers(0, *vertex_buffer, static_cast<vk::DeviceSize>(0));
    command_buffer.bindIndexBuffer(*index_buffer.buffer, 0, index_buffer.index_type);
    command_buffer.drawIndexed(index_buffer.index_count, 1, 0, 0, 0);
  }
}

void Render(const Node& node,
            const glm::mat4& parent_transform,
            const glm::mat4& view_transform,
            const vk::PipelineLayout graphics_pipeline_layout,
            const vk::CommandBuffer command_buffer) {
  const auto model_transform = parent_transform * node.transform;

  if (node.mesh != nullptr) {
    Render(*node.mesh, view_transform * model_transform, graphics_pipeline_layout, command_buffer);
  }

  for (const auto& child_node : node.children) {
    Render(*child_node, model_transform, view_transform, graphics_pipeline_layout, command_buffer);
  }
}

}  // namespace

namespace gfx {

Scene::Scene(const std::filesystem::path& gltf_filepath,
             const SubmitCopyCommandsOptions& submit_copy_commands_options,
             const CreateTextureOptions& create_texture_options,
             const CreateGraphicsPipelineOptions& create_graphics_pipeline_options) {
  const auto gltf_directory = gltf_filepath.parent_path();
  const auto cgltf_data = Load(gltf_filepath.string());
  const auto& [device, transfer_queue, transfer_queue_family_index, allocator] = submit_copy_commands_options;

  std::unordered_map<vk::SamplerCreateInfo, vk::UniqueSampler> samplers;
  auto material_futures =
      std::span{cgltf_data->materials, cgltf_data->materials_count}  //
      | std::views::transform([&gltf_directory, &create_texture_options, device, &samplers](
                                  const auto& cgltf_material) {
          return std::async([&cgltf_material, &gltf_directory, &create_texture_options, device, &samplers] {
            return std::pair{&cgltf_material,
                             CreateMaterial(cgltf_material, create_texture_options, gltf_directory, device, samplers)};
          });
        })
      | std::ranges::to<std::vector>();

  const auto material_count = static_cast<std::uint32_t>(material_futures.size());
  descriptor_pool_ = CreateDescriptorPool(device, material_count);
  descriptor_set_layout_ = CreateDescriptorSetLayout(device);
  graphics_pipeline_layout_ = CreateGraphicsPipelineLayout(device, *descriptor_set_layout_);
  graphics_pipeline_ = CreateGraphicsPipeline(device, *graphics_pipeline_layout_, create_graphics_pipeline_options);

  const auto staging_buffers_size = cgltf_data->buffers_count + cgltf_data->images_count;
  auto copy_buffer_options =
      CreateCopyBufferOptions(device, transfer_queue_family_index, staging_buffers_size, allocator);

  const auto copy_command_buffer = *copy_buffer_options.command_buffer;
  copy_command_buffer.begin(vk::CommandBufferBeginInfo{.flags = vk::CommandBufferUsageFlagBits::eOneTimeSubmit});

  auto descriptor_sets = AllocateDescriptorSets(device, *descriptor_pool_, *descriptor_set_layout_, material_count);
  auto materials =
      std::views::zip_transform(
          [device, &copy_buffer_options](auto& material_future, auto&& descriptor_set) {
            auto [cgltf_material, material] = material_future.get();
            return std::pair{cgltf_material,
                             CreateImages(std::move(material), std::move(descriptor_set), device, copy_buffer_options)};
          },
          material_futures,
          descriptor_sets | std::views::as_rvalue)
      | std::ranges::to<std::unordered_map>();

  auto meshes = std::span{cgltf_data->meshes, cgltf_data->meshes_count}
                | std::views::transform([&materials, &copy_buffer_options](const auto& cgltf_mesh) {
                    return std::pair{&cgltf_mesh, CreateMesh(cgltf_mesh, materials, copy_buffer_options)};
                  })
                | std::ranges::to<std::unordered_map>();

  copy_command_buffer.end();

  const auto copy_fence = device.createFenceUnique(vk::FenceCreateInfo{});
  transfer_queue.submit(vk::SubmitInfo{.commandBufferCount = 1, .pCommandBuffers = &copy_command_buffer}, *copy_fence);

  root_node_ = CreateRootNode(*cgltf_data, meshes);
  materials_ = materials | std::views::values | std::views::as_rvalue | std::ranges::to<std::vector>();
  meshes_ = meshes | std::views::values | std::views::as_rvalue | std::ranges::to<std::vector>();
  samplers_ = samplers | std::views::values | std::views::as_rvalue | std::ranges::to<std::vector>();

  static constexpr auto kMaxTimeout = std::numeric_limits<std::uint64_t>::max();
  const auto result = device.waitForFences(*copy_fence, vk::True, kMaxTimeout);
  vk::detail::resultCheck(result, "Copy fence failed to enter a signaled state");
}

void Scene::Render(const Camera& camera, const vk::CommandBuffer command_buffer) const {
  command_buffer.bindPipeline(vk::PipelineBindPoint::eGraphics, *graphics_pipeline_);

  using ProjectionTransform = decltype(PushConstants::projection_transform);
  command_buffer.pushConstants<ProjectionTransform>(*graphics_pipeline_layout_,
                                                    vk::ShaderStageFlagBits::eVertex,
                                                    offsetof(PushConstants, projection_transform),
                                                    camera.GetProjectionTransform());

  for (const auto view_transform = camera.GetViewTransform(); const auto& child_node : root_node_->children) {
    ::Render(*child_node, root_node_->transform, view_transform, *graphics_pipeline_layout_, command_buffer);
  }
}

}  // namespace gfx
