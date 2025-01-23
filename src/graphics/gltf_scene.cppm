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
#include <initializer_list>
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
#include <ktx.h>
#include <vk_mem_alloc.h>
#include <glm/glm.hpp>
#include <glm/gtc/epsilon.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <vulkan/vulkan.hpp>
#include <vulkan/vulkan_hash.hpp>

export module gltf_scene;

import allocator;
import buffer;
import camera;
import data_view;
import descriptor_sets;
import image;
import ktx_texture;
import shader_module;

namespace {

struct Texture {
  gfx::Image image;
  vk::Sampler sampler;
};

struct Material {
  std::optional<Texture> maybe_base_color_texture;
  std::optional<Texture> maybe_metallic_roughness_texture;
  std::optional<Texture> maybe_normal_texture;
  std::optional<gfx::Buffer> maybe_properties_buffer;
  vk::DescriptorSet descriptor_set;
};

struct IndexBuffer {
  std::uint32_t index_count = 0;
  vk::IndexType index_type = vk::IndexType::eUint16;
  gfx::Buffer buffer;
};

struct Primitive {
  gfx::Buffer vertex_buffer;
  IndexBuffer index_buffer;
  Material* material;
};

struct Mesh {
  std::vector<Primitive> primitives;
};

struct Light {
  glm::vec4 position{0.0f};
  glm::vec4 color{0.0f};  // padding applied to conform to std140 layout requirements
};

struct Node {
  const Mesh* mesh = nullptr;
  const Light* light = nullptr;
  glm::mat4 transform{0.0f};
  std::vector<std::unique_ptr<const Node>> children;
};

}  // namespace

namespace gfx {

export class GltfScene {
public:
  GltfScene(const std::filesystem::path& gltf_filepath,
            vk::PhysicalDevice physical_device,
            vk::Bool32 enable_sampler_anisotropy,
            float max_sampler_anisotropy,
            vk::Device device,
            vk::Queue queue,
            std::uint32_t queue_family_index,
            vk::Extent2D viewport_extent,
            vk::SampleCountFlagBits msaa_sample_count,
            vk::RenderPass render_pass,
            VmaAllocator allocator,
            std::size_t max_render_frames);

  void Render(const Camera& camera, std::size_t frame_index, vk::CommandBuffer command_buffer) const;

private:
  std::vector<std::unique_ptr<Material>> materials_;
  std::vector<vk::UniqueSampler> samplers_;
  std::vector<std::unique_ptr<const Mesh>> meshes_;
  std::vector<std::unique_ptr<Light>> lights_;
  std::unique_ptr<const Node> root_node_;
  std::vector<Buffer> camera_buffers_;
  std::vector<Buffer> light_buffers_;
  DescriptorSets global_descriptor_sets_;
  DescriptorSets material_descriptor_sets_;
  vk::UniquePipelineLayout graphics_pipeline_layout_;
  vk::UniquePipeline graphics_pipeline_;
};

}  // namespace gfx

module :private;

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

namespace {

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

template <typename Key, typename Value>
using UnorderedPtrMap = std::unordered_map<const Key*, std::unique_ptr<Value>>;

template <typename Key, typename Value>
Value* Find(const Key* const key, const UnorderedPtrMap<Key, Value>& map) {
  if (key == nullptr) return nullptr;
  const auto iterator = map.find(key);
  assert(iterator != map.cend());  // map should be initialized with all known key values before this function is called
  return iterator->second.get();
}

// ======================================================= glTF ========================================================

using UniqueGltfData = std::unique_ptr<cgltf_data, decltype(&cgltf_free)>;

template <typename T>
  requires requires(T gltf_element) {
    { gltf_element.name } -> std::same_as<char*&>;
  }
std::string_view GetName(const T& gltf_element) {
  if (const auto* const name = gltf_element.name; name != nullptr) {
    if (const auto length = std::strlen(name); length > 0) {
      return std::string_view{name, length};
    }
  }
  return "unknown";
}

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

const cgltf_scene& GetDefaultScene(const cgltf_data& gltf_data) {
  if (const auto* const gltf_scene = gltf_data.scene; gltf_scene != nullptr) {
    return *gltf_scene;
  }
  if (const std::span gltf_scenes{gltf_data.scenes, gltf_data.scenes_count}; !gltf_scenes.empty()) {
    return gltf_scenes.front();
  }
  // TODO: glTF files not containing scene data should be treated as a library of individual entities
  throw std::runtime_error{"At least one glTF scene is required to render"};
}

// ====================================================== Buffers ======================================================

struct CopyBufferOptions {
  vk::UniqueCommandPool command_pool;
  vk::UniqueCommandBuffer command_buffer;
  std::vector<gfx::Buffer> staging_buffers;  // staging buffers must remain in scope until copy commands complete
  VmaAllocator allocator = nullptr;
};

constexpr VmaAllocationCreateInfo kHostVisibleAllocationCreateInfo{
    .flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT,
    .usage = VMA_MEMORY_USAGE_AUTO};

CopyBufferOptions CreateCopyBufferOptions(const vk::Device device,
                                          const std::uint32_t transfer_queue_family_index,
                                          const std::size_t staging_buffer_count,
                                          const VmaAllocator allocator) {
  auto copy_command_pool =
      device.createCommandPoolUnique(vk::CommandPoolCreateInfo{.flags = vk::CommandPoolCreateFlagBits::eTransient,
                                                               .queueFamilyIndex = transfer_queue_family_index});
  auto copy_command_buffers =
      device.allocateCommandBuffersUnique(vk::CommandBufferAllocateInfo{.commandPool = *copy_command_pool,
                                                                        .level = vk::CommandBufferLevel::ePrimary,
                                                                        .commandBufferCount = 1});

  std::vector<gfx::Buffer> staging_buffers;
  staging_buffers.reserve(staging_buffer_count);

  return CopyBufferOptions{.command_pool = std::move(copy_command_pool),
                           .command_buffer = std::move(copy_command_buffers.front()),
                           .staging_buffers = std::move(staging_buffers),
                           .allocator = allocator};
}

template <typename T>
vk::Buffer CreateStagingBuffer(const gfx::DataView<const T> data_view,
                               const VmaAllocator allocator,
                               std::vector<gfx::Buffer>& staging_buffers) {
  auto& staging_buffer = staging_buffers.emplace_back(data_view.size_bytes(),
                                                      vk::BufferUsageFlagBits::eTransferSrc,
                                                      allocator,
                                                      kHostVisibleAllocationCreateInfo);

  staging_buffer.MapMemory();
  staging_buffer.Copy(data_view);
  staging_buffer.UnmapMemory();  // staging buffers are only copied once so they can be unmapped immediately

  return *staging_buffer;
}

template <typename T>
gfx::Buffer CreateBuffer(const gfx::DataView<const T>& data_view,
                         const vk::BufferUsageFlags usage_flags,
                         CopyBufferOptions& copy_buffer_options) {
  auto& [_, command_buffer, staging_buffers, allocator] = copy_buffer_options;
  const auto staging_buffer = CreateStagingBuffer(data_view, allocator, staging_buffers);

  gfx::Buffer buffer{data_view.size_bytes(), usage_flags | vk::BufferUsageFlagBits::eTransferDst, allocator};
  command_buffer->copyBuffer(staging_buffer, *buffer, vk::BufferCopy{.size = data_view.size_bytes()});

  return buffer;
}

std::vector<gfx::Buffer> CreateMappedUniformBuffers(const std::size_t buffer_count,
                                                    const std::size_t buffer_size_bytes,
                                                    const VmaAllocator allocator) {
  return std::views::iota(0u, buffer_count)  //
         | std::views::transform([buffer_size_bytes, allocator](const auto /*frame_index*/) {
             gfx::Buffer buffer{buffer_size_bytes,
                                vk::BufferUsageFlagBits::eUniformBuffer,
                                allocator,
                                kHostVisibleAllocationCreateInfo};
             buffer.MapMemory();  // enable persistent mapping
             return buffer;
           })
         | std::ranges::to<std::vector>();
}

// ============================================= Global Descriptor Sets ================================================

gfx::DescriptorSets CreateGlobalDescriptorSets(const vk::Device device, const std::uint32_t max_render_frames) {
  const std::array descriptor_pool_sizes{
      vk::DescriptorPoolSize{.type = vk::DescriptorType::eUniformBuffer, .descriptorCount = 2 * max_render_frames}};

  static constexpr std::array kDescriptorSetLayoutBindings{
      vk::DescriptorSetLayoutBinding{.binding = 0,  // camera buffer
                                     .descriptorType = vk::DescriptorType::eUniformBuffer,
                                     .descriptorCount = 1,
                                     .stageFlags = vk::ShaderStageFlagBits::eVertex},
      vk::DescriptorSetLayoutBinding{.binding = 1,  // light buffer
                                     .descriptorType = vk::DescriptorType::eUniformBuffer,
                                     .descriptorCount = 1,
                                     .stageFlags = vk::ShaderStageFlagBits::eFragment}};

  return gfx::DescriptorSets{device, max_render_frames, descriptor_pool_sizes, kDescriptorSetLayoutBindings};
}

void UpdateGlobalDescriptorSets(const vk::Device device,
                                const gfx::DescriptorSets& global_descriptor_sets,
                                const std::vector<gfx::Buffer>& camera_buffers,
                                const std::vector<gfx::Buffer>& light_buffers) {
  assert(camera_buffers.size() == light_buffers.size());

  std::vector<vk::DescriptorBufferInfo> descriptor_buffer_infos;
  descriptor_buffer_infos.reserve(camera_buffers.size() + light_buffers.size());

  std::vector<vk::WriteDescriptorSet> descriptor_set_writes;
  descriptor_set_writes.reserve(camera_buffers.size() + light_buffers.size());

  for (const auto& [descriptor_set, camera_buffer, light_buffer] :
       std::views::zip(global_descriptor_sets, camera_buffers, light_buffers)) {
    const auto& camera_descriptor_buffer_info = descriptor_buffer_infos.emplace_back(
        vk::DescriptorBufferInfo{.buffer = *camera_buffer, .range = vk::WholeSize});

    descriptor_set_writes.push_back(vk::WriteDescriptorSet{.dstSet = descriptor_set,
                                                           .dstBinding = 0,
                                                           .dstArrayElement = 0,
                                                           .descriptorCount = 1,
                                                           .descriptorType = vk::DescriptorType::eUniformBuffer,
                                                           .pBufferInfo = &camera_descriptor_buffer_info});

    const auto& light_descriptor_buffer_info =
        descriptor_buffer_infos.emplace_back(vk::DescriptorBufferInfo{.buffer = *light_buffer, .range = vk::WholeSize});

    descriptor_set_writes.push_back(vk::WriteDescriptorSet{.dstSet = descriptor_set,
                                                           .dstBinding = 1,
                                                           .dstArrayElement = 0,
                                                           .descriptorCount = 1,
                                                           .descriptorType = vk::DescriptorType::eUniformBuffer,
                                                           .pBufferInfo = &light_descriptor_buffer_info});
  }

  device.updateDescriptorSets(descriptor_set_writes, nullptr);
}

// ===================================================== Samplers ======================================================

struct CreateSamplerOptions {
  vk::Bool32 enable_anisotropy = vk::False;
  float max_anisotropy = 0.0f;
  std::unordered_map<vk::SamplerCreateInfo, vk::UniqueSampler>* samplers = nullptr;
};

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

vk::Sampler CreateSampler(const vk::Device device,
                          const cgltf_sampler* const gltf_sampler,
                          const std::uint32_t mip_levels,
                          CreateSamplerOptions& create_sampler_options) {
  auto& [enable_anisotropy, max_anisotropy, samplers] = create_sampler_options;
  vk::SamplerCreateInfo sampler_create_info{.anisotropyEnable = enable_anisotropy,
                                            .maxAnisotropy = max_anisotropy,
                                            .maxLod = static_cast<float>(mip_levels)};

  if (gltf_sampler != nullptr) {
    const auto [min_filter, mipmap_mode] = GetSamplerMinFilterAndMipmapMode(gltf_sampler->min_filter);
    sampler_create_info.magFilter = GetSamplerMagFilter(gltf_sampler->mag_filter);
    sampler_create_info.minFilter = min_filter;
    sampler_create_info.mipmapMode = mipmap_mode;
    sampler_create_info.addressModeU = GetSamplerAddressMode(gltf_sampler->wrap_s);
    sampler_create_info.addressModeV = GetSamplerAddressMode(gltf_sampler->wrap_t);
  }

  auto iterator = samplers->find(sampler_create_info);
  if (iterator == samplers->cend()) {
    iterator = samplers->emplace(sampler_create_info, device.createSamplerUnique(sampler_create_info)).first;
  }

  return *iterator->second;
}

// ====================================================== Images =======================================================

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

gfx::Image CreateImage(const vk::Device device,
                       const ktxTexture2& ktx_texture2,
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
  image.Copy(staging_buffer, buffer_image_copies, *command_buffer);

  return image;
}

// ===================================================== Materials =====================================================

struct MaterialKtxTextures {
  std::optional<gfx::KtxTexture> maybe_base_color_texture;
  std::optional<gfx::KtxTexture> maybe_metallic_roughness_texture;
  std::optional<gfx::KtxTexture> maybe_normal_texture;
};

struct MaterialProperties {
  glm::vec4 base_color_factor{0.0f};
  glm::vec2 metallic_roughness_factor{0.0f};
  float normal_scale = 0.0f;
};

std::optional<gfx::KtxTexture> CreateKtxTexture(const cgltf_texture_view& gltf_texture_view,
                                                const gfx::ColorSpace color_space,
                                                const std::filesystem::path& gltf_directory,
                                                const vk::PhysicalDevice physical_device) {
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

  return gfx::KtxTexture{gltf_directory / gltf_image_uri, color_space, physical_device};
}

MaterialKtxTextures CreateKtxTextures(const cgltf_material& gltf_material,
                                      const std::filesystem::path& gltf_directory,
                                      const vk::PhysicalDevice physical_device) {
  if (gltf_material.has_pbr_metallic_roughness == 0) {
    return {};  // TODO: add support for non PBR metallic-roughness materials
  }

  const auto& pbr_metallic_roughness = gltf_material.pbr_metallic_roughness;
  auto base_color_texture_future = std::async(std::launch::async,
                                              CreateKtxTexture,
                                              pbr_metallic_roughness.base_color_texture,
                                              gfx::ColorSpace::kSrgb,
                                              gltf_directory,
                                              physical_device);

  auto metallic_roughness_texture_future = std::async(std::launch::async,
                                                      CreateKtxTexture,
                                                      pbr_metallic_roughness.metallic_roughness_texture,
                                                      gfx::ColorSpace::kLinear,
                                                      gltf_directory,
                                                      physical_device);

  auto normal_texture_future = std::async(std::launch::async,
                                          CreateKtxTexture,
                                          gltf_material.normal_texture,
                                          gfx::ColorSpace::kLinear,
                                          gltf_directory,
                                          physical_device);

  return MaterialKtxTextures{.maybe_base_color_texture = base_color_texture_future.get(),
                             .maybe_metallic_roughness_texture = metallic_roughness_texture_future.get(),
                             .maybe_normal_texture = normal_texture_future.get()};
}

std::unique_ptr<Material> CreateMaterial(const vk::Device device,
                                         const cgltf_material& gltf_material,
                                         const MaterialKtxTextures& material_ktx_textures,
                                         CopyBufferOptions& copy_buffer_options,
                                         CreateSamplerOptions& create_sampler_options) {
  const auto& [maybe_base_color_ktx_texture, maybe_metallic_roughness_ktx_texture, maybe_normal_ktx_texture] =
      material_ktx_textures;

  if (std::ranges::any_of(
          std::array{&maybe_base_color_ktx_texture, &maybe_metallic_roughness_ktx_texture, &maybe_normal_ktx_texture},
          [](const auto* const maybe_ktx_texture) { return !maybe_ktx_texture->has_value(); })) {
    std::println(std::cerr,
                 "Failed to create material {} because it's missing required PBR metallic-roughness textures",
                 GetName(gltf_material));
    return nullptr;  // TODO: add support for optional material textures
  }

  const auto& [base_color_texture_view,
               metallic_roughness_texture_view,
               base_color_factor,
               metallic_factor,
               roughness_factor] = gltf_material.pbr_metallic_roughness;
  const auto normal_texture_view = gltf_material.normal_texture;

  const auto& base_color_ktx_texture = *maybe_base_color_ktx_texture;
  const auto& metallic_roughness_ktx_texture = *maybe_metallic_roughness_ktx_texture;
  const auto& normal_ktx_texture = *maybe_normal_ktx_texture;

  return std::make_unique<Material>(
      Texture{.image = CreateImage(device, *base_color_ktx_texture, copy_buffer_options),
              .sampler = CreateSampler(device,
                                       base_color_texture_view.texture->sampler,
                                       base_color_ktx_texture->numLevels,
                                       create_sampler_options)},
      Texture{.image = CreateImage(device, *metallic_roughness_ktx_texture, copy_buffer_options),
              .sampler = CreateSampler(device,
                                       metallic_roughness_texture_view.texture->sampler,
                                       metallic_roughness_ktx_texture->numLevels,
                                       create_sampler_options)},
      Texture{.image = CreateImage(device, *normal_ktx_texture, copy_buffer_options),
              .sampler = CreateSampler(device,
                                       normal_texture_view.texture->sampler,
                                       normal_ktx_texture->numLevels,
                                       create_sampler_options)},
      CreateBuffer<MaterialProperties>(
          MaterialProperties{.base_color_factor = ToVec(base_color_factor),
                             .metallic_roughness_factor = glm::vec2{metallic_factor, roughness_factor},
                             .normal_scale = normal_texture_view.scale},
          vk::BufferUsageFlagBits::eUniformBuffer,
          copy_buffer_options));
}

gfx::DescriptorSets CreateMaterialDescriptorSets(const vk::Device device, const std::uint32_t material_count) {
  static constexpr std::uint32_t kImagesPerMaterial = 3;

  const std::array descriptor_pool_sizes{
      vk::DescriptorPoolSize{.type = vk::DescriptorType::eUniformBuffer, .descriptorCount = material_count},
      vk::DescriptorPoolSize{.type = vk::DescriptorType::eCombinedImageSampler,
                             .descriptorCount = kImagesPerMaterial * material_count}};

  static constexpr std::array kDescriptorSetLayoutBindings{
      vk::DescriptorSetLayoutBinding{.binding = 0,
                                     .descriptorType = vk::DescriptorType::eUniformBuffer,
                                     .descriptorCount = 1,
                                     .stageFlags = vk::ShaderStageFlagBits::eFragment},
      vk::DescriptorSetLayoutBinding{.binding = 1,
                                     .descriptorType = vk::DescriptorType::eCombinedImageSampler,
                                     .descriptorCount = kImagesPerMaterial,
                                     .stageFlags = vk::ShaderStageFlagBits::eFragment}};

  return gfx::DescriptorSets{device, material_count, descriptor_pool_sizes, kDescriptorSetLayoutBindings};
}

void UpdateMaterialDescriptorSets(const vk::Device device,
                                  const gfx::DescriptorSets& material_descriptor_sets,
                                  UnorderedPtrMap<cgltf_material, Material>& materials) {
  std::vector<vk::DescriptorBufferInfo> descriptor_buffer_infos;
  descriptor_buffer_infos.resize(materials.size());

  std::vector<std::vector<vk::DescriptorImageInfo>> descriptor_image_infos;
  descriptor_image_infos.resize(materials.size());

  std::vector<vk::WriteDescriptorSet> descriptor_set_writes;
  descriptor_set_writes.reserve(materials.size());

  for (const auto& [descriptor_set, material] :
       std::views::zip(material_descriptor_sets, materials | std::views::values)) {
    if (material == nullptr) continue;  // TODO: avoid creating descriptor set for unsupported material

    const auto& [base_color_image, base_color_sampler] = *material->maybe_base_color_texture;
    const auto& [metallic_roughness_image, metallic_roughness_sampler] = *material->maybe_metallic_roughness_texture;
    const auto& [normal_image, normal_sampler] = *material->maybe_normal_texture;
    const auto& properties_buffer = *material->maybe_properties_buffer;
    material->descriptor_set = descriptor_set;

    const auto& descriptor_buffer_info = descriptor_buffer_infos.emplace_back(
        vk::DescriptorBufferInfo{.buffer = *properties_buffer, .range = vk::WholeSize});

    descriptor_set_writes.push_back(vk::WriteDescriptorSet{.dstSet = descriptor_set,
                                                           .dstBinding = 0,
                                                           .dstArrayElement = 0,
                                                           .descriptorCount = 1,
                                                           .descriptorType = vk::DescriptorType::eUniformBuffer,
                                                           .pBufferInfo = &descriptor_buffer_info});

    const auto& descriptor_image_info = descriptor_image_infos.emplace_back(
        std::initializer_list{vk::DescriptorImageInfo{.sampler = base_color_sampler,
                                                      .imageView = base_color_image.image_view(),
                                                      .imageLayout = vk::ImageLayout::eShaderReadOnlyOptimal},
                              vk::DescriptorImageInfo{.sampler = metallic_roughness_sampler,
                                                      .imageView = metallic_roughness_image.image_view(),
                                                      .imageLayout = vk::ImageLayout::eShaderReadOnlyOptimal},
                              vk::DescriptorImageInfo{.sampler = normal_sampler,
                                                      .imageView = normal_image.image_view(),
                                                      .imageLayout = vk::ImageLayout::eShaderReadOnlyOptimal}});

    descriptor_set_writes.push_back(
        vk::WriteDescriptorSet{.dstSet = descriptor_set,
                               .dstBinding = 1,
                               .dstArrayElement = 0,
                               .descriptorCount = static_cast<uint32_t>(descriptor_image_info.size()),
                               .descriptorType = vk::DescriptorType::eCombinedImageSampler,
                               .pImageInfo = descriptor_image_info.data(),
                               .pBufferInfo = &descriptor_buffer_info});
  }

  device.updateDescriptorSets(descriptor_set_writes, nullptr);
}

// ===================================================== Vertices ======================================================

template <typename T, glm::length_t N>
  requires VecConstructible<T, N>
struct VertexAttribute {
  using Data = std::vector<glm::vec<N, T>>;
  std::string_view name;  // guaranteed to reference a string literal with static storage duration
  std::optional<Data> maybe_data;
};

struct Vertex {
  glm::vec3 position{0.0f};
  glm::vec3 normal{0.0f};
  glm::vec4 tangent{0.0f};
  glm::vec2 texture_coordinates_0{0.0f};
  glm::vec4 color_0{0.0f};
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

template <glm::length_t N>
void ValidateOptionalAttribute(VertexAttribute<float, N>& vertex_attribute,
                               const std::size_t attribute_count,
                               const glm::vec<N, float>& default_value) {
  if (auto& maybe_attribute_data = vertex_attribute.maybe_data; !maybe_attribute_data.has_value()) {
    // TODO: avoid filling the vertex buffer with default values when an attribute is missing
    maybe_attribute_data = std::vector<glm::vec<N, float>>(attribute_count, default_value);
  }
}

void ValidateAttributes(const VertexAttribute<float, 3>& position_attribute,
                        const VertexAttribute<float, 3>& normal_attribute,
                        const VertexAttribute<float, 4>& tangent_attribute,
                        const VertexAttribute<float, 2>& texture_coordinates_0_attribute,
                        VertexAttribute<float, 4>& color_0_attribute) {
  ValidateRequiredAttribute(position_attribute);
  ValidateRequiredAttribute(normal_attribute);  // TODO: derive normals from positions data when missing
  ValidateRequiredAttribute(tangent_attribute);
  ValidateRequiredAttribute(texture_coordinates_0_attribute);

  const auto position_count = position_attribute.maybe_data->size();
  ValidateOptionalAttribute(color_0_attribute, position_count, glm::vec4{1.0f});

#ifndef NDEBUG
  // the glTF specification requires all primitive attributes have the same count
  for (const auto& attribute_count : {normal_attribute.maybe_data->size(),
                                      tangent_attribute.maybe_data->size(),
                                      texture_coordinates_0_attribute.maybe_data->size(),
                                      color_0_attribute.maybe_data->size()}) {
    assert(attribute_count == position_count);
  }
#endif
}

std::vector<Vertex> CreateVertices(const cgltf_primitive& gltf_primitive) {
  VertexAttribute<float, 3> position_attribute{.name = "POSITION"};
  VertexAttribute<float, 3> normal_attribute{.name = "NORMAL"};
  VertexAttribute<float, 4> tangent_attribute{.name = "TANGENT"};
  VertexAttribute<float, 2> texture_coordinates_0_attribute{.name = "TEXCOORD_0"};
  VertexAttribute<float, 4> color_0_attribute{.name = "COLOR_0"};

  for (const auto& gltf_attribute : std::span{gltf_primitive.attributes, gltf_primitive.attributes_count}) {
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
      case cgltf_attribute_type_color:
        if (TryUnpackAttribute(gltf_attribute, color_0_attribute)) continue;
        break;
      default:
        break;
    }
    std::println(std::cerr, "Unsupported primitive attribute {}", GetName(gltf_attribute));
  }

  ValidateAttributes(position_attribute,
                     normal_attribute,
                     tangent_attribute,
                     texture_coordinates_0_attribute,
                     color_0_attribute);

  return std::views::zip_transform(
             [](const auto& position,
                const auto& normal,
                const auto& tangent,
                const auto& texture_coordinates_0,
                const auto& color_0) {
#ifndef NDEBUG
               // the glTF specification requires unit length normals and tangent vectors
               static constexpr auto kEpsilon = 1.0e-6f;
               assert(glm::epsilonEqual(glm::length(normal), 1.0f, kEpsilon));
               assert(glm::epsilonEqual(glm::length(glm::vec3{tangent}), 1.0f, kEpsilon));
#endif
               return Vertex{.position = position,
                             .normal = normal,
                             .tangent = tangent,
                             .texture_coordinates_0 = texture_coordinates_0,
                             .color_0 = color_0};
             },
             *position_attribute.maybe_data,
             *normal_attribute.maybe_data,
             *tangent_attribute.maybe_data,
             *texture_coordinates_0_attribute.maybe_data,
             *color_0_attribute.maybe_data)
         | std::ranges::to<std::vector>();
}

// ====================================================== Meshes =======================================================

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

IndexBuffer CreateIndexBuffer(const cgltf_accessor& gltf_accessor, CopyBufferOptions& copy_buffer_options) {
  switch (const auto component_size = cgltf_component_size(gltf_accessor.component_type)) {
    case 2: {
      const auto indices = UnpackIndices<std::uint16_t>(gltf_accessor);
      return IndexBuffer{
          .index_count = static_cast<std::uint32_t>(indices.size()),
          .index_type = vk::IndexType::eUint16,
          .buffer = CreateBuffer<std::uint16_t>(indices, vk::BufferUsageFlagBits::eIndexBuffer, copy_buffer_options)};
    }
    case 4: {
      const auto indices = UnpackIndices<std::uint32_t>(gltf_accessor);
      return IndexBuffer{
          .index_count = static_cast<std::uint32_t>(indices.size()),
          .index_type = vk::IndexType::eUint32,
          .buffer = CreateBuffer<std::uint32_t>(indices, vk::BufferUsageFlagBits::eIndexBuffer, copy_buffer_options)};
    }
    default: {
      static constexpr auto kBitsPerByte = 8;
      throw std::runtime_error{std::format("Unsupported {}-bit index type", component_size * kBitsPerByte)};
    }
  }
}

std::unique_ptr<const Mesh> CreateMesh(const cgltf_mesh& gltf_mesh,
                                       const UnorderedPtrMap<cgltf_material, Material>& materials,
                                       CopyBufferOptions& copy_buffer_options) {
  std::vector<Primitive> primitives;
  primitives.reserve(gltf_mesh.primitives_count);

  for (auto index = 0; const auto& gltf_primitive : std::span{gltf_mesh.primitives, gltf_mesh.primitives_count}) {
    if (gltf_primitive.type != cgltf_primitive_type_triangles) {
      static constexpr auto kMessageFormat = "Mesh {} primitive {} with type {} is unsupported";
      std::println(std::cerr, kMessageFormat, GetName(gltf_mesh), index++, gltf_primitive.type);
      continue;  // TODO: add support for other primitive types
    }

    if (gltf_primitive.indices == nullptr || gltf_primitive.indices->count == 0) {
      static constexpr auto kMessageFormat = "Mesh {} primitive {} without an indices accessor is unsupported";
      std::println(std::cerr, kMessageFormat, GetName(gltf_mesh), index++);
      continue;  // TODO: add support for non-indexed triangle meshes
    }

    auto* const material = Find(gltf_primitive.material, materials);
    if (material == nullptr) {
      static constexpr auto kMessageFormat = "Mesh {} primitive {} with material {} is unsupported";
      std::println(std::cerr, kMessageFormat, GetName(gltf_mesh), index++, GetName(*gltf_primitive.material));
      continue;  // TODO: add default material support
    }

    const auto vertices = CreateVertices(gltf_primitive);
    primitives.emplace_back(CreateBuffer<Vertex>(vertices, vk::BufferUsageFlagBits::eVertexBuffer, copy_buffer_options),
                            CreateIndexBuffer(*gltf_primitive.indices, copy_buffer_options),
                            material);
    ++index;
  }

  return std::make_unique<const Mesh>(std::move(primitives));
}

// ======================================================= Nodes =======================================================

glm::mat4 GetTransform(const cgltf_node& gltf_node) {
  glm::mat4 transform{0.0f};
  cgltf_node_transform_local(&gltf_node, glm::value_ptr(transform));
  return transform;
}

std::vector<std::unique_ptr<const Node>> CreateNodes(const cgltf_node* const* const gltf_nodes,
                                                     const cgltf_size gltf_nodes_count,
                                                     const UnorderedPtrMap<cgltf_mesh, const Mesh>& meshes,
                                                     const UnorderedPtrMap<cgltf_light, Light>& lights) {
  return std::span{gltf_nodes, gltf_nodes_count}  //
         | std::views::transform([&meshes, &lights](const auto* const gltf_node) {
             return std::make_unique<const Node>(
                 Find(gltf_node->mesh, meshes),
                 Find(gltf_node->light, lights),
                 GetTransform(*gltf_node),
                 CreateNodes(gltf_node->children, gltf_node->children_count, meshes, lights));
           })
         | std::ranges::to<std::vector>();
}

// ================================================= Graphics Pipeline =================================================

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
  const std::filesystem::path vertex_shader_filepath{"shaders/vertex.glsl"};
  const gfx::ShaderModule vertex_shader_module{device, vertex_shader_filepath, vk::ShaderStageFlagBits::eVertex};

  const std::filesystem::path fragment_shader_filepath{"shaders/fragment.glsl"};
  const gfx::ShaderModule fragment_shader_module{device, fragment_shader_filepath, vk::ShaderStageFlagBits::eFragment};

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
                                          .offset = offsetof(Vertex, texture_coordinates_0)},
      vk::VertexInputAttributeDescription{.location = 4,
                                          .binding = 0,
                                          .format = vk::Format::eR32G32B32A32Sfloat,
                                          .offset = offsetof(Vertex, color_0)}};

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

// ==================================================== Rendering ======================================================

void Render(const Mesh& mesh,
            const glm::mat4& node_transform,
            const vk::PipelineLayout graphics_pipeline_layout,
            const vk::CommandBuffer command_buffer) {
  using ModelTransform = decltype(PushConstants::model_transform);
  command_buffer.pushConstants<ModelTransform>(graphics_pipeline_layout,
                                               vk::ShaderStageFlagBits::eVertex,
                                               offsetof(PushConstants, model_transform),
                                               node_transform);

  for (const auto& [vertex_buffer, index_buffer, material] : mesh.primitives) {
    command_buffer.bindDescriptorSets(vk::PipelineBindPoint::eGraphics,
                                      graphics_pipeline_layout,
                                      1,
                                      material->descriptor_set,
                                      nullptr);
    command_buffer.bindVertexBuffers(0, *vertex_buffer, static_cast<vk::DeviceSize>(0));
    command_buffer.bindIndexBuffer(*index_buffer.buffer, 0, index_buffer.index_type);
    command_buffer.drawIndexed(index_buffer.index_count, 1, 0, 0, 0);
  }
}

void Render(const Node& node,
            const glm::mat4& parent_transform,
            const vk::PipelineLayout graphics_pipeline_layout,
            const vk::CommandBuffer command_buffer,
            std::vector<Light>& lights_buffer) {
  const auto node_transform = parent_transform * node.transform;

  if (node.light != nullptr) {
    if (node.light->position.w == 0.0f) {
      const auto& direction = node_transform[2];  // light direction is derived from the node orientation z-axis
      lights_buffer.emplace_back(glm::normalize(direction), node.light->color);
    } else {
      assert(node.light->position.w == 1.0f);
      const auto& position = node_transform[3];  // light position derived from the node translation vector
      lights_buffer.emplace_back(position, node.light->color);
    }
  }

  if (const auto* const mesh = node.mesh; mesh != nullptr) {
    Render(*mesh, node_transform, graphics_pipeline_layout, command_buffer);
  }

  for (const auto& child_node : node.children) {
    Render(*child_node, node_transform, graphics_pipeline_layout, command_buffer, lights_buffer);
  }
}

}  // namespace

namespace gfx {

struct CameraTransforms {
  glm::mat4 view_transform{0.0f};
  glm::mat4 projection_transform{0.0f};
};

GltfScene::GltfScene(const std::filesystem::path& gltf_filepath,
                     const vk::PhysicalDevice physical_device,
                     const vk::Bool32 enable_sampler_anisotropy,
                     const float max_sampler_anisotropy,
                     const vk::Device device,
                     const vk::Queue queue,
                     const std::uint32_t queue_family_index,
                     const vk::Extent2D viewport_extent,
                     const vk::SampleCountFlagBits msaa_sample_count,
                     const vk::RenderPass render_pass,
                     const VmaAllocator allocator,
                     const std::size_t max_render_frames) {
  const auto gltf_directory = gltf_filepath.parent_path();
  const auto gltf_data = Load(gltf_filepath.string());

  auto material_futures =
      std::span{gltf_data->materials, gltf_data->materials_count}
      | std::views::transform([&gltf_directory, physical_device](const auto& gltf_material) {
          return std::async(std::launch::async, [&gltf_material, &gltf_directory, physical_device] {
            return std::pair{&gltf_material, CreateKtxTextures(gltf_material, gltf_directory, physical_device)};
          });
        })
      | std::ranges::to<std::vector>();

  std::unordered_map<vk::SamplerCreateInfo, vk::UniqueSampler> samplers;
  CreateSamplerOptions create_sampler_options{.enable_anisotropy = enable_sampler_anisotropy,
                                              .max_anisotropy = max_sampler_anisotropy,
                                              .samplers = &samplers};

  const auto staging_buffer_count = gltf_data->buffers_count + gltf_data->images_count;
  auto copy_buffer_options = CreateCopyBufferOptions(device, queue_family_index, staging_buffer_count, allocator);

  const auto copy_command_buffer = *copy_buffer_options.command_buffer;
  copy_command_buffer.begin(vk::CommandBufferBeginInfo{.flags = vk::CommandBufferUsageFlagBits::eOneTimeSubmit});

  auto materials =
      material_futures
      | std::views::transform([device, &copy_buffer_options, &create_sampler_options](auto& material_future) {
          auto [gltf_material, ktx_textures] = material_future.get();
          return std::pair{
              gltf_material,
              CreateMaterial(device, *gltf_material, ktx_textures, copy_buffer_options, create_sampler_options)};
        })
      | std::ranges::to<std::unordered_map>();

  auto meshes = std::span{gltf_data->meshes, gltf_data->meshes_count}
                | std::views::transform([&materials, &copy_buffer_options](const auto& gltf_mesh) {
                    return std::pair{&gltf_mesh, CreateMesh(gltf_mesh, materials, copy_buffer_options)};
                  })
                | std::ranges::to<std::unordered_map>();

  copy_command_buffer.end();

  const auto copy_fence = device.createFenceUnique(vk::FenceCreateInfo{});
  queue.submit(vk::SubmitInfo{.commandBufferCount = 1, .pCommandBuffers = &copy_command_buffer}, *copy_fence);

  auto lights =
      std::span{gltf_data->lights, gltf_data->lights_count}  //
      | std::views::filter([](const auto& gltf_light) {
          switch (gltf_light.type) {
            case cgltf_light_type_directional:
            case cgltf_light_type_point:
              return true;
            default:
              std::println(std::cerr, "Unsupported light {} with type {}", GetName(gltf_light), gltf_light.type);
              return false;  // TODO: add spot light support
          }
        })
      | std::views::transform([](const auto& gltf_light) {
          const glm::vec4 light_position{glm::vec3{0.0f},  // light position set dynamically based on node transform
                                         static_cast<float>(gltf_light.type == cgltf_light_type_point)};
          const glm::vec4 light_color{ToVec(gltf_light.color), 1.0f};
          return std::pair{&gltf_light, std::make_unique<Light>(light_position, light_color)};
        })
      | std::ranges::to<std::unordered_map>();

  const auto& gltf_scene = GetDefaultScene(*gltf_data);
  root_node_ = std::make_unique<const Node>(nullptr,
                                            nullptr,
                                            glm::mat4{1.0f},
                                            CreateNodes(gltf_scene.nodes, gltf_scene.nodes_count, meshes, lights));

  camera_buffers_ = CreateMappedUniformBuffers(max_render_frames, sizeof(CameraTransforms), allocator);
  light_buffers_ = CreateMappedUniformBuffers(max_render_frames, sizeof(Light) * lights.size(), allocator);
  global_descriptor_sets_ = CreateGlobalDescriptorSets(device, static_cast<std::uint32_t>(max_render_frames));
  UpdateGlobalDescriptorSets(device, global_descriptor_sets_, camera_buffers_, light_buffers_);

  material_descriptor_sets_ = CreateMaterialDescriptorSets(device, static_cast<std::uint32_t>(materials.size()));
  UpdateMaterialDescriptorSets(device, material_descriptor_sets_, materials);

  graphics_pipeline_layout_ = CreateGraphicsPipelineLayout(
      device,
      std::array{global_descriptor_sets_.descriptor_set_layout(), material_descriptor_sets_.descriptor_set_layout()});
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
  command_buffer.bindPipeline(vk::PipelineBindPoint::eGraphics, *graphics_pipeline_);
  command_buffer.bindDescriptorSets(vk::PipelineBindPoint::eGraphics,
                                    *graphics_pipeline_layout_,
                                    0,
                                    global_descriptor_sets_[frame_index],
                                    nullptr);

  using ViewPosition = decltype(PushConstants::view_position);
  command_buffer.pushConstants<ViewPosition>(*graphics_pipeline_layout_,
                                             vk::ShaderStageFlagBits::eFragment,
                                             offsetof(PushConstants, view_position),
                                             camera.GetPosition());

  camera_buffers_[frame_index].Copy<CameraTransforms>(
      CameraTransforms{.view_transform = camera.view_transform(),
                       .projection_transform = camera.projection_transform()});

  std::vector<Light> lights_buffer;
  lights_buffer.reserve(lights_.size());

  for (const auto& child_node : root_node_->children) {
    ::Render(*child_node, root_node_->transform, *graphics_pipeline_layout_, command_buffer, lights_buffer);
  }

  light_buffers_[frame_index].Copy<Light>(lights_buffer);
}

}  // namespace gfx
