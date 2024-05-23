#include "graphics/scene.h"

#include <array>
#include <cassert>
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
#include <string_view>
#include <utility>

#include <cgltf.h>
#include <ktx.h>
#include <glm/glm.hpp>
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

template <>
struct std::formatter<cgltf_attribute_type> : std::formatter<std::string_view> {
  [[nodiscard]] auto format(const cgltf_attribute_type gltf_attribute_type, auto& format_context) const {
    return std::formatter<std::string_view>::format(to_string(gltf_attribute_type), format_context);
  }

private:
  static std::string_view to_string(const cgltf_attribute_type gltf_attribute_type) noexcept {
    switch (gltf_attribute_type) {
      // clang-format off
#define CASE(kValue) case kValue: return #kValue;  // NOLINT(cppcoreguidelines-macro-usage)
      CASE(cgltf_attribute_type_position)
      CASE(cgltf_attribute_type_normal)
      CASE(cgltf_attribute_type_tangent)
      CASE(cgltf_attribute_type_texcoord)
      CASE(cgltf_attribute_type_color)
      CASE(cgltf_attribute_type_joints)
      CASE(cgltf_attribute_type_weights)
      CASE(cgltf_attribute_type_custom)
#undef CASE
      // clang-format on
      default:
        std::unreachable();
    }
  }
};

namespace {

struct Vertex {
  glm::vec3 position{0.0f};
  glm::vec3 normal{0.0f};
  glm::vec2 texture_coordinates{0.0f};
};

struct PushConstants {
  glm::mat4 model_view_transform{1.0f};
  glm::mat4 projection_transform{1.0f};
};

using UniqueGltfData = std::unique_ptr<cgltf_data, decltype(&cgltf_free)>;

// NOLINTNEXTLINE(cppcoreguidelines-pro-type-cstyle-cast) }
void DestroyKtxTexture2(ktxTexture2* const ktx_texture2) noexcept { ktxTexture_Destroy(ktxTexture(ktx_texture2)); }
using UniqueKtxTexture2 = std::unique_ptr<ktxTexture2, decltype(&DestroyKtxTexture2)>;

UniqueGltfData ParseFile(const char* const gltf_filepath) {
  static constexpr cgltf_options kGltfOptions{};
  UniqueGltfData gltf_data{nullptr, nullptr};

  if (const auto gltf_result = cgltf_parse_file(&kGltfOptions, gltf_filepath, std::out_ptr(gltf_data, cgltf_free));
      gltf_result != cgltf_result_success) {
    throw std::runtime_error{std::format("Failed to parse {} with error {}", gltf_filepath, gltf_result)};
  }
  if (const auto gltf_result = cgltf_load_buffers(&kGltfOptions, gltf_data.get(), gltf_filepath);
      gltf_result != cgltf_result_success) {
    throw std::runtime_error{std::format("Failed to load buffers for {} with error {}", gltf_filepath, gltf_result)};
  }
#ifndef NDEBUG
  if (const auto gltf_result = cgltf_validate(gltf_data.get()); gltf_result != cgltf_result_success) {
    throw std::runtime_error{std::format("Failed to validate {} with error {}", gltf_filepath, gltf_result)};
  }
#endif

  return gltf_data;
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
  if (const auto components = cgltf_num_components(gltf_accessor.type); components != N) {
    throw std::runtime_error{std::format("Failed to unpack floats for {}", GetName(gltf_accessor))};
  }
  std::vector<glm::vec<N, float>> data(gltf_accessor.count);
  cgltf_accessor_unpack_floats(&gltf_accessor, glm::value_ptr(data.front()), N * gltf_accessor.count);
  return data;
}

std::vector<Vertex> GetVertices(const cgltf_primitive& gltf_primitive) {
  std::optional<std::vector<glm::vec3>> maybe_positions, maybe_normals;
  std::optional<std::vector<glm::vec2>> maybe_texture_coordinates;

  for (const auto& gltf_attribute : std::span{gltf_primitive.attributes, gltf_primitive.attributes_count}) {
    switch (const auto& gltf_accessor = *gltf_attribute.data; gltf_attribute.type) {
      case cgltf_attribute_type_position:
        if (!maybe_positions.has_value()) {
          maybe_positions = UnpackFloats<3>(gltf_accessor);
          break;
        }
        [[fallthrough]];
      case cgltf_attribute_type_normal:
        if (!maybe_normals.has_value()) {
          maybe_normals = UnpackFloats<3>(gltf_accessor);
          break;
        }
        [[fallthrough]];
      case cgltf_attribute_type_texcoord:
        if (!maybe_texture_coordinates.has_value()) {
          maybe_texture_coordinates = UnpackFloats<2>(gltf_accessor);
          break;
        }
        [[fallthrough]];
      default:
        std::println(std::cerr, "Unsupported primitive attribute {}", gltf_attribute.type);
        break;
    }
  }

  if (!maybe_positions.has_value()) throw std::runtime_error{"No vertex positions"};
  if (!maybe_normals.has_value()) throw std::runtime_error{"No vertex normals"};
  if (!maybe_texture_coordinates.has_value()) throw std::runtime_error{"No vertex texture coordinates"};

  // primitives are expected to represent an indexed triangle mesh with a matching number of vertex attributes
  if (maybe_positions->size() != maybe_normals->size()) {
    throw std::runtime_error{
        std::format("The number of vertex positions {} does not match the number of vertex normals {}",
                    maybe_positions->size(),
                    maybe_normals->size())};
  }
  if (maybe_positions->size() != maybe_texture_coordinates->size()) {
    throw std::runtime_error{
        std::format("The number of vertex positions {} does not match the number of vertex texture coordinates {}",
                    maybe_positions->size(),
                    maybe_texture_coordinates->size())};
  }

  return std::views::zip_transform(
             [](const auto& position, const auto& normal, const auto& texture_coordinates) {
               return Vertex{.position = position, .normal = normal, .texture_coordinates = texture_coordinates};
             },
             *maybe_positions,
             *maybe_normals,
             *maybe_texture_coordinates)
         | std::ranges::to<std::vector>();
}

// TODO(matthew-rister): add support for 32-bit indices
std::vector<std::uint16_t> GetIndices(const cgltf_primitive& gltf_primitive) {
  const auto* const gltf_accessor = gltf_primitive.indices;
  if (gltf_accessor == nullptr || gltf_accessor->count == 0) {
    throw std::runtime_error{"Primitive must represent an indexed triangle mesh"};
  }
  std::vector<std::uint16_t> indices(gltf_accessor->count);
  if (cgltf_accessor_unpack_indices(gltf_accessor, indices.data(), sizeof(std::uint16_t), indices.size()) == 0) {
    throw std::runtime_error{std::format("Failed to unpack indices for {}", GetName(*gltf_accessor))};
  }
  return indices;
}

glm::mat4 GetTransform(const cgltf_node& gltf_node) {
  glm::mat4 transform{1.0f};
  if (gltf_node.has_matrix != 0 || gltf_node.has_translation != 0 || gltf_node.has_rotation != 0
      || gltf_node.has_scale != 0) {
    cgltf_node_transform_local(&gltf_node, glm::value_ptr(transform));
  }
  return transform;
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

std::vector<gfx::Mesh> CreateSubmeshes(
    const cgltf_mesh& gltf_mesh,
    const vk::CommandBuffer command_buffer,
    const VmaAllocator allocator,
    const std::unordered_map<const cgltf_material*, vk::DescriptorSet>& descriptor_sets_by_gltf_material,
    std::vector<gfx::Buffer>& staging_buffers) {
  return std::span{gltf_mesh.primitives, gltf_mesh.primitives_count}
         | std::views::filter([&descriptor_sets_by_gltf_material](const auto& gltf_primitive) {
             if (gltf_primitive.type != cgltf_primitive_type_triangles) {
               std::println(std::cerr, "Unsupported primitive type {}", gltf_primitive.type);
               return false;
             }
             return descriptor_sets_by_gltf_material.contains(gltf_primitive.material);  // exclude unsupported material
           })
         | std::views::transform([command_buffer, allocator, &staging_buffers, &descriptor_sets_by_gltf_material](
                                     const auto& gltf_primitive) {
             using enum vk::BufferUsageFlagBits;
             const auto vertices = GetVertices(gltf_primitive);
             const auto indices = GetIndices(gltf_primitive);
             return gfx::Mesh{CreateBuffer(vertices, eVertexBuffer, command_buffer, allocator, staging_buffers),
                              CreateBuffer(indices, eIndexBuffer, command_buffer, allocator, staging_buffers),
                              static_cast<std::uint32_t>(indices.size()),
                              descriptor_sets_by_gltf_material.find(gltf_primitive.material)->second};
           })
         | std::ranges::to<std::vector>();
}

std::vector<gfx::Mesh> GetSubmeshes(
    const cgltf_mesh* const gltf_mesh,
    std::unordered_map<const cgltf_mesh*, std::vector<gfx::Mesh>>& submeshes_by_gltf_mesh) {
  if (gltf_mesh == nullptr) return {};
  const auto iterator = submeshes_by_gltf_mesh.find(gltf_mesh);
  assert(iterator != submeshes_by_gltf_mesh.cend());
  return std::move(iterator->second);
}

ktx_transcode_fmt_e GetTranscodeFormat(ktxTexture2& ktx_texture2,
                                       const vk::PhysicalDeviceFeatures& physical_device_features) {
  // format selection based on https://github.com/KhronosGroup/3D-Formats-Guidelines/blob/main/KTXDeveloperGuide.md
  // TODO(matthew-rister): check corresponding vk::Format support
  switch (ktxTexture2_GetColorModel_e(&ktx_texture2)) {
    case KHR_DF_MODEL_UASTC:
      if (physical_device_features.textureCompressionASTC_LDR == vk::True) return KTX_TTF_ASTC_4x4_RGBA;
      if (physical_device_features.textureCompressionBC == vk::True) return KTX_TTF_BC7_RGBA;
      if (physical_device_features.textureCompressionETC2 == vk::True) return KTX_TTF_ETC;
      break;
    case KHR_DF_MODEL_ETC1S:
      if (physical_device_features.textureCompressionETC2 == vk::True) return KTX_TTF_ETC;
      if (physical_device_features.textureCompressionBC == vk::True) return KTX_TTF_BC7_RGBA;
      break;
    default:
      std::unreachable();  // basis universal only supports UASTC/ETC1S transmission formats
  }

  static constexpr auto kDecompressionFallback = KTX_TTF_RGBA32;
  std::println(std::cerr,
               "No supported texture compression format could be found. Decompressing to {}",
               ktxTranscodeFormatString(kDecompressionFallback));
  return kDecompressionFallback;
}

UniqueKtxTexture2 LoadBaseColorTexture(const cgltf_material& gltf_material,
                                       const std::filesystem::path& gltf_parent_filepath,
                                       const vk::PhysicalDeviceFeatures& physical_device_features) {
  UniqueKtxTexture2 ktx_texture2{nullptr, nullptr};
  if (gltf_material.has_pbr_metallic_roughness != 0) {
    const auto& gltf_pbr_metallic_roughness = gltf_material.pbr_metallic_roughness;
    const auto& gltf_base_color_texture = gltf_pbr_metallic_roughness.base_color_texture.texture;

    if (gltf_base_color_texture != nullptr) {
      const auto ktx_filepath = gltf_parent_filepath / gltf_base_color_texture->image->uri;
      if (const auto ktx_error_code = ktxTexture2_CreateFromNamedFile(ktx_filepath.string().c_str(),
                                                                      KTX_TEXTURE_CREATE_CHECK_GLTF_BASISU_BIT,
                                                                      std::out_ptr(ktx_texture2, DestroyKtxTexture2));
          ktx_error_code != KTX_SUCCESS) {
        throw std::runtime_error{std::format("Failed to create KTX texture for {} with error {}",
                                             ktx_filepath.string(),
                                             ktxErrorString(ktx_error_code))};
      }

      if (ktxTexture2_NeedsTranscoding(ktx_texture2.get())) {
        const auto ktx_transcode_format = GetTranscodeFormat(*ktx_texture2, physical_device_features);
        if (const auto ktx_error_code = ktxTexture2_TranscodeBasis(ktx_texture2.get(), ktx_transcode_format, 0);
            ktx_error_code != KTX_SUCCESS) {
          throw std::runtime_error{std::format("Failed to transcode {} to {} with error {}",
                                               ktx_filepath.string(),
                                               ktxTranscodeFormatString(ktx_transcode_format),
                                               ktxErrorString(ktx_error_code))};
        }
      }
    }
  }
  return ktx_texture2;
}

gfx::Image CreateImage(ktxTexture2& ktx_texture2,
                       const vk::Device device,
                       const vk::CommandBuffer command_buffer,
                       const VmaAllocator allocator,
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
          ktx_size_t buffer_offset = 0;
          if (const auto ktx_error_code =  // NOLINTNEXTLINE(cppcoreguidelines-pro-type-cstyle-cast) }
              ktxTexture_GetImageOffset(ktxTexture(&ktx_texture2), mip_level, 0, 0, &buffer_offset);
              ktx_error_code != KTX_SUCCESS) {
            throw std::runtime_error{std::format("Failed to get image offset for mip level {} with error {}",
                                                 mip_level,
                                                 ktxErrorString(ktx_error_code))};
          }
          return vk::BufferImageCopy{
              .bufferOffset = buffer_offset,
              .imageSubresource = vk::ImageSubresourceLayers{.aspectMask = vk::ImageAspectFlagBits::eColor,
                                                             .mipLevel = mip_level,
                                                             .layerCount = 1},
              .imageExtent = vk::Extent3D{.width = ktx_texture2.baseWidth >> mip_level,
                                          .height = ktx_texture2.baseHeight >> mip_level,
                                          .depth = 1}};
        })
      | std::ranges::to<std::vector>();
  image.Copy(*staging_buffer, buffer_image_copies, command_buffer);

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
                                          .format = vk::Format::eR32G32Sfloat,
                                          .offset = offsetof(Vertex, texture_coordinates)}};

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

class Scene::Node {
public:
  Node(const cgltf_scene& gltf_scene, std::unordered_map<const cgltf_mesh*, std::vector<Mesh>>& submeshes_by_gltf_mesh);
  Node(const cgltf_node& gltf_node, std::unordered_map<const cgltf_mesh*, std::vector<Mesh>>& submeshes_by_gltf_mesh);

  void Render(const glm::mat4& model_transform,
              const glm::mat4& view_transform,
              const glm::mat4& projection_transform,
              const vk::PipelineLayout pipeline_layout,
              const vk::CommandBuffer command_buffer) const;

private:
  std::vector<Mesh> meshes_;
  std::vector<std::unique_ptr<const Node>> children_;
  glm::mat4 transform_{1.0f};
};

struct Scene::Material {
  const cgltf_material* gltf_material = nullptr;
  Image base_color_image;
  vk::Sampler base_color_sampler;
};

Scene::Scene(const std::filesystem::path& gltf_filepath,
             const vk::PhysicalDeviceFeatures& physical_device_features,
             const vk::PhysicalDeviceLimits& physical_device_limits,
             const vk::Device device,
             const vk::Queue queue,
             const std::uint32_t queue_family_index,
             const vk::Extent2D viewport_extent,
             const vk::SampleCountFlagBits msaa_sample_count,
             const vk::RenderPass render_pass,
             const VmaAllocator allocator) {
  const auto gltf_data = ParseFile(gltf_filepath.string().c_str());
  const auto gltf_parent_filepath = gltf_filepath.parent_path();

  auto material_futures =
      std::span{gltf_data->materials, gltf_data->materials_count}
      | std::views::transform([&gltf_parent_filepath, &physical_device_features](const auto& gltf_material) {
          return std::async(std::launch::async, [&gltf_material, &gltf_parent_filepath, &physical_device_features] {
            return std::pair{&gltf_material,
                             LoadBaseColorTexture(gltf_material, gltf_parent_filepath, physical_device_features)};
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
    const auto [gltf_material, base_color_texture] = material_future.get();
    if (base_color_texture == nullptr) {
      std::println(std::cerr, "Unsupported material {}", GetName(*gltf_material));
      continue;
    }
    materials_.emplace_back(gltf_material,
                            CreateImage(*base_color_texture, device, command_buffer, allocator, staging_buffers),
                            CreateSampler(device,
                                          base_color_texture->numLevels,
                                          physical_device_features,
                                          physical_device_limits,
                                          samplers_));
  }

  const auto descriptor_sets_by_gltf_material =
      std::views::zip_transform(
          [device](const auto& material, auto& descriptor_set) {
            const auto& [gltf_material, base_color_image, base_color_sampler] = material;
            UpdateDescriptorSet(device, base_color_image.image_view(), base_color_sampler, descriptor_set);
            return std::pair{gltf_material, descriptor_set};
          },
          materials_,
          AllocateDescriptorSets(device, *descriptor_pool_, *descriptor_set_layout_, material_count))
      | std::ranges::to<std::unordered_map>();

  auto submeshes_by_gltf_mesh =
      std::span{gltf_data->meshes, gltf_data->meshes_count}
      | std::views::transform([command_buffer, allocator, &staging_buffers, &descriptor_sets_by_gltf_material](
                                  const auto& gltf_mesh) {
          return std::pair{
              &gltf_mesh,
              CreateSubmeshes(gltf_mesh, command_buffer, allocator, descriptor_sets_by_gltf_material, staging_buffers)};
        })
      | std::ranges::to<std::unordered_map>();

  command_buffer.end();

  const auto fence = device.createFenceUnique(vk::FenceCreateInfo{});
  queue.submit(vk::SubmitInfo{.commandBufferCount = 1, .pCommandBuffers = &command_buffer}, *fence);

  root_node_ = std::make_unique<const Node>(*gltf_data->scene, submeshes_by_gltf_mesh);

  static constexpr auto kMaxTimeout = std::numeric_limits<std::uint64_t>::max();
  const auto result = device.waitForFences(*fence, vk::True, kMaxTimeout);
  vk::resultCheck(result, "Fence failed to enter a signaled state");
}

Scene::~Scene() noexcept = default;  // this is necessary to enable forward declaring Scene::Node with std::unique_ptr

void Scene::Render(const Camera& camera, const vk::CommandBuffer command_buffer) const {
  static constexpr glm::mat4 kModelTransform{1.0f};
  command_buffer.bindPipeline(vk::PipelineBindPoint::eGraphics, *pipeline_);
  root_node_->Render(kModelTransform,
                     camera.GetViewTransform(),
                     camera.GetProjectionTransform(),
                     *pipeline_layout_,
                     command_buffer);
}

Scene::Node::Node(const cgltf_scene& gltf_scene,
                  std::unordered_map<const cgltf_mesh*, std::vector<Mesh>>& submeshes_by_gltf_mesh)
    : children_{std::span{gltf_scene.nodes, gltf_scene.nodes_count}
                | std::views::transform([&submeshes_by_gltf_mesh](const auto* const scene_gltf_node) {
                    return std::make_unique<const Node>(*scene_gltf_node, submeshes_by_gltf_mesh);
                  })
                | std::ranges::to<std::vector>()} {}

Scene::Node::Node(const cgltf_node& gltf_node,
                  std::unordered_map<const cgltf_mesh*, std::vector<Mesh>>& submeshes_by_gltf_mesh)
    : meshes_{GetSubmeshes(gltf_node.mesh, submeshes_by_gltf_mesh)},
      children_{std::span{gltf_node.children, gltf_node.children_count}
                | std::views::transform([&submeshes_by_gltf_mesh](const auto* const child_gltf_node) {
                    return std::make_unique<const Node>(*child_gltf_node, submeshes_by_gltf_mesh);
                  })
                | std::ranges::to<std::vector>()},
      transform_{GetTransform(gltf_node)} {}

void Scene::Node::Render(const glm::mat4& model_transform,
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
