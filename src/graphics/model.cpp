#include "graphics/model.h"

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
  [[nodiscard]] auto format(const cgltf_result result, auto& format_context) const {
    return std::formatter<std::string_view>::format(to_string(result), format_context);
  }

private:
  static std::string_view to_string(const cgltf_result result) noexcept {
    switch (result) {
      // clang-format off
#define CASE(kResult) case kResult: return #kResult;  // NOLINT(cppcoreguidelines-macro-usage)
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
  [[nodiscard]] auto format(const cgltf_primitive_type primitive_type, auto& format_context) const {
    return std::formatter<std::string_view>::format(to_string(primitive_type), format_context);
  }

private:
  static std::string_view to_string(const cgltf_primitive_type primitive_type) noexcept {
    switch (primitive_type) {
      // clang-format off
#define CASE(kResult) case kResult: return #kResult;  // NOLINT(cppcoreguidelines-macro-usage)
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

using UniqueCgltfData = std::unique_ptr<cgltf_data, decltype(&cgltf_free)>;

// NOLINTNEXTLINE(cppcoreguidelines-pro-type-cstyle-cast) }
void DestroyKtxTexture2(ktxTexture2* const ktx_texture2) noexcept { ktxTexture_Destroy(ktxTexture(ktx_texture2)); }
using UniqueKtxTexture2 = std::unique_ptr<ktxTexture2, decltype(&DestroyKtxTexture2)>;

struct Vertex {
  glm::vec3 position{0.0f};
  glm::vec3 normal{0.0f};
  glm::vec2 texture_coordinates{0.0f};
};

struct PushConstants {
  glm::mat4 model_view_transform{1.0f};
  glm::mat4 projection_transform{1.0f};
};

UniqueCgltfData ParseFile(const char* const gltf_filepath) {
  static constexpr cgltf_options kOptions{};
  UniqueCgltfData gltf_data{nullptr, nullptr};

  if (const auto result = cgltf_parse_file(&kOptions, gltf_filepath, std::out_ptr(gltf_data, cgltf_free));
      result != cgltf_result_success) {
    throw std::runtime_error{std::format("Failed to parse {} with error {}", gltf_filepath, result)};
  }
  if (const auto result = cgltf_load_buffers(&kOptions, gltf_data.get(), gltf_filepath);
      result != cgltf_result_success) {
    throw std::runtime_error{std::format("Failed to load buffers for {} with error {}", gltf_filepath, result)};
  }
#ifndef NDEBUG
  if (const auto result = cgltf_validate(gltf_data.get()); result != cgltf_result_success) {
    throw std::runtime_error{std::format("Failed to validate {} with error {}", gltf_filepath, result)};
  }
#endif

  return gltf_data;
}

template <typename T>
  requires requires(T element) {
    { element.name } -> std::same_as<char*&>;
  }
std::string_view GetName(const T& element) {
  const auto size = element.name == nullptr ? 0 : std::strlen(element.name);
  return size == 0 ? "unknown" : std::string_view{element.name, size};
}

template <glm::length_t N>
std::vector<glm::vec<N, float>> UnpackFloats(const cgltf_accessor& accessor) {
  if (const auto components = cgltf_num_components(accessor.type); components != N) {
    throw std::runtime_error{
        std::format("Failed to unpack floats for {} with {} components", GetName(accessor), components)};
  }
  std::vector<glm::vec<N, float>> data(accessor.count);
  cgltf_accessor_unpack_floats(&accessor, glm::value_ptr(data.front()), N * accessor.count);
  return data;
}

// TODO(matthew-rister): improve error logging for this method
std::vector<Vertex> GetVertices(const cgltf_primitive& primitive) {
  std::optional<std::vector<glm::vec3>> positions, normals;
  std::optional<std::vector<glm::vec2>> texture_coordinates;

  for (const auto& attribute : std::span{primitive.attributes, primitive.attributes_count}) {
    switch (const auto& accessor = *attribute.data; attribute.type) {
      case cgltf_attribute_type_position:
        if (!positions.has_value()) positions = UnpackFloats<3>(accessor);
        break;
      case cgltf_attribute_type_normal:
        if (!normals.has_value()) normals = UnpackFloats<3>(accessor);
        break;
      case cgltf_attribute_type_texcoord:
        if (!texture_coordinates.has_value()) texture_coordinates = UnpackFloats<2>(accessor);
        break;
      default:
        break;
    }
  }

  return std::views::zip_transform(
             [](const auto& position, const auto& normal, const auto& texture_coords) {
               return Vertex{.position = position, .normal = normal, .texture_coordinates = texture_coords};
             },
             positions.value(),
             normals.value(),
             texture_coordinates.value())
         | std::ranges::to<std::vector>();
}

std::vector<std::uint16_t> GetIndices(const cgltf_accessor& accessor) {
  return std::views::iota(0u, accessor.count)  //
         | std::views::transform([&accessor](const auto accessor_index) {
             const auto vertex_index = cgltf_accessor_read_index(&accessor, accessor_index);
             return static_cast<std::uint16_t>(vertex_index);
           })
         | std::ranges::to<std::vector>();
}

glm::mat4 GetTransform(const cgltf_node& node) {
  glm::mat4 transform{1.0f};
  if (node.has_matrix != 0 || node.has_translation != 0 || node.has_rotation != 0 || node.has_scale != 0) {
    cgltf_node_transform_local(&node, glm::value_ptr(transform));
  }
  return transform;
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

UniqueKtxTexture2 LoadBaseColorTexture(const cgltf_material& material,
                                       const std::filesystem::path& gltf_parent_filepath,
                                       const vk::PhysicalDeviceFeatures& physical_device_features) {
  UniqueKtxTexture2 ktx_texture2{nullptr, nullptr};
  if (material.has_pbr_metallic_roughness != 0) {
    const auto& pbr_metallic_roughness = material.pbr_metallic_roughness;
    const auto& base_color_texture = pbr_metallic_roughness.base_color_texture.texture;

    if (base_color_texture != nullptr) {
      const auto ktx_filepath = gltf_parent_filepath / base_color_texture->image->uri;
      if (const auto result = ktxTexture2_CreateFromNamedFile(ktx_filepath.string().c_str(),
                                                              KTX_TEXTURE_CREATE_CHECK_GLTF_BASISU_BIT,
                                                              std::out_ptr(ktx_texture2, DestroyKtxTexture2));
          result != KTX_SUCCESS) {
        throw std::runtime_error{std::format("Failed to create KTX texture for {} with error {}",
                                             ktx_filepath.string(),
                                             ktxErrorString(result))};
      }

      if (ktxTexture2_NeedsTranscoding(ktx_texture2.get())) {
        const auto transcode_format = GetTranscodeFormat(*ktx_texture2, physical_device_features);
        if (const auto result = ktxTexture2_TranscodeBasis(ktx_texture2.get(), transcode_format, 0);
            result != KTX_SUCCESS) {
          throw std::runtime_error{std::format("Failed to transcode {} to {} with error {}",
                                               ktx_filepath.string(),
                                               ktxTranscodeFormatString(transcode_format),
                                               ktxErrorString(result))};
        }
      }
    }
  }
  return ktx_texture2;
}

vk::Sampler CreateSampler(const vk::Device device,
                          const std::uint32_t mip_levels,
                          const vk::PhysicalDeviceFeatures& physical_device_features,
                          const vk::PhysicalDeviceLimits& physical_device_limits,
                          std::unordered_map<vk::SamplerCreateInfo, vk::UniqueSampler>& samplers) {
  const vk::SamplerCreateInfo sampler_create_info{.magFilter = vk::Filter::eLinear,
                                                  .minFilter = vk::Filter::eLinear,
                                                  .mipmapMode = VULKAN_HPP_NAMESPACE::SamplerMipmapMode::eLinear,
                                                  .anisotropyEnable = physical_device_features.samplerAnisotropy,
                                                  .maxAnisotropy = physical_device_limits.maxSamplerAnisotropy,
                                                  .maxLod = static_cast<float>(mip_levels)};
  auto iterator = samplers.find(sampler_create_info);
  if (iterator == samplers.cend()) {
    std::tie(iterator, std::ignore) =
        samplers.emplace(sampler_create_info, device.createSamplerUnique(sampler_create_info));
  }
  return *iterator->second;
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
          if (const auto result = ktxTexture_GetImageOffset(ktxTexture(&ktx_texture2), mip_level, 0, 0, &buffer_offset);
              result != KTX_SUCCESS) {
            throw std::runtime_error{std::format("Failed to get image offset for mip level {} with error {}",
                                                 mip_level,
                                                 ktxErrorString(result))};
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

template <typename T>
gfx::Buffer CreateBuffer(const std::vector<T>& data,
                         const vk::BufferUsageFlags buffer_usage_flags,
                         const vk::CommandBuffer command_buffer,
                         const VmaAllocator allocator,
                         std::vector<gfx::Buffer>& staging_buffers) {
  const auto size_bytes = sizeof(T) * data.size();

  static constexpr VmaAllocationCreateInfo kStagingBufferAllocationCreateInfo{
      .flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT,
      .usage = VMA_MEMORY_USAGE_AUTO};
  auto& staging_buffer = staging_buffers.emplace_back(size_bytes,
                                                      vk::BufferUsageFlagBits::eTransferSrc,
                                                      allocator,
                                                      kStagingBufferAllocationCreateInfo);
  staging_buffer.template CopyOnce<T>(data);

  static constexpr VmaAllocationCreateInfo kBufferAllocationCreateInfo{.usage = VMA_MEMORY_USAGE_AUTO};
  gfx::Buffer buffer{size_bytes,
                     buffer_usage_flags | vk::BufferUsageFlagBits::eTransferDst,
                     allocator,
                     kBufferAllocationCreateInfo};
  command_buffer.copyBuffer(*staging_buffer, *buffer, vk::BufferCopy{.size = size_bytes});

  return buffer;
}

std::vector<gfx::Mesh> CreateMeshes(const cgltf_mesh& mesh,
                                    const vk::CommandBuffer command_buffer,
                                    const VmaAllocator allocator,
                                    const std::unordered_map<const cgltf_material*, vk::DescriptorSet>& descriptor_sets,
                                    std::vector<gfx::Buffer>& staging_buffers) {
  return std::span{mesh.primitives, mesh.primitives_count}
         | std::views::filter([&mesh, &descriptor_sets](const auto& primitive) {
             if (primitive.type != cgltf_primitive_type_triangles) {
               std::println(std::cerr, "Unsupported mesh primitive {} for {}", primitive.type, GetName(mesh));
               return false;
             }
             return descriptor_sets.contains(primitive.material);  // exclude unsupported primitive material
           })
         | std::views::transform(
             [command_buffer, allocator, &staging_buffers, &descriptor_sets](const auto& primitive) {
               using enum vk::BufferUsageFlagBits;
               const auto vertices = GetVertices(primitive);
               const auto indices = GetIndices(*primitive.indices);
               return gfx::Mesh{CreateBuffer(vertices, eVertexBuffer, command_buffer, allocator, staging_buffers),
                                CreateBuffer(indices, eIndexBuffer, command_buffer, allocator, staging_buffers),
                                static_cast<std::uint32_t>(indices.size()),
                                descriptor_sets.find(primitive.material)->second};
             })
         | std::ranges::to<std::vector>();
}

std::vector<gfx::Mesh> GetMeshes(const cgltf_node& node,
                                 std::unordered_map<const cgltf_mesh*, std::vector<gfx::Mesh>>& meshes) {
  if (node.mesh == nullptr) return {};
  const auto iterator = meshes.find(node.mesh);
  assert(iterator != meshes.cend());
  return std::move(iterator->second);
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

}  // namespace

namespace gfx {

class Model::Node {
public:
  Node(const cgltf_scene& scene, std::unordered_map<const cgltf_mesh*, std::vector<Mesh>>& meshes);
  Node(const cgltf_node& node, std::unordered_map<const cgltf_mesh*, std::vector<Mesh>>& meshes);

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

struct Model::Material {
  const cgltf_material* material = nullptr;
  Image base_color_image;
  vk::Sampler base_color_sampler;
};

Model::Model(const std::filesystem::path& gltf_filepath,
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

  const auto material_count = static_cast<std::uint32_t>(gltf_data->materials_count);
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

  materials_.reserve(material_futures.size());
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

  const auto descriptor_sets =
      std::views::zip_transform(
          [device](const auto& material, auto& descriptor_set) {
            const auto& [gltf_material, base_color_image, base_color_sampler] = material;
            UpdateDescriptorSet(device, base_color_image.image_view(), base_color_sampler, descriptor_set);
            return std::pair{gltf_material, descriptor_set};
          },
          materials_,
          AllocateDescriptorSets(device, *descriptor_pool_, *descriptor_set_layout_, material_count))
      | std::ranges::to<std::unordered_map>();

  auto meshes =
      std::span{gltf_data->meshes, gltf_data->meshes_count}
      | std::views::transform([command_buffer, allocator, &staging_buffers, &descriptor_sets](const auto& mesh) {
          return std::pair{&mesh, CreateMeshes(mesh, command_buffer, allocator, descriptor_sets, staging_buffers)};
        })
      | std::ranges::to<std::unordered_map>();

  command_buffer.end();

  const auto fence = device.createFenceUnique(vk::FenceCreateInfo{});
  queue.submit(vk::SubmitInfo{.commandBufferCount = 1, .pCommandBuffers = &command_buffer}, *fence);

  root_node_ = std::make_unique<const Node>(*gltf_data->scene, meshes);

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

Model::Node::Node(const cgltf_scene& scene, std::unordered_map<const cgltf_mesh*, std::vector<Mesh>>& meshes)
    : children_{std::span{scene.nodes, scene.nodes_count}
                | std::views::transform([&meshes](const auto* const scene_node) {
                    return std::make_unique<const Node>(*scene_node, meshes);
                  })
                | std::ranges::to<std::vector>()} {}

Model::Node::Node(const cgltf_node& node, std::unordered_map<const cgltf_mesh*, std::vector<Mesh>>& meshes)
    : meshes_{GetMeshes(node, meshes)},
      children_{std::span{node.children, node.children_count}
                | std::views::transform([&meshes](const auto* const child_node) {
                    return std::make_unique<const Node>(*child_node, meshes);
                  })
                | std::ranges::to<std::vector>()},
      transform_{GetTransform(node)} {}

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
