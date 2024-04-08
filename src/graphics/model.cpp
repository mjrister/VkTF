#include "graphics/model.h"

#include <array>
#include <cassert>
#include <cstring>
#include <format>
#include <limits>
#include <ranges>
#include <span>
#include <stdexcept>
#include <string_view>
#include <unordered_map>
#include <utility>
#include <vector>

#include <cgltf.h>
#include <glm/glm.hpp>
#include <glm/gtc/type_ptr.hpp>

#include "graphics/buffer.h"
#include "graphics/camera.h"
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

namespace {

struct Vertex {
  glm::vec3 position{0.0f};
  glm::vec3 normal{0.0f};
};

struct PushConstants {
  glm::mat4 model_transform{1.0f};
  glm::mat4 view_projection_transform{1.0f};
};

using UniqueCgltfData = std::unique_ptr<cgltf_data, decltype(&cgltf_free)>;

UniqueCgltfData ParseFile(const char* const gltf_filepath) {
  static constexpr cgltf_options kOptions{};
  UniqueCgltfData data{nullptr, nullptr};

  if (const auto result = cgltf_parse_file(&kOptions, gltf_filepath, std::out_ptr(data, cgltf_free));
      result != cgltf_result_success) {
    throw std::runtime_error{std::format("Parse file failed with with error {}", result)};
  }
  if (const auto result = cgltf_load_buffers(&kOptions, data.get(), gltf_filepath); result != cgltf_result_success) {
    throw std::runtime_error{std::format("Load buffers failed with error {}", result)};
  }
#ifndef NDEBUG
  if (const auto result = cgltf_validate(data.get()); result != cgltf_result_success) {
    throw std::runtime_error{std::format("Validation failed with error {}", result)};
  }
#endif

  return data;
}

template <glm::length_t N>
std::vector<glm::vec<N, float>> UnpackFloats(const cgltf_accessor& accessor) {
  if (const auto components = cgltf_num_components(accessor.type); components != N) {
    const auto* const name = accessor.name != nullptr && std::strlen(accessor.name) > 0 ? accessor.name : "unknown";
    throw std::runtime_error{std::format("Failed to unpack floats for {} with {} components", name, components)};
  }
  std::vector<glm::vec<N, float>> data(accessor.count);
  cgltf_accessor_unpack_floats(&accessor, glm::value_ptr(data.front()), N * accessor.count);
  return data;
}

std::vector<Vertex> GetVertices(const cgltf_primitive& primitive) {
  std::vector<glm::vec3> positions, normals;

  for (const auto& attribute : std::span{primitive.attributes, primitive.attributes_count}) {
    switch (const auto& accessor = *attribute.data; attribute.type) {
      case cgltf_attribute_type_position: {
        positions = UnpackFloats<3>(accessor);
        break;
      }
      case cgltf_attribute_type_normal: {
        normals = UnpackFloats<3>(accessor);
        break;
      }
      default:
        break;
    }
  }

  static constexpr auto kCreateVertex = [](const auto& position, const auto& normal) {
    return Vertex{.position = position, .normal = normal};
  };
  return std::views::zip_transform(kCreateVertex, positions, normals) | std::ranges::to<std::vector>();
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

std::vector<gfx::Mesh> GetSubMeshes(const cgltf_node& node,
                                    std::unordered_map<const cgltf_mesh*, std::vector<gfx::Mesh>>& meshes) {
  if (node.mesh == nullptr) return {};
  const auto iterator = meshes.find(node.mesh);
  assert(iterator != meshes.cend());
  return std::move(iterator->second);
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

std::vector<gfx::Mesh> CreateSubMeshes(const cgltf_mesh& mesh,
                                       const vk::CommandBuffer command_buffer,
                                       const VmaAllocator allocator,
                                       std::vector<gfx::Buffer>& staging_buffers) {
  return std::span{mesh.primitives, mesh.primitives_count}
         | std::views::filter([](const auto& primitive) { return primitive.type == cgltf_primitive_type_triangles; })
         | std::views::transform([command_buffer, allocator, &staging_buffers](const auto& primitive) {
             using enum vk::BufferUsageFlagBits;
             const auto vertices = GetVertices(primitive);
             const auto indices = GetIndices(*primitive.indices);
             return gfx::Mesh{CreateBuffer(vertices, eVertexBuffer, command_buffer, allocator, staging_buffers),
                              CreateBuffer(indices, eIndexBuffer, command_buffer, allocator, staging_buffers),
                              static_cast<std::uint32_t>(indices.size())};
           })
         | std::ranges::to<std::vector>();
}

std::unordered_map<const cgltf_mesh*, std::vector<gfx::Mesh>> CreateMeshes(const cgltf_data& data,
                                                                           const vk::Device device,
                                                                           const vk::Queue queue,
                                                                           const std::uint32_t queue_family_index,
                                                                           const VmaAllocator allocator) {
  const auto command_pool =
      device.createCommandPoolUnique(vk::CommandPoolCreateInfo{.flags = vk::CommandPoolCreateFlagBits::eTransient,
                                                               .queueFamilyIndex = queue_family_index});
  const auto command_buffers =
      device.allocateCommandBuffersUnique(vk::CommandBufferAllocateInfo{.commandPool = *command_pool,
                                                                        .level = vk::CommandBufferLevel::ePrimary,
                                                                        .commandBufferCount = 1});
  const auto command_buffer = *command_buffers.front();
  command_buffer.begin(vk::CommandBufferBeginInfo{.flags = vk::CommandBufferUsageFlagBits::eOneTimeSubmit});

  std::vector<gfx::Buffer> staging_buffers;  // staging buffers must remain in scope until copy commands complete
  auto meshes = std::span{data.meshes, data.meshes_count}
                | std::views::transform([command_buffer, allocator, &staging_buffers](const auto& mesh) {
                    return std::pair{&mesh, CreateSubMeshes(mesh, command_buffer, allocator, staging_buffers)};
                  })
                | std::ranges::to<std::unordered_map>();

  command_buffer.end();

  const auto fence = device.createFenceUnique(vk::FenceCreateInfo{});
  queue.submit(vk::SubmitInfo{.commandBufferCount = 1, .pCommandBuffers = &command_buffer}, *fence);

  static constexpr auto kMaxTimeout = std::numeric_limits<std::uint64_t>::max();
  const auto result = device.waitForFences(*fence, vk::True, kMaxTimeout);
  vk::resultCheck(result, "Fence failed to enter a signaled state");

  return meshes;
}

vk::UniquePipelineLayout CreatePipelineLayout(const vk::Device device) {
  static constexpr vk::PushConstantRange kPushConstantRange{.stageFlags = vk::ShaderStageFlagBits::eVertex,
                                                            .offset = 0,
                                                            .size = sizeof(PushConstants)};
  return device.createPipelineLayoutUnique(
      vk::PipelineLayoutCreateInfo{.pushConstantRangeCount = 1, .pPushConstantRanges = &kPushConstantRange});
}

vk::UniquePipeline CreatePipeline(const vk::Device device,
                                  const vk::Extent2D viewport_extent,
                                  const vk::SampleCountFlagBits msaa_sample_count,
                                  const vk::RenderPass render_pass,
                                  const vk::PipelineLayout pipeline_layout) {
  const std::filesystem::path vertex_shader_filepath{"assets/shaders/mesh.vert"};
  const gfx::ShaderModule vertex_shader_module{vertex_shader_filepath, vk::ShaderStageFlagBits::eVertex, device};

  const std::filesystem::path fragment_shader_filepath{"assets/shaders/mesh.frag"};
  const gfx::ShaderModule fragment_shader_module{fragment_shader_filepath, vk::ShaderStageFlagBits::eFragment, device};

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
                                          .offset = offsetof(Vertex, normal)}};

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

  void Render(vk::CommandBuffer command_buffer, vk::PipelineLayout pipeline_layout, PushConstants push_constants) const;

private:
  std::vector<Mesh> meshes_;
  std::vector<std::unique_ptr<Node>> children_;
  glm::mat4 transform_{1.0f};
};

Model::Model(const std::filesystem::path& gltf_filepath,
             const vk::Device device,
             const vk::Queue transfer_queue,
             const std::uint32_t transfer_queue_family_index,
             const vk::Extent2D viewport_extent,
             const vk::SampleCountFlagBits msaa_sample_count,
             const vk::RenderPass render_pass,
             const VmaAllocator allocator) {
  const auto data = ParseFile(gltf_filepath.string().c_str());
  auto meshes = CreateMeshes(*data, device, transfer_queue, transfer_queue_family_index, allocator);
  root_node_ = std::make_unique<Node>(*data->scene, meshes);
  pipeline_layout_ = CreatePipelineLayout(device);
  pipeline_ = CreatePipeline(device, viewport_extent, msaa_sample_count, render_pass, *pipeline_layout_);
}

Model::~Model() noexcept = default;  // this is necessary to enable forward declaring Model::Node with std::unique_ptr

void Model::Render(const Camera& camera, const vk::CommandBuffer command_buffer) const {
  command_buffer.bindPipeline(vk::PipelineBindPoint::eGraphics, *pipeline_);
  root_node_->Render(
      command_buffer,
      *pipeline_layout_,
      PushConstants{.model_transform = glm::mat4{1.0f},
                    .view_projection_transform = camera.GetProjectionTransform() * camera.GetViewTransform()});
}

Model::Node::Node(const cgltf_scene& scene, std::unordered_map<const cgltf_mesh*, std::vector<Mesh>>& meshes)
    : children_{std::span{scene.nodes, scene.nodes_count}
                | std::views::transform(
                    [&meshes](const auto* const scene_node) { return std::make_unique<Node>(*scene_node, meshes); })
                | std::ranges::to<std::vector>()} {}

Model::Node::Node(const cgltf_node& node, std::unordered_map<const cgltf_mesh*, std::vector<Mesh>>& meshes)
    : meshes_{GetSubMeshes(node, meshes)},
      children_{std::span{node.children, node.children_count}
                | std::views::transform(
                    [&meshes](const auto* const child_node) { return std::make_unique<Node>(*child_node, meshes); })
                | std::ranges::to<std::vector>()},
      transform_{GetTransform(node)} {}

void Model::Node::Render(const vk::CommandBuffer command_buffer,
                         const vk::PipelineLayout pipeline_layout,
                         PushConstants push_constants) const {
  push_constants.model_transform *= transform_;
  command_buffer.pushConstants<PushConstants>(pipeline_layout, vk::ShaderStageFlagBits::eVertex, 0, push_constants);

  for (const auto& mesh : meshes_) {
    mesh.Render(command_buffer);
  }

  for (const auto& child_node : children_) {
    child_node->Render(command_buffer, pipeline_layout, push_constants);
  }
}

}  // namespace gfx
