module;

#include <algorithm>
#include <array>
#include <concepts>
#include <cstdint>
#include <filesystem>
#include <limits>
#include <optional>
#include <ranges>
#include <utility>
#include <vector>

#include <glm/glm.hpp>
#include <vulkan/vulkan.hpp>

export module scene;

import buffer;
import camera;
import command_pool;
import gltf_asset;
import log;
import mesh;
import model;
import queue;
import shader_module;
import vma_allocator;

namespace vktf {

export class [[nodiscard]] Scene {
public:
  struct [[nodiscard]] CameraTransforms {
    glm::mat4 view_transform{0.0f};
    glm::mat4 projection_transform{0.0f};
  };

  struct [[nodiscard]] WorldLight {
    glm::vec4 position{0.0f};  // represents normalized light direction when w-component is zero
    glm::vec4 color{0.0f};
  };

  struct [[nodiscard]] CreateInfo {
    const std::vector<gltf::Asset>& gltf_assets;
    const Queue& transfer_queue;
    const vk::PhysicalDeviceFeatures& physical_device_features;
    std::optional<float> sampler_anisotropy;
    vk::Extent2D viewport_extent;
    vk::SampleCountFlagBits msaa_sample_count = vk::SampleCountFlagBits::e1;
    vk::RenderPass render_pass;
    vk::DescriptorSetLayout global_descriptor_set_layout;
    Log& log;
  };

  Scene(const vma::Allocator& allocator, const CreateInfo& create_info);

  [[nodiscard]] auto& camera(this auto& self) noexcept { return self.camera_; }
  [[nodiscard]] std::uint32_t light_count() const noexcept { return light_count_; }

  void Update(HostVisibleBuffer& camera_uniform_buffer, HostVisibleBuffer& lights_uniform_buffer);
  void Render(vk::CommandBuffer command_buffer, vk::DescriptorSet global_descriptor_set) const;

private:
  Camera camera_;
  std::uint32_t light_count_;
  std::vector<Model> models_;
  vk::UniqueDescriptorSetLayout material_descriptor_set_layout_;  // TODO: avoid fixed material descriptor set layout
  vk::UniquePipelineLayout graphics_pipeline_layout_;
  vk::UniquePipeline graphics_pipeline_;
};

}  // namespace vktf

module :private;

namespace vktf {

namespace {

// =====================================================================================================================
// Camera
// =====================================================================================================================

float GetAspectRatio(const vk::Extent2D viewport_extent) {
  const auto [width, height] = viewport_extent;
  assert(width > 0 && height > 0);  // assume valid viewport extent
  return height == 0 ? 0.0f : static_cast<float>(width) / static_cast<float>(height);
}

Camera CreateCamera(const vk::Extent2D viewport_extent) {
  static constexpr glm::vec3 kPosition{0.0f, 1.0f, 0.0f};
  static constexpr glm::vec3 kDirection{1.0f, 0.0f, 0.0f};

  return Camera{kPosition,
                kDirection,
                ViewFrustum{.field_of_view_y = glm::radians(45.0f),
                            .aspect_ratio = GetAspectRatio(viewport_extent),
                            .z_near = 0.1f,
                            .z_far = 1.0e6f}};
}

// =====================================================================================================================
// Lights
// =====================================================================================================================

using WorldLight = Scene::WorldLight;

std::uint32_t GetLightCount(const std::vector<gltf::Asset>& gltf_assets) {
  return std::ranges::fold_left(gltf_assets, 0u, [](const auto& light_count, const auto& gltf_asset) {
    return light_count + static_cast<std::uint32_t>(gltf_asset.lights.size());
  });
}

void EmplaceWorldLight(const Model::Node& node, std::vector<WorldLight>& world_lights) {
  const auto& light = node.light;
  if (light == nullptr) return;

  static constexpr auto kAlphaPadding = 1.0f;  // alpha padding applied to conform to std140 layout requirements
  const glm::vec4 light_color{light->color, kAlphaPadding};

  switch (const auto& world_transform = node.global_transform; light->type) {
    using enum Model::Light::Type;
    case kDirectional: {
      const auto& light_direction = world_transform[2];  // node orientation +z-axis
      world_lights.emplace_back(glm::normalize(light_direction), light_color);
      break;
    }
    case kPoint: {
      const auto& light_position = world_transform[3];  // node world-space position
      world_lights.emplace_back(light_position, light_color);
      break;
    }
    default:
      std::unreachable();
  }
}

// =====================================================================================================================
// Models
// =====================================================================================================================

using StagingModelPair = std::pair<const gltf::Asset*, StagingModel>;

std::vector<StagingModelPair> CreateStagingModels(const vma::Allocator& allocator,
                                                  const std::vector<gltf::Asset>& gltf_assets,
                                                  const vk::PhysicalDeviceFeatures& physical_device_features,
                                                  Log& log) {
  return gltf_assets  //
         | std::views::transform([&allocator, &physical_device_features, &log](const auto& gltf_asset) {
             return std::pair{
                 &gltf_asset,
                 StagingModel{allocator,
                              StagingModel::CreateInfo{.gltf_asset = gltf_asset,
                                                       .physical_device_features = physical_device_features,
                                                       .log = log}}};
           })
         | std::ranges::to<std::vector>();
}

std::vector<Model> CreateModels(const vma::Allocator& allocator,
                                const vk::CommandBuffer command_buffer,
                                const std::vector<StagingModelPair>& staging_models,
                                const vk::DescriptorSetLayout material_descriptor_set_layout,
                                const std::optional<float> sampler_anisotropy) {
  return staging_models  //
         | std::views::transform([=, &allocator](const auto& key_value_pair) {
             const auto& [gltf_asset, staging_model] = key_value_pair;
             return Model{allocator,
                          command_buffer,
                          Model::CreateInfo{.gltf_asset = *gltf_asset,
                                            .staging_model = staging_model,
                                            .material_descriptor_set_layout = material_descriptor_set_layout,
                                            .sampler_anisotropy = sampler_anisotropy}};
           })
         | std::ranges::to<std::vector>();
}

// =====================================================================================================================
// Graphics Pipeline Layout
// =====================================================================================================================

struct PushConstants {
  glm::mat4 model_transform{0.0f};
  glm::vec3 view_position{0.0f};
};

vk::UniqueDescriptorSetLayout CreateMaterialDescriptorSetLayout(const vk::Device device) {
  static constexpr std::array kDescriptorSetLayoutBindings{
      vk::DescriptorSetLayoutBinding{.binding = 0,  // properties uniform buffer
                                     .descriptorType = vk::DescriptorType::eUniformBuffer,
                                     .descriptorCount = 1,
                                     .stageFlags = vk::ShaderStageFlagBits::eFragment},
      vk::DescriptorSetLayoutBinding{.binding = 1,  // PBR metallic-roughness textures
                                     .descriptorType = vk::DescriptorType::eCombinedImageSampler,
                                     .descriptorCount = 3,
                                     .stageFlags = vk::ShaderStageFlagBits::eFragment}};

  return device.createDescriptorSetLayoutUnique(
      vk::DescriptorSetLayoutCreateInfo{.bindingCount = static_cast<std::uint32_t>(kDescriptorSetLayoutBindings.size()),
                                        .pBindings = kDescriptorSetLayoutBindings.data()});
}

vk::UniquePipelineLayout CreateGraphicsPipelineLayout(const vk::Device device,
                                                      const vk::DescriptorSetLayout global_descriptor_set_layout,
                                                      const vk::DescriptorSetLayout material_descriptor_set_layout) {
  static constexpr std::array kPushConstantRanges{
      vk::PushConstantRange{.stageFlags = vk::ShaderStageFlagBits::eVertex,
                            .offset = offsetof(PushConstants, model_transform),
                            .size = sizeof(PushConstants::model_transform)},
      vk::PushConstantRange{.stageFlags = vk::ShaderStageFlagBits::eFragment,
                            .offset = offsetof(PushConstants, view_position),
                            .size = sizeof(PushConstants::view_position)}};

  const std::array descriptor_set_layouts{global_descriptor_set_layout, material_descriptor_set_layout};

  return device.createPipelineLayoutUnique(
      vk::PipelineLayoutCreateInfo{.setLayoutCount = static_cast<std::uint32_t>(descriptor_set_layouts.size()),
                                   .pSetLayouts = descriptor_set_layouts.data(),
                                   .pushConstantRangeCount = static_cast<std::uint32_t>(kPushConstantRanges.size()),
                                   .pPushConstantRanges = kPushConstantRanges.data()});
}

// =====================================================================================================================
// Graphics Pipeline
// =====================================================================================================================

template <typename T>
concept VertexAttribute = requires {
  typename T::value_type;
  requires std::same_as<typename T::value_type, float>;

  { T::length() } -> std::same_as<glm::length_t>;
  requires T::length() == std::clamp(T::length(), 1, 4);
};

template <VertexAttribute T>
consteval vk::Format GetVertexAttributeFormat() {
  if constexpr (static constexpr auto kComponentCount = T::length(); kComponentCount == 1) {
    return vk::Format::eR32Sfloat;
  } else if constexpr (kComponentCount == 2) {
    return vk::Format::eR32G32Sfloat;
  } else if constexpr (kComponentCount == 3) {
    return vk::Format::eR32G32B32Sfloat;
  } else {
    static_assert(kComponentCount == 4, "Unsupported vertex attribute format");
    return vk::Format::eR32G32B32A32Sfloat;
  }
}

// TODO: move graphics pipeline creation to a separate class
vk::UniquePipeline CreateGraphicsPipeline(const vk::Device device,
                                          const vk::PipelineLayout graphics_pipeline_layout,
                                          const vk::Extent2D viewport_extent,
                                          const vk::SampleCountFlagBits msaa_sample_count,
                                          const vk::RenderPass render_pass,
                                          const std::uint32_t light_count,
                                          Log& log) {
  const ShaderModule vertex_shader_module{device,
                                          ShaderModule::CreateInfo{.shader_filepath = "shaders/vertex.glsl.spv",
                                                                   .shader_stage = vk::ShaderStageFlagBits::eVertex,
                                                                   .log = log}};

  const ShaderModule fragment_shader_module{device,
                                            ShaderModule::CreateInfo{.shader_filepath = "shaders/fragment.glsl.spv",
                                                                     .shader_stage = vk::ShaderStageFlagBits::eFragment,
                                                                     .log = log}};

  static constexpr auto kLightCountSize = sizeof(light_count);
  static constexpr vk::SpecializationMapEntry kSpecializationMapEntry{.constantID = 0,
                                                                      .offset = 0,
                                                                      .size = kLightCountSize};
  const vk::SpecializationInfo specialization_info{.mapEntryCount = 1,
                                                   .pMapEntries = &kSpecializationMapEntry,
                                                   .dataSize = kLightCountSize,
                                                   .pData = &light_count};

  static constexpr auto* kShaderEntryPointName = "main";
  const std::array shader_stage_create_info{
      vk::PipelineShaderStageCreateInfo{.stage = vk::ShaderStageFlagBits::eVertex,
                                        .module = *vertex_shader_module,
                                        .pName = kShaderEntryPointName},
      vk::PipelineShaderStageCreateInfo{.stage = vk::ShaderStageFlagBits::eFragment,
                                        .module = *fragment_shader_module,
                                        .pName = kShaderEntryPointName,
                                        .pSpecializationInfo = &specialization_info}};

  static constexpr vk::VertexInputBindingDescription kVertexInputBindingDescription{
      .binding = 0,
      .stride = sizeof(Vertex),
      .inputRate = vk::VertexInputRate::eVertex};

  static constexpr std::array kVertexAttributeDescriptions{
      vk::VertexInputAttributeDescription{.location = 0,
                                          .binding = 0,
                                          .format = GetVertexAttributeFormat<decltype(Vertex::position)>(),
                                          .offset = offsetof(Vertex, position)},
      vk::VertexInputAttributeDescription{.location = 1,
                                          .binding = 0,
                                          .format = GetVertexAttributeFormat<decltype(Vertex::normal)>(),
                                          .offset = offsetof(Vertex, normal)},
      vk::VertexInputAttributeDescription{.location = 2,
                                          .binding = 0,
                                          .format = GetVertexAttributeFormat<decltype(Vertex::tangent)>(),
                                          .offset = offsetof(Vertex, tangent)},
      vk::VertexInputAttributeDescription{.location = 3,
                                          .binding = 0,
                                          .format = GetVertexAttributeFormat<decltype(Vertex::texcoord_0)>(),
                                          .offset = offsetof(Vertex, texcoord_0)}};

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

}  // namespace

Scene::Scene(const vma::Allocator& allocator, const CreateInfo& create_info)
    : camera_{CreateCamera(create_info.viewport_extent)},
      light_count_{GetLightCount(create_info.gltf_assets)},
      material_descriptor_set_layout_{CreateMaterialDescriptorSetLayout(allocator.device())},
      graphics_pipeline_layout_{CreateGraphicsPipelineLayout(allocator.device(),
                                                             create_info.global_descriptor_set_layout,
                                                             *material_descriptor_set_layout_)},
      graphics_pipeline_{CreateGraphicsPipeline(allocator.device(),
                                                *graphics_pipeline_layout_,
                                                create_info.viewport_extent,
                                                create_info.msaa_sample_count,
                                                create_info.render_pass,
                                                light_count_,
                                                create_info.log)} {
  const auto& device = allocator.device();
  const auto& [gltf_assets,
               transfer_queue,
               physical_device_features,
               sampler_anisotropy,
               viewport_extent,
               msaa_sample_count,
               render_pass,
               global_descriptor_set_layout,
               log] = create_info;

  static constexpr auto kCommandBufferCount = 1;
  const CommandPool copy_command_pool{
      device,
      CommandPool::CreateInfo{.command_pool_create_flags = vk::CommandPoolCreateFlagBits::eTransient,
                              .queue_family_index = transfer_queue.queue_family_index(),
                              .command_buffer_count = kCommandBufferCount}};

  const auto command_buffer = copy_command_pool.command_buffers().front();
  command_buffer.begin(vk::CommandBufferBeginInfo{.flags = vk::CommandBufferUsageFlagBits::eOneTimeSubmit});

  const auto staging_models = CreateStagingModels(allocator, gltf_assets, physical_device_features, log);
  models_ =
      CreateModels(allocator, command_buffer, staging_models, *material_descriptor_set_layout_, sampler_anisotropy);

  command_buffer.end();

  const auto copy_fence = device.createFenceUnique(vk::FenceCreateInfo{});
  transfer_queue->submit(vk::SubmitInfo{.commandBufferCount = kCommandBufferCount, .pCommandBuffers = &command_buffer},
                         *copy_fence);

  static constexpr auto kMaxTimeout = std::numeric_limits<std::uint64_t>::max();
  const auto result = device.waitForFences(*copy_fence, vk::True, kMaxTimeout);
  vk::detail::resultCheck(result, "Copy fence failed to enter a signaled state");
}

void Scene::Update(HostVisibleBuffer& camera_uniform_buffer, HostVisibleBuffer& lights_uniform_buffer) {
  std::vector<WorldLight> world_lights;
  world_lights.reserve(light_count_);  // TODO: avoid per-frame allocation

  for (const auto node_visitor = [&world_lights](const auto& node) { EmplaceWorldLight(node, world_lights); };
       auto& model : models_) {
    model.Update(node_visitor);
  }

  camera_uniform_buffer.Copy<CameraTransforms>(
      CameraTransforms{.view_transform = camera_.view_transform(),
                       .projection_transform = camera_.projection_transform()});

  assert(light_count_ == world_lights.size());  // ensure all scene lights are accounted for
  lights_uniform_buffer.Copy<WorldLight>(world_lights);
}

void Scene::Render(const vk::CommandBuffer command_buffer, const vk::DescriptorSet global_descriptor_set) const {
  using enum vk::PipelineBindPoint;
  command_buffer.bindPipeline(eGraphics, *graphics_pipeline_);
  command_buffer.bindDescriptorSets(eGraphics, *graphics_pipeline_layout_, 0, global_descriptor_set, nullptr);

  using ViewPosition = decltype(PushConstants::view_position);
  command_buffer.pushConstants<ViewPosition>(*graphics_pipeline_layout_,
                                             vk::ShaderStageFlagBits::eFragment,
                                             offsetof(PushConstants, view_position),
                                             camera_.GetPosition());

  for (const auto& model : models_) {
    model.Render(command_buffer, *graphics_pipeline_layout_);
  }
}

}  // namespace vktf
