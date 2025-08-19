module;

#include <algorithm>
#include <array>
#include <concepts>
#include <cstdint>
#include <utility>

#include <glm/glm.hpp>
#include <vulkan/vulkan.hpp>

export module graphics_pipeline;

import log;
import mesh;
import shader_module;

namespace vktf {

/**
 * @brief An abstraction for a Vulkan graphics pipeline.
 * @details This class handles the creation of a fixed graphics pipeline and corresponding graphics pipeline layout.
 * @note This project does not yet support dynamic graphics pipeline generation to handle variable vertex attribute and
 *       descriptor set layouts. As a result, assets are expected to conform to a fixed pipeline layout to be supported.
 * @see https://registry.khronos.org/vulkan/specs/latest/man/html/VkPipeline.html VkPipeline
 */
export class [[nodiscard]] GraphicsPipeline {
public:
  /** @brief A structure representing shader push constants. */
  struct [[nodiscard]] PushConstants {
    /** @brief The model transform that converts a local-space vertex position into world-space coordinates. */
    glm::mat4 model_transform{0.0f};
  };

  /** @brief The parameters for creating a @ref GraphicsPipeline. */
  struct [[nodiscard]] CreateInfo {
    /** @brief The descriptor set layout for top-level descriptor sets bound once per frame (e.g., camera, lights). */
    vk::DescriptorSetLayout global_descriptor_set_layout;

    /** @brief The descriptor set layout for PBR metallic-roughness materials. */
    vk::DescriptorSetLayout material_descriptor_set_layout;

    /** @brief The viewport and scissor extent.  */
    vk::Extent2D viewport_extent;

    /** @brief The number of samples for multisample anti-aliasing (MSAA). */
    vk::SampleCountFlagBits msaa_sample_count = vk::SampleCountFlagBits::e1;

    /** @brief The render pass for the graphics pipeline. */
    vk::RenderPass render_pass;

    /** @brief The number of lights to use as specialization constant in the fragment shader. */
    std::uint32_t light_count = 0;

    /** @brief The log for writing messages when creating a graphics pipeline. */
    Log& log;
  };

  /**
   * @brief Creates a @ref GraphicsPipeline.
   * @param device The device for creating the graphics pipeline.
   * @param create_info @copybrief GraphicsPipeline::CreateInfo
   */
  GraphicsPipeline(vk::Device device, const CreateInfo& create_info);

  /** @brief Gets the underlying Vulkan pipeline handle. */
  [[nodiscard]] vk::Pipeline operator*() const noexcept { return *pipeline_; }

  /** @brief Gets the underlying Vulkan pipeline layout handle. */
  [[nodiscard]] vk::PipelineLayout layout() const noexcept { return *pipeline_layout_; }

private:
  vk::UniquePipelineLayout pipeline_layout_;
  vk::UniquePipeline pipeline_;
};

}  // namespace vktf

module :private;

namespace vktf {

namespace {

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

vk::UniquePipelineLayout CreateGraphicsPipelineLayout(const vk::Device device,
                                                      const vk::DescriptorSetLayout global_descriptor_set_layout,
                                                      const vk::DescriptorSetLayout material_descriptor_set_layout) {
  using PushConstants = GraphicsPipeline::PushConstants;

  static constexpr std::array kPushConstantRanges{
      vk::PushConstantRange{.stageFlags = vk::ShaderStageFlagBits::eVertex,
                            .offset = offsetof(PushConstants, model_transform),
                            .size = sizeof(PushConstants::model_transform)}};

  const std::array descriptor_set_layouts{global_descriptor_set_layout, material_descriptor_set_layout};

  return device.createPipelineLayoutUnique(
      vk::PipelineLayoutCreateInfo{.setLayoutCount = static_cast<std::uint32_t>(descriptor_set_layouts.size()),
                                   .pSetLayouts = descriptor_set_layouts.data(),
                                   .pushConstantRangeCount = static_cast<std::uint32_t>(kPushConstantRanges.size()),
                                   .pPushConstantRanges = kPushConstantRanges.data()});
}

vk::UniquePipeline CreateGraphicsPipeline(const vk::Device device,
                                          const vk::PipelineLayout graphics_pipeline_layout,
                                          const GraphicsPipeline::CreateInfo& create_info) {
  const auto& [global_descriptor_set_layout,
               material_descriptor_set_layout,
               viewport_extent,
               msaa_sample_count,
               render_pass,
               light_count,
               log] = create_info;

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

GraphicsPipeline::GraphicsPipeline(const vk::Device device, const CreateInfo& create_info)
    : pipeline_layout_{CreateGraphicsPipelineLayout(device,
                                                    create_info.global_descriptor_set_layout,
                                                    create_info.material_descriptor_set_layout)},
      pipeline_{CreateGraphicsPipeline(device, *pipeline_layout_, create_info)} {}

}  // namespace vktf
