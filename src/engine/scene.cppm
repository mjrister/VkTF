module;

#include <algorithm>
#include <array>
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
import graphics_pipeline;
import log;
import model;
import queue;
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
  GraphicsPipeline graphics_pipeline_;
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

}  // namespace

Scene::Scene(const vma::Allocator& allocator, const CreateInfo& create_info)
    : camera_{CreateCamera(create_info.viewport_extent)},
      light_count_{GetLightCount(create_info.gltf_assets)},
      material_descriptor_set_layout_{CreateMaterialDescriptorSetLayout(allocator.device())},
      graphics_pipeline_{
          allocator.device(),
          GraphicsPipeline::CreateInfo{.global_descriptor_set_layout = create_info.global_descriptor_set_layout,
                                       .material_descriptor_set_layout = *material_descriptor_set_layout_,
                                       .viewport_extent = create_info.viewport_extent,
                                       .msaa_sample_count = create_info.msaa_sample_count,
                                       .render_pass = create_info.render_pass,
                                       .light_count = light_count_,
                                       .log = create_info.log}} {
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
      CameraTransforms{.view_transform = camera_.GetViewTransform(),
                       .projection_transform = camera_.GetProjectionTransform()});

  assert(light_count_ == world_lights.size());  // ensure all scene lights are accounted for
  lights_uniform_buffer.Copy<WorldLight>(world_lights);
}

void Scene::Render(const vk::CommandBuffer command_buffer, const vk::DescriptorSet global_descriptor_set) const {
  using enum vk::PipelineBindPoint;
  command_buffer.bindPipeline(eGraphics, *graphics_pipeline_);

  const auto graphics_pipeline_layout = graphics_pipeline_.layout();
  command_buffer.bindDescriptorSets(eGraphics, graphics_pipeline_layout, 0, global_descriptor_set, nullptr);

  using ViewPosition = decltype(GraphicsPipeline::PushConstants::view_position);
  command_buffer.pushConstants<ViewPosition>(graphics_pipeline_layout,
                                             vk::ShaderStageFlagBits::eFragment,
                                             offsetof(GraphicsPipeline::PushConstants, view_position),
                                             camera_.position());

  for (const auto& model : models_) {
    model.Render(command_buffer, graphics_pipeline_layout);
  }
}

}  // namespace vktf
