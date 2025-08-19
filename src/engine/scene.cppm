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
import view_frustum;
import vma_allocator;

namespace vktf {

/**
 * @brief A scene consisting of multiple glTF assets.
 * @details This class handles loading multiple glTF assets that combine together to form a cohesive scene and manages
 *          global resources that are common to all models in the scene (e.g., cameras, lights). When constructed, it
 *          handles the creation and submission of command buffers to copy glTF asset resources to device-local memory.
 *          It also provides high-level APIs for updating and rendering the scene on a per-frame basis.
 */
export class [[nodiscard]] Scene {
public:
  /** @brief A structure representing properties for the active camera in the scene. */
  struct [[nodiscard]] CameraProperties {
    /** @brief The view-projection matrix that transforms a world-space vertex position into clip-space coordinates. */
    glm::mat4 view_projection_transform{0.0f};

    /** @brief The world-space position of the camera. */
    glm::vec3 world_position;
  };

  /** @brief A light in world-space. */
  struct [[nodiscard]] WorldLight {
    /**
     * @brief The light position in world-space.
     * @attention This property represents the normalized light direction in world-space when the w-component is zero.
     */
    glm::vec4 position{0.0f};

    /**
     * @brief The RGB light color.
     * @attention The alpha component is padded to conform to std140 layout requirements.
     */
    glm::vec4 color{0.0f};
  };

  /** @brief The parameters for creating a @ref Scene. */
  struct [[nodiscard]] CreateInfo {
    /** @brief The glTF assets to load. */
    const std::vector<gltf::Asset>& gltf_assets;

    /** @brief The queue for submitting command buffers that require transfer capabilities. */
    const Queue& transfer_queue;

    /** @brief The physical device features for determining the transcode target of basis universal KTX textures. */
    const vk::PhysicalDeviceFeatures& physical_device_features;

    /**
     * @brief The anisotropy for sampling textures.
     * @note A value of @c std::nullopt indicates this feature is not enabled.
     */
    std::optional<float> sampler_anisotropy;

    /** @brief The fixed viewport and scissor extent for creating graphics pipelines. */
    vk::Extent2D viewport_extent;

    /** @brief The fixed multisample anti-aliasing (MSAA) sample count for creating graphics pipelines. */
    vk::SampleCountFlagBits msaa_sample_count = vk::SampleCountFlagBits::e1;

    /** @brief The fixed render pass for creating graphics pipelines. */
    vk::RenderPass render_pass;

    /**
     * @brief The fixed descriptor set layout for global scene resources (e.g., cameras, lights).
     * @note Global descriptor sets are frame-dependent and therefore managed by @ref Engine.
     */
    vk::DescriptorSetLayout global_descriptor_set_layout;

    /** @brief The log for writing messages when creating the scene. */
    Log& log;
  };

  /**
   * @brief Creates a @ref Scene.
   * @param allocator The allocator for creating buffers and images.
   * @param create_info @copybrief Scene::CreateInfo.
   */
  Scene(const vma::Allocator& allocator, const CreateInfo& create_info);

  /** @brief Gets the active camera in the scene. */
  [[nodiscard]] auto& camera(this auto& self) noexcept { return self.camera_; }

  /** @brief Gets the number of lights in the scene. */
  [[nodiscard]] std::uint32_t light_count() const noexcept { return light_count_; }

  /**
   * @brief Updates each node in the scene.
   * @details This function traverses the scene graph, updates global transforms for each node in the scene, and copies
   *          global scene data to frame-dependent resources managed by @ref Engine.
   * @param camera_uniform_buffer The camera properties uniform buffer for the current frame.
   * @param lights_uniform_buffer The world-space lights uniform buffer for the current frame.
   */
  void Update(HostVisibleBuffer& camera_uniform_buffer, HostVisibleBuffer& lights_uniform_buffer);

  /**
   * @brief Records draw commands to render models in the scene.
   * @details This function binds compatible graphics pipelines and descriptor sets, traverses the scene graph, and
   *          and records draw commands to render each model in the scene.
   * @param command_buffer The command buffer for recording draw commands.
   * @param global_descriptor_set The global descriptor set to bind for the current frame.
   * @warning The caller is responsible for submitting @p command_buffer to a Vulkan queue to begin execution.
   */
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
                Camera::ViewFrustum{.field_of_view_y = glm::radians(45.0f),
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

  static constexpr auto kAlphaPadding = 1.0f;
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

  camera_uniform_buffer.Copy<CameraProperties>(
      CameraProperties{.view_projection_transform = camera_.projection_transform() * camera_.view_transform(),
                       .world_position = camera_.position()});

  assert(light_count_ == world_lights.size());  // ensure all scene lights are accounted for
  lights_uniform_buffer.Copy<WorldLight>(world_lights);
}

void Scene::Render(const vk::CommandBuffer command_buffer, const vk::DescriptorSet global_descriptor_set) const {
  using enum vk::PipelineBindPoint;
  command_buffer.bindPipeline(eGraphics, *graphics_pipeline_);

  const auto graphics_pipeline_layout = graphics_pipeline_.layout();
  command_buffer.bindDescriptorSets(eGraphics, graphics_pipeline_layout, 0, global_descriptor_set, nullptr);

  for (const ViewFrustum view_frustum{camera_.projection_transform() * camera_.view_transform()};
       const auto& model : models_) {
    model.Render(command_buffer, graphics_pipeline_layout, view_frustum);
  }
}

}  // namespace vktf
