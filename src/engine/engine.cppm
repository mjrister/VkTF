module;

#include <algorithm>
#include <array>
#include <cassert>
#include <concepts>
#include <cstdint>
#include <filesystem>
#include <initializer_list>
#include <limits>
#include <optional>
#include <ranges>
#include <vector>

#include <vk_mem_alloc.h>
#include <vulkan/vulkan.hpp>

export module engine;

import buffer;
import camera;
import command_pool;
import delta_time;
import descriptor_pool;
import device;
import gltf_asset;
import image;
import instance;
import log;
import model;
import physical_device;
import queue;
import scene;
import swapchain;
import vma_allocator;
import window;

namespace vktf {

constexpr std::size_t kMaxRenderFrames = 2;

export class [[nodiscard]] Engine {
public:
  explicit Engine(const Window& window);

  [[nodiscard]] std::optional<Scene> Load(const std::span<const std::filesystem::path> asset_filepaths,
                                          Log& log = Log::Default());

  template <std::invocable<DeltaTime> Fn>
  void Run(const Window& window, Fn&& main_loop) const {
    for (DeltaTime delta_time; !window.IsClosed();) {
      delta_time.Update();
      window.Update();
      std::forward<Fn>(main_loop)(delta_time);
    }
    device_->waitIdle();
  }

  void Render(Scene& scene);

private:
  std::size_t current_frame_index_ = 0;
  Instance instance_;
  vk::UniqueSurfaceKHR surface_;
  PhysicalDevice physical_device_;
  Device device_;
  vma::Allocator allocator_;
  Swapchain swapchain_;
  vk::SampleCountFlagBits msaa_sample_count_ = vk::SampleCountFlagBits::e1;
  Image color_attachment_;
  Image depth_attachment_;
  vk::UniqueRenderPass render_pass_;
  std::vector<vk::UniqueFramebuffer> framebuffers_;
  Queue graphics_queue_;
  Queue present_queue_;
  CommandPool render_command_pool_;
  std::array<vk::UniqueFence, kMaxRenderFrames> render_fences_;
  std::array<vk::UniqueSemaphore, kMaxRenderFrames> acquire_next_image_semaphores_;
  std::array<vk::UniqueSemaphore, kMaxRenderFrames> present_image_semaphores_;
  vk::UniqueDescriptorSetLayout global_descriptor_set_layout_;
  DescriptorPool global_descriptor_pool_;  // per-frame descriptor set bindings
  std::vector<HostVisibleBuffer> camera_uniform_buffers_;
  std::vector<HostVisibleBuffer> lights_uniform_buffers_;
};

}  // namespace vktf

module :private;

namespace vktf {

namespace {

constexpr auto kVulkanApiVersion = vk::ApiVersion14;

constexpr std::initializer_list<const char*> kRequiredInstanceLayers{
#ifndef NDEBUG
    "VK_LAYER_KHRONOS_validation"
#endif
};

constexpr std::initializer_list kRequiredDeviceExtension{VK_KHR_SWAPCHAIN_EXTENSION_NAME};

vk::PhysicalDeviceFeatures GetEnabledFeatures(const vk::PhysicalDeviceFeatures& physical_device_features) {
  return vk::PhysicalDeviceFeatures{.samplerAnisotropy = physical_device_features.samplerAnisotropy,
                                    .textureCompressionETC2 = physical_device_features.textureCompressionETC2,
                                    .textureCompressionASTC_LDR = physical_device_features.textureCompressionASTC_LDR,
                                    .textureCompressionBC = physical_device_features.textureCompressionBC};
}

vk::SampleCountFlagBits GetMsaaSampleCount(const vk::PhysicalDeviceLimits& physical_device_limits) {
  const auto color_sample_count_flags = physical_device_limits.framebufferColorSampleCounts;
  const auto depth_sample_count_flags = physical_device_limits.framebufferDepthSampleCounts;
  const auto color_depth_sample_count_flags = color_sample_count_flags & depth_sample_count_flags;

  using enum vk::SampleCountFlagBits;
  for (const auto msaa_sample_count_bit : {e8, e4, e2}) {
    if (msaa_sample_count_bit & color_depth_sample_count_flags) {
      return msaa_sample_count_bit;
    }
  }

  assert(color_depth_sample_count_flags & e1);
  return e1;  // multisample anti-aliasing is not supported on this device
}

vk::Format GetDepthAttachmentFormat(const vk::PhysicalDevice physical_device) {
  // the Vulkan specification requires VK_FORMAT_FEATURE_DEPTH_STENCIL_ATTACHMENT_BIT support for VK_FORMAT_D16_UNORM
  // and at least one of VK_FORMAT_X8_D24_UNORM_PACK32 and VK_FORMAT_D32_SFLOAT
  using enum vk::Format;
  for (const auto depth_attachment_format : {eD32Sfloat, eX8D24UnormPack32}) {
    const auto format_properties = physical_device.getFormatProperties(depth_attachment_format);
    if (format_properties.optimalTilingFeatures & vk::FormatFeatureFlagBits::eDepthStencilAttachment) {
      return depth_attachment_format;
    }
  }

#ifndef NDEBUG
  const auto d16_unorm_format_properties = physical_device.getFormatProperties(eD16Unorm);
  assert(d16_unorm_format_properties.optimalTilingFeatures & vk::FormatFeatureFlagBits::eDepthStencilAttachment);
#endif
  return eD16Unorm;
}

std::optional<float> GetMaxSamplerAnisotropy(const vk::PhysicalDeviceFeatures& physical_device_features,
                                             const vk::PhysicalDeviceLimits& physical_device_limits) {
  if (static constexpr auto kMinSamplerAnisotropy = 1.0f; physical_device_features.samplerAnisotropy == vk::True) {
    const auto max_sampler_anisotropy = physical_device_limits.maxSamplerAnisotropy;
    assert(max_sampler_anisotropy >= kMinSamplerAnisotropy);  // required by the Vulkan specification
    return std::max(kMinSamplerAnisotropy, max_sampler_anisotropy);
  }
  return std::nullopt;
}

vk::UniqueRenderPass CreateRenderPass(const vk::Device device,
                                      const vk::SampleCountFlagBits msaa_sample_count,
                                      const vk::Format color_attachment_format,
                                      const vk::Format depth_attachment_format) {
  const vk::AttachmentDescription color_attachment_description{.format = color_attachment_format,
                                                               .samples = msaa_sample_count,
                                                               .loadOp = vk::AttachmentLoadOp::eClear,
                                                               .storeOp = vk::AttachmentStoreOp::eDontCare,
                                                               .stencilLoadOp = vk::AttachmentLoadOp::eDontCare,
                                                               .stencilStoreOp = vk::AttachmentStoreOp::eDontCare,
                                                               .initialLayout = vk::ImageLayout::eUndefined,
                                                               .finalLayout = vk::ImageLayout::eColorAttachmentOptimal};

  const vk::AttachmentDescription color_resolve_attachment_description{
      .format = color_attachment_format,
      .samples = vk::SampleCountFlagBits::e1,
      .loadOp = vk::AttachmentLoadOp::eClear,
      .storeOp = vk::AttachmentStoreOp::eStore,
      .stencilLoadOp = vk::AttachmentLoadOp::eDontCare,
      .stencilStoreOp = vk::AttachmentStoreOp::eDontCare,
      .initialLayout = vk::ImageLayout::eUndefined,
      .finalLayout = vk::ImageLayout::ePresentSrcKHR};

  const vk::AttachmentDescription depth_attachment_description{
      .format = depth_attachment_format,
      .samples = msaa_sample_count,
      .loadOp = vk::AttachmentLoadOp::eClear,
      .storeOp = vk::AttachmentStoreOp::eDontCare,
      .stencilLoadOp = vk::AttachmentLoadOp::eDontCare,
      .stencilStoreOp = vk::AttachmentStoreOp::eDontCare,
      .initialLayout = vk::ImageLayout::eUndefined,
      .finalLayout = vk::ImageLayout::eDepthStencilAttachmentOptimal};

  const std::array attachment_descriptions{color_attachment_description,
                                           color_resolve_attachment_description,
                                           depth_attachment_description};

  static constexpr vk::AttachmentReference kColorAttachmentReference{
      .attachment = 0,
      .layout = vk::ImageLayout::eColorAttachmentOptimal};

  static constexpr vk::AttachmentReference kColorResolveAttachmentReference{
      .attachment = 1,
      .layout = vk::ImageLayout::eColorAttachmentOptimal};

  static constexpr vk::AttachmentReference kDepthAttachmentReference{
      .attachment = 2,
      .layout = vk::ImageLayout::eDepthStencilAttachmentOptimal};

  static constexpr vk::SubpassDescription kSubpassDescription{.pipelineBindPoint = vk::PipelineBindPoint::eGraphics,
                                                              .colorAttachmentCount = 1,
                                                              .pColorAttachments = &kColorAttachmentReference,
                                                              .pResolveAttachments = &kColorResolveAttachmentReference,
                                                              .pDepthStencilAttachment = &kDepthAttachmentReference};

  static constexpr vk::SubpassDependency kSubpassDependency{
      .srcSubpass = vk::SubpassExternal,
      .dstSubpass = 0,
      .srcStageMask = vk::PipelineStageFlagBits::eColorAttachmentOutput,
      .dstStageMask = vk::PipelineStageFlagBits::eColorAttachmentOutput,
      .srcAccessMask = vk::AccessFlagBits::eNone,
      .dstAccessMask = vk::AccessFlagBits::eColorAttachmentRead | vk::AccessFlagBits::eColorAttachmentWrite};

  return device.createRenderPassUnique(
      vk::RenderPassCreateInfo{.attachmentCount = static_cast<std::uint32_t>(attachment_descriptions.size()),
                               .pAttachments = attachment_descriptions.data(),
                               .subpassCount = 1,
                               .pSubpasses = &kSubpassDescription,
                               .dependencyCount = 1,
                               .pDependencies = &kSubpassDependency});
}

std::vector<vk::UniqueFramebuffer> CreateFramebuffers(const vk::Device device,
                                                      const Swapchain& swapchain,
                                                      const vk::RenderPass render_pass,
                                                      const vk::ImageView color_attachment,
                                                      const vk::ImageView depth_attachment) {
  return swapchain.image_views()
         | std::views::transform([=, image_extent = swapchain.image_extent()](const auto& color_resolve_attachment) {
             const std::array image_attachments{color_attachment, *color_resolve_attachment, depth_attachment};
             return device.createFramebufferUnique(
                 vk::FramebufferCreateInfo{.renderPass = render_pass,
                                           .attachmentCount = static_cast<std::uint32_t>(image_attachments.size()),
                                           .pAttachments = image_attachments.data(),
                                           .width = image_extent.width,
                                           .height = image_extent.height,
                                           .layers = 1});
           })
         | std::ranges::to<std::vector>();
}

std::array<vk::UniqueFence, kMaxRenderFrames> CreateFences(const vk::Device device) {
  std::array<vk::UniqueFence, kMaxRenderFrames> fences;
  std::ranges::generate(fences, [device] {
    // create the fence in a signaled state to avoid waiting on the first frame
    return device.createFenceUnique(vk::FenceCreateInfo{.flags = vk::FenceCreateFlagBits::eSignaled});
  });
  return fences;
}

std::array<vk::UniqueSemaphore, kMaxRenderFrames> CreateSemaphores(const vk::Device device) {
  std::array<vk::UniqueSemaphore, kMaxRenderFrames> semaphores;
  std::ranges::generate(semaphores, [device] { return device.createSemaphoreUnique(vk::SemaphoreCreateInfo{}); });
  return semaphores;
}

std::vector<HostVisibleBuffer> CreateUniformBuffers(const vma::Allocator& allocator,
                                                    const std::size_t buffer_size_bytes) {
  return std::views::iota(0uz, kMaxRenderFrames)
         | std::views::transform([&allocator, buffer_size_bytes]([[maybe_unused]] const auto /*index*/) {
             HostVisibleBuffer uniform_buffer{
                 allocator,
                 HostVisibleBuffer::CreateInfo{.size_bytes = buffer_size_bytes,
                                               .usage_flags = vk::BufferUsageFlagBits::eUniformBuffer}};
             uniform_buffer.MapMemory();  // enable persistent mapping
             return uniform_buffer;
           })
         | std::ranges::to<std::vector>();
}

vk::UniqueDescriptorSetLayout CreateGlobalDescriptorSetLayout(const vk::Device device) {
  using enum vk::ShaderStageFlagBits;

  static constexpr std::array kDescriptorSetLayoutBindings{
      vk::DescriptorSetLayoutBinding{.binding = 0,  // camera uniform buffer
                                     .descriptorType = vk::DescriptorType::eUniformBuffer,
                                     .descriptorCount = 1,
                                     .stageFlags = eVertex | eFragment},
      vk::DescriptorSetLayoutBinding{.binding = 1,  // lights uniform buffer
                                     .descriptorType = vk::DescriptorType::eUniformBuffer,
                                     .descriptorCount = 1,
                                     .stageFlags = eFragment}};

  return device.createDescriptorSetLayoutUnique(
      vk::DescriptorSetLayoutCreateInfo{.bindingCount = static_cast<std::uint32_t>(kDescriptorSetLayoutBindings.size()),
                                        .pBindings = kDescriptorSetLayoutBindings.data()});
}

DescriptorPool CreateGlobalDescriptorPool(const vk::Device device,
                                          const vk::DescriptorSetLayout global_descriptor_set_layout) {
  static constexpr std::uint32_t kUniformBuffersPerRenderFrame = 2;  // camera transforms, world-space lights

  static const std::vector descriptor_pool_sizes{
      vk::DescriptorPoolSize{.type = vk::DescriptorType::eUniformBuffer,
                             .descriptorCount = kUniformBuffersPerRenderFrame * kMaxRenderFrames}};

  return DescriptorPool{device,
                        DescriptorPool::CreateInfo{.descriptor_pool_sizes = descriptor_pool_sizes,
                                                   .descriptor_set_layout = global_descriptor_set_layout,
                                                   .descriptor_set_count = kMaxRenderFrames}};
}

void UpdateGlobalDescriptorSets(const vk::Device device,
                                const std::vector<vk::DescriptorSet>& global_descriptor_sets,
                                const std::vector<HostVisibleBuffer>& camera_uniform_buffers,
                                const std::vector<HostVisibleBuffer>& lights_uniform_buffers) {
  assert(global_descriptor_sets.size() == camera_uniform_buffers.size());
  assert(global_descriptor_sets.size() == lights_uniform_buffers.size());

  std::vector<vk::DescriptorBufferInfo> descriptor_buffer_infos;
  const auto uniform_buffer_count = camera_uniform_buffers.size() + lights_uniform_buffers.size();
  descriptor_buffer_infos.reserve(uniform_buffer_count);

  std::vector<vk::WriteDescriptorSet> descriptor_set_writes;
  descriptor_set_writes.reserve(uniform_buffer_count);

  for (const auto& [descriptor_set, camera_uniform_buffer, lights_uniform_buffer] :
       std::views::zip(global_descriptor_sets, camera_uniform_buffers, lights_uniform_buffers)) {
    const auto& camera_uniform_buffer_info = descriptor_buffer_infos.emplace_back(
        vk::DescriptorBufferInfo{.buffer = *camera_uniform_buffer, .range = vk::WholeSize});

    descriptor_set_writes.push_back(vk::WriteDescriptorSet{.dstSet = descriptor_set,
                                                           .dstBinding = 0,
                                                           .dstArrayElement = 0,
                                                           .descriptorCount = 1,
                                                           .descriptorType = vk::DescriptorType::eUniformBuffer,
                                                           .pBufferInfo = &camera_uniform_buffer_info});

    const auto& lights_uniform_buffer_info = descriptor_buffer_infos.emplace_back(
        vk::DescriptorBufferInfo{.buffer = *lights_uniform_buffer, .range = vk::WholeSize});

    descriptor_set_writes.push_back(vk::WriteDescriptorSet{.dstSet = descriptor_set,
                                                           .dstBinding = 1,
                                                           .dstArrayElement = 0,
                                                           .descriptorCount = 1,
                                                           .descriptorType = vk::DescriptorType::eUniformBuffer,
                                                           .pBufferInfo = &lights_uniform_buffer_info});
  }

  device.updateDescriptorSets(descriptor_set_writes, nullptr);
}

}  // namespace

Engine::Engine(const Window& window)
    : instance_{Instance::CreateInfo{.application_info = vk::ApplicationInfo{.apiVersion = kVulkanApiVersion},
                                     .required_layers = kRequiredInstanceLayers,
                                     .required_extensions = Window::GetInstanceExtensions()}},
      surface_{window.CreateSurface(*instance_)},
      physical_device_{
          *instance_,
          PhysicalDevice::CreateInfo{.surface = *surface_, .required_extensions = kRequiredDeviceExtension}},
      device_{*physical_device_,
              Device::CreateInfo{.queue_families = physical_device_.queue_families(),
                                 .enabled_extensions = kRequiredDeviceExtension,
                                 .enabled_features = GetEnabledFeatures(physical_device_.features())}},
      allocator_{*device_,
                 vma::Allocator::CreateInfo{.instance = *instance_,
                                            .physical_device = *physical_device_,
                                            .vulkan_api_version = kVulkanApiVersion}},
      swapchain_{*device_,
                 Swapchain::CreateInfo{.framebuffer_extent = window.GetFramebufferExtent(),
                                       .surface = *surface_,
                                       .physical_device = *physical_device_,
                                       .queue_families = physical_device_.queue_families()}},
      msaa_sample_count_{GetMsaaSampleCount(physical_device_.limits())},
      color_attachment_{allocator_,
                        Image::CreateInfo{.format = swapchain_.image_format(),
                                          .extent = swapchain_.image_extent(),
                                          .mip_levels = 1,
                                          .sample_count = msaa_sample_count_,
                                          .usage_flags = vk::ImageUsageFlagBits::eColorAttachment
                                                         | vk::ImageUsageFlagBits::eTransientAttachment,
                                          .aspect_mask = vk::ImageAspectFlagBits::eColor,
                                          .allocation_create_info = vma::kDedicatedMemoryAllocationCreateInfo}},
      depth_attachment_{allocator_,
                        Image::CreateInfo{.format = GetDepthAttachmentFormat(*physical_device_),
                                          .extent = swapchain_.image_extent(),
                                          .mip_levels = 1,
                                          .sample_count = msaa_sample_count_,
                                          .usage_flags = vk::ImageUsageFlagBits::eDepthStencilAttachment
                                                         | vk::ImageUsageFlagBits::eTransientAttachment,
                                          .aspect_mask = vk::ImageAspectFlagBits::eDepth,
                                          .allocation_create_info = vma::kDedicatedMemoryAllocationCreateInfo}},
      render_pass_{
          CreateRenderPass(*device_, msaa_sample_count_, color_attachment_.format(), depth_attachment_.format())},
      framebuffers_{CreateFramebuffers(*device_,
                                       swapchain_,
                                       *render_pass_,
                                       color_attachment_.image_view(),
                                       depth_attachment_.image_view())},
      graphics_queue_{
          *device_,
          Queue::CreateInfo{.queue_family = physical_device_.queue_families().graphics_family, .queue_index = 0}},
      present_queue_{
          *device_,
          Queue::CreateInfo{.queue_family = physical_device_.queue_families().present_family, .queue_index = 0}},
      render_command_pool_{
          *device_,
          CommandPool::CreateInfo{.command_pool_create_flags = vk::CommandPoolCreateFlagBits::eResetCommandBuffer,
                                  .queue_family_index = graphics_queue_.queue_family_index(),
                                  .command_buffer_count = static_cast<std::uint32_t>(kMaxRenderFrames)}},
      render_fences_{CreateFences(*device_)},
      acquire_next_image_semaphores_{CreateSemaphores(*device_)},
      present_image_semaphores_{CreateSemaphores(*device_)},
      global_descriptor_set_layout_{CreateGlobalDescriptorSetLayout(*device_)},
      global_descriptor_pool_{CreateGlobalDescriptorPool(*device_, *global_descriptor_set_layout_)} {}

std::optional<Scene> Engine::Load(const std::span<const std::filesystem::path> asset_filepaths, Log& log) {
  using Severity = Log::Severity;

  const auto gltf_assets =
      asset_filepaths  //
      | std::views::filter([&log](const auto& asset_filepath) {
          if (const auto extension = asset_filepath.extension(); extension != ".gltf") {
            log(Severity::kError) << std::format("Failed to load asset {} with unsupported file extension",
                                                 asset_filepath.string());
            return false;  // TODO: add support for loading glTF binary files
          }
          return true;
        })
      | std::views::transform([&log](const auto& gltf_filepath) { return gltf::Load(gltf_filepath, log); })
      | std::ranges::to<std::vector>();

  if (gltf_assets.empty()) {
    log(Severity::kError) << "Failed to create scene with no valid glTF assets";
    return std::nullopt;
  }

  Scene scene{allocator_,
              Scene::CreateInfo{
                  .gltf_assets = gltf_assets,
                  // TODO: use a dedicated transfer queue to copy data to device-local memory
                  .transfer_queue = graphics_queue_,
                  .physical_device_features = physical_device_.features(),
                  .sampler_anisotropy = GetMaxSamplerAnisotropy(physical_device_.features(), physical_device_.limits()),
                  .viewport_extent = swapchain_.image_extent(),
                  .msaa_sample_count = msaa_sample_count_,
                  .render_pass = *render_pass_,
                  .global_descriptor_set_layout = *global_descriptor_set_layout_,
                  .log = log}};

  camera_uniform_buffers_ = CreateUniformBuffers(allocator_, sizeof(Scene::CameraProperties));
  lights_uniform_buffers_ = CreateUniformBuffers(allocator_, sizeof(Scene::WorldLight) * scene.light_count());
  const auto& global_descriptor_sets = global_descriptor_pool_.descriptor_sets();
  UpdateGlobalDescriptorSets(*device_, global_descriptor_sets, camera_uniform_buffers_, lights_uniform_buffers_);

  return scene;
}

void Engine::Render(Scene& scene) {
  assert(current_frame_index_ < kMaxRenderFrames);
  if (++current_frame_index_ == kMaxRenderFrames) current_frame_index_ = 0;

  static constexpr auto kMaxTimeout = std::numeric_limits<std::uint64_t>::max();
  const auto render_fence = *render_fences_[current_frame_index_];
  auto result = device_->waitForFences(render_fence, vk::True, kMaxTimeout);
  vk::detail::resultCheck(result, "Render fence failed to enter a signaled state");
  device_->resetFences(render_fence);

  std::uint32_t image_index = 0;
  const auto acquire_next_image_semaphore = *acquire_next_image_semaphores_[current_frame_index_];
  std::tie(result, image_index) = device_->acquireNextImageKHR(*swapchain_, kMaxTimeout, acquire_next_image_semaphore);
  vk::detail::resultCheck(result, "Acquire next swapchain image failed");

  const auto& command_buffers = render_command_pool_.command_buffers();
  const auto command_buffer = command_buffers[current_frame_index_];
  command_buffer.begin(vk::CommandBufferBeginInfo{.flags = vk::CommandBufferUsageFlagBits::eOneTimeSubmit});

  static constexpr std::array kClearColor{0.0f, 0.0f, 0.0f, 0.0f};
  static constexpr std::array kClearValues{
      vk::ClearValue{.color = vk::ClearColorValue{kClearColor}},
      vk::ClearValue{.color = vk::ClearColorValue{kClearColor}},
      vk::ClearValue{.depthStencil = vk::ClearDepthStencilValue{.depth = 1.0f, .stencil = 0}}};

  command_buffer.beginRenderPass(
      vk::RenderPassBeginInfo{
          .renderPass = *render_pass_,
          .framebuffer = *framebuffers_[image_index],
          .renderArea = vk::Rect2D{.offset = vk::Offset2D{0, 0}, .extent = swapchain_.image_extent()},
          .clearValueCount = static_cast<std::uint32_t>(kClearValues.size()),
          .pClearValues = kClearValues.data()},
      vk::SubpassContents::eInline);

  auto& camera_uniform_buffer = camera_uniform_buffers_[current_frame_index_];
  auto& lights_uniform_buffer = lights_uniform_buffers_[current_frame_index_];
  scene.Update(camera_uniform_buffer, lights_uniform_buffer);

  const auto& global_descriptor_sets = global_descriptor_pool_.descriptor_sets();
  const auto global_descriptor_set = global_descriptor_sets[current_frame_index_];
  scene.Render(command_buffer, global_descriptor_set);

  command_buffer.endRenderPass();
  command_buffer.end();

  static constexpr vk::PipelineStageFlags kPipelineWaitStage = vk::PipelineStageFlagBits::eColorAttachmentOutput;
  const auto present_image_semaphore = *present_image_semaphores_[current_frame_index_];
  graphics_queue_->submit(vk::SubmitInfo{.waitSemaphoreCount = 1,
                                         .pWaitSemaphores = &acquire_next_image_semaphore,
                                         .pWaitDstStageMask = &kPipelineWaitStage,
                                         .commandBufferCount = 1,
                                         .pCommandBuffers = &command_buffer,
                                         .signalSemaphoreCount = 1,
                                         .pSignalSemaphores = &present_image_semaphore},
                          render_fence);

  const auto swapchain = *swapchain_;
  result = present_queue_->presentKHR(vk::PresentInfoKHR{.waitSemaphoreCount = 1,
                                                         .pWaitSemaphores = &present_image_semaphore,
                                                         .swapchainCount = 1,
                                                         .pSwapchains = &swapchain,
                                                         .pImageIndices = &image_index});
  vk::detail::resultCheck(result, "Present swapchain image failed");
}

}  // namespace vktf
