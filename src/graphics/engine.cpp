#include "graphics/engine.h"

#include <algorithm>
#include <cassert>
#include <cstdint>
#include <limits>
#include <ranges>

#include <vk_mem_alloc.h>
#include <vulkan/vulkan.hpp>

#include "graphics/camera.h"
#include "graphics/model.h"
#include "graphics/window.h"

namespace {

vk::SampleCountFlagBits GetMsaaSampleCount(const vk::PhysicalDeviceLimits& physical_device_limits) {
  const auto color_sample_count_flags = physical_device_limits.framebufferColorSampleCounts;
  const auto depth_sample_count_flags = physical_device_limits.framebufferDepthSampleCounts;
  const auto color_depth_sample_count_flags = color_sample_count_flags & depth_sample_count_flags;

  using enum vk::SampleCountFlagBits;
  for (const auto sample_count_flag_bit : {e8, e4, e2}) {
    if (color_depth_sample_count_flags & sample_count_flag_bit) {
      return sample_count_flag_bit;
    }
  }

  assert(color_depth_sample_count_flags & e1);
  return e1;
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
                                                      const gfx::Swapchain& swapchain,
                                                      const vk::RenderPass render_pass,
                                                      const vk::ImageView color_attachment,
                                                      const vk::ImageView depth_attachment) {
  return swapchain.image_views()
         | std::views::transform([=, extent = swapchain.image_extent()](const auto color_resolve_attachment) {
             const std::array image_attachments{color_attachment, color_resolve_attachment, depth_attachment};
             return device.createFramebufferUnique(
                 vk::FramebufferCreateInfo{.renderPass = render_pass,
                                           .attachmentCount = static_cast<std::uint32_t>(image_attachments.size()),
                                           .pAttachments = image_attachments.data(),
                                           .width = extent.width,
                                           .height = extent.height,
                                           .layers = 1});
           })
         | std::ranges::to<std::vector>();
}

vk::UniqueCommandPool CreateCommandPool(const gfx::Device& device) {
  return device->createCommandPoolUnique(
      vk::CommandPoolCreateInfo{.flags = vk::CommandPoolCreateFlagBits::eResetCommandBuffer,
                                .queueFamilyIndex = device.queue_family_indices().graphics_index});
}

template <std::size_t N>
std::vector<vk::UniqueCommandBuffer> AllocateCommandBuffers(const vk::Device device,
                                                            const vk::CommandPool command_pool) {
  return device.allocateCommandBuffersUnique(vk::CommandBufferAllocateInfo{.commandPool = command_pool,
                                                                           .level = vk::CommandBufferLevel::ePrimary,
                                                                           .commandBufferCount = N});
}

template <std::size_t N>
std::array<vk::UniqueSemaphore, N> CreateSemaphores(const vk::Device device) {
  std::array<vk::UniqueSemaphore, N> semaphores;
  std::ranges::generate(semaphores, [device] {  // NOLINT(whitespace/newline)
    return device.createSemaphoreUnique(vk::SemaphoreCreateInfo{});
  });
  return semaphores;
}

template <std::size_t N>
std::array<vk::UniqueFence, N> CreateFences(const vk::Device device) {
  std::array<vk::UniqueFence, N> fences;
  std::ranges::generate(fences, [device] {
    return device.createFenceUnique(vk::FenceCreateInfo{.flags = vk::FenceCreateFlagBits::eSignaled});
  });
  return fences;
}

}  // namespace

namespace gfx {

Engine::Engine(const Window& window)
    : surface_{window.CreateSurface(*instance_)},
      device_{*instance_, *surface_},
      allocator_{*instance_, *device_.physical_device(), *device_},
      swapchain_{*device_,
                 *device_.physical_device(),
                 *surface_,
                 window.GetFramebufferExtent(),
                 device_.queue_family_indices()},
      msaa_sample_count_{GetMsaaSampleCount(device_.physical_device().limits())},
      color_attachment_{*device_,
                        swapchain_.image_format(),
                        swapchain_.image_extent(),
                        1,
                        msaa_sample_count_,
                        vk::ImageUsageFlagBits::eColorAttachment | vk::ImageUsageFlagBits::eTransientAttachment,
                        vk::ImageAspectFlagBits::eColor,
                        *allocator_,
                        VmaAllocationCreateInfo{.flags = VMA_ALLOCATION_CREATE_DEDICATED_MEMORY_BIT,
                                                .usage = VMA_MEMORY_USAGE_AUTO,
                                                .priority = 1.0f}},
      depth_attachment_{*device_,
                        vk::Format::eD24UnormS8Uint,  // TODO(#54): check device support
                        swapchain_.image_extent(),
                        1,
                        msaa_sample_count_,
                        vk::ImageUsageFlagBits::eDepthStencilAttachment | vk::ImageUsageFlagBits::eTransientAttachment,
                        vk::ImageAspectFlagBits::eDepth,
                        *allocator_,
                        VmaAllocationCreateInfo{.flags = VMA_ALLOCATION_CREATE_DEDICATED_MEMORY_BIT,
                                                .usage = VMA_MEMORY_USAGE_AUTO,
                                                .priority = 1.0f}},
      render_pass_{
          CreateRenderPass(*device_, msaa_sample_count_, swapchain_.image_format(), depth_attachment_.format())},
      framebuffers_{CreateFramebuffers(*device_,
                                       swapchain_,
                                       *render_pass_,
                                       color_attachment_.image_view(),
                                       depth_attachment_.image_view())},
      command_pool_{CreateCommandPool(device_)},
      command_buffers_{AllocateCommandBuffers<kMaxRenderFrames>(*device_, *command_pool_)},
      acquire_next_image_semaphores_{CreateSemaphores<kMaxRenderFrames>(*device_)},
      present_image_semaphores_{CreateSemaphores<kMaxRenderFrames>(*device_)},
      draw_fences_{CreateFences<kMaxRenderFrames>(*device_)} {}

Model Engine::LoadModel(const std::filesystem::path& gltf_filepath) const {
  const auto& physical_device = device_.physical_device();
  return Model{gltf_filepath,
               physical_device.features(),
               physical_device.limits(),
               *device_,
               device_.graphics_queue(),  // TODO(matthew-rister): prefer a dedicated transfer queue
               device_.queue_family_indices().graphics_index,
               swapchain_.image_extent(),
               msaa_sample_count_,
               *render_pass_,
               *allocator_};
}

void Engine::Render(const Model& model, const Camera& camera) {
  if (++current_frame_index_ == kMaxRenderFrames) {
    current_frame_index_ = 0;
  }

  // NOLINTBEGIN(cppcoreguidelines-pro-bounds-constant-array-index)
  const auto draw_fence = *draw_fences_[current_frame_index_];
  const auto acquire_next_image_semaphore = *acquire_next_image_semaphores_[current_frame_index_];
  const auto present_image_semaphore = *present_image_semaphores_[current_frame_index_];
  // NOLINTEND(cppcoreguidelines-pro-bounds-constant-array-index)

  static constexpr auto kMaxTimeout = std::numeric_limits<std::uint64_t>::max();
  auto result = device_->waitForFences(draw_fence, vk::True, kMaxTimeout);
  vk::resultCheck(result, "Fence failed to enter a signaled state");
  device_->resetFences(draw_fence);

  std::uint32_t image_index = 0;
  std::tie(result, image_index) = device_->acquireNextImageKHR(*swapchain_, kMaxTimeout, acquire_next_image_semaphore);
  vk::resultCheck(result, "Acquire next swapchain image failed");

  const auto command_buffer = *command_buffers_[current_frame_index_];
  command_buffer.begin(vk::CommandBufferBeginInfo{.flags = vk::CommandBufferUsageFlagBits::eOneTimeSubmit});

  static constexpr std::array kClearColor{0.0f, 0.0f, 0.0f, 1.0f};
  static constexpr std::array kClearValues{vk::ClearValue{.color = vk::ClearColorValue{kClearColor}},
                                           vk::ClearValue{.color = vk::ClearColorValue{kClearColor}},
                                           vk::ClearValue{.depthStencil = vk::ClearDepthStencilValue{1.0f, 0}}};
  command_buffer.beginRenderPass(
      vk::RenderPassBeginInfo{
          .renderPass = *render_pass_,
          .framebuffer = *framebuffers_[image_index],
          .renderArea = vk::Rect2D{.offset = vk::Offset2D{0, 0}, .extent = swapchain_.image_extent()},
          .clearValueCount = static_cast<std::uint32_t>(kClearValues.size()),
          .pClearValues = kClearValues.data()},
      vk::SubpassContents::eInline);

  model.Render(camera, command_buffer);

  command_buffer.endRenderPass();
  command_buffer.end();

  static constexpr vk::PipelineStageFlags kPipelineWaitStage = vk::PipelineStageFlagBits::eColorAttachmentOutput;
  device_.graphics_queue().submit(vk::SubmitInfo{.waitSemaphoreCount = 1,
                                                 .pWaitSemaphores = &acquire_next_image_semaphore,
                                                 .pWaitDstStageMask = &kPipelineWaitStage,
                                                 .commandBufferCount = 1,
                                                 .pCommandBuffers = &command_buffer,
                                                 .signalSemaphoreCount = 1,
                                                 .pSignalSemaphores = &present_image_semaphore},
                                  draw_fence);

  const auto swapchain = *swapchain_;
  result = device_.present_queue().presentKHR(vk::PresentInfoKHR{.waitSemaphoreCount = 1,
                                                                 .pWaitSemaphores = &present_image_semaphore,
                                                                 .swapchainCount = 1,
                                                                 .pSwapchains = &swapchain,
                                                                 .pImageIndices = &image_index});
  vk::resultCheck(result, "Present swapchain image failed");
}

}  // namespace gfx
