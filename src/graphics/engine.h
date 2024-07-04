#ifndef GRAPHICS_ENGINE_H_
#define GRAPHICS_ENGINE_H_

#include <array>
#include <concepts>
#include <filesystem>
#include <vector>

#include "graphics/allocator.h"
#include "graphics/delta_time.h"
#include "graphics/device.h"
#include "graphics/image.h"
#include "graphics/instance.h"
#include "graphics/physical_device.h"
#include "graphics/swapchain.h"
#include "graphics/window.h"

namespace gfx {
class Camera;
class Model;

class Engine {
public:
  explicit Engine(const Window& window);

  [[nodiscard]] Model LoadModel(const std::filesystem::path& gltf_filepath) const;

  void Run(const Window& window, std::invocable<DeltaTime> auto&& main_loop) const {
    for (DeltaTime delta_time; !window.ShouldClose();) {
      Window::Update();
      delta_time.Update();
      main_loop(delta_time);
    }
    device_->waitIdle();
  }

  void Render(const Camera& camera, const Model& model);

private:
  static constexpr std::size_t kMaxRenderFrames = 2;
  std::size_t current_frame_index_ = 0;
  Instance instance_;
  vk::UniqueSurfaceKHR surface_;
  PhysicalDevice physical_device_;
  Device device_;
  Allocator allocator_;
  Swapchain swapchain_;
  vk::SampleCountFlagBits msaa_sample_count_;
  Image color_attachment_;
  Image depth_attachment_;
  vk::UniqueRenderPass render_pass_;
  std::vector<vk::UniqueFramebuffer> framebuffers_;
  vk::Queue graphics_queue_;
  vk::Queue present_queue_;
  vk::UniqueCommandPool command_pool_;
  std::vector<vk::UniqueCommandBuffer> command_buffers_;
  std::array<vk::UniqueSemaphore, kMaxRenderFrames> acquire_next_image_semaphores_;
  std::array<vk::UniqueSemaphore, kMaxRenderFrames> present_image_semaphores_;
  std::array<vk::UniqueFence, kMaxRenderFrames> draw_fences_;
};

}  // namespace gfx

#endif  // GRAPHICS_ENGINE_H_
