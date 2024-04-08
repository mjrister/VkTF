#ifndef SRC_GRAPHICS_INCLUDE_GRAPHICS_ENGINE_H_
#define SRC_GRAPHICS_INCLUDE_GRAPHICS_ENGINE_H_

#include <array>
#include <filesystem>
#include <vector>

#include "graphics/allocator.h"
#include "graphics/device.h"
#include "graphics/image.h"
#include "graphics/instance.h"
#include "graphics/swapchain.h"

namespace gfx {
class Camera;
class Model;
class Window;

class Engine {
public:
  explicit Engine(const Window& window);

  [[nodiscard]] vk::Device device() const noexcept { return *device_; }

  [[nodiscard]] Model LoadModel(const std::filesystem::path& gltf_filepath) const;
  void Render(const Model& model, const Camera& camera);

private:
  static constexpr std::size_t kMaxRenderFrames = 2;
  std::size_t current_frame_index_ = 0;
  Instance instance_;
  vk::UniqueSurfaceKHR surface_;
  Device device_;
  Allocator allocator_;
  Swapchain swapchain_;
  vk::SampleCountFlagBits msaa_sample_count_;
  Image color_attachment_;
  Image depth_attachment_;
  vk::UniqueRenderPass render_pass_;
  std::vector<vk::UniqueFramebuffer> framebuffers_;
  vk::UniqueCommandPool command_pool_;
  std::vector<vk::UniqueCommandBuffer> command_buffers_;
  std::array<vk::UniqueSemaphore, kMaxRenderFrames> acquire_next_image_semaphores_;
  std::array<vk::UniqueSemaphore, kMaxRenderFrames> present_image_semaphores_;
  std::array<vk::UniqueFence, kMaxRenderFrames> draw_fences_;
};

}  // namespace gfx

#endif  // SRC_GRAPHICS_INCLUDE_GRAPHICS_ENGINE_H_
