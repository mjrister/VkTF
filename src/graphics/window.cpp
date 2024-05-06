#include "graphics/window.h"

#include <cassert>
#include <cstdint>
#include <format>
#include <iostream>
#include <print>
#include <stdexcept>

namespace {

class GlfwContext {
public:
  static const GlfwContext& Get() {
    static const GlfwContext kInstance;
    return kInstance;
  }

  GlfwContext(const GlfwContext&) = delete;
  GlfwContext(GlfwContext&&) noexcept = delete;

  GlfwContext& operator=(const GlfwContext&) = delete;
  GlfwContext& operator=(GlfwContext&&) noexcept = delete;

  ~GlfwContext() noexcept { glfwTerminate(); }

private:
  GlfwContext() {
#ifndef NDEBUG
    glfwSetErrorCallback([](const int error_code, const char* const description) {
      std::println(std::cerr, "GLFW error {}: {}", error_code, description);
    });
#endif
    if (glfwInit() == GLFW_FALSE) {
      throw std::runtime_error{"GLFW initialization failed"};
    }
    glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
    glfwWindowHint(GLFW_RESIZABLE, GLFW_FALSE);  // TODO(#50): enable after implementing swapchain recreation
  }
};

using UniqueGlfwWindow = std::unique_ptr<GLFWwindow, decltype(&glfwDestroyWindow)>;

UniqueGlfwWindow CreateGlfwWindow(const char* const title, int width, int height) {
  [[maybe_unused]] const auto& glfw_context = GlfwContext::Get();

  auto* const monitor = glfwGetPrimaryMonitor();
  const auto* const video_mode = glfwGetVideoMode(monitor);

  glfwWindowHint(GLFW_RED_BITS, video_mode->redBits);
  glfwWindowHint(GLFW_GREEN_BITS, video_mode->greenBits);
  glfwWindowHint(GLFW_BLUE_BITS, video_mode->blueBits);
  glfwWindowHint(GLFW_REFRESH_RATE, video_mode->refreshRate);

  assert(width > 0);
  assert(height > 0);
  width = std::min(width, video_mode->width);
  height = std::min(height, video_mode->height);

  UniqueGlfwWindow window{glfwCreateWindow(width, height, title, nullptr, nullptr), glfwDestroyWindow};
  if (window == nullptr) throw std::runtime_error{"GLFW window creation failed"};

  const auto center_x = (video_mode->width - width) / 2;
  const auto center_y = (video_mode->height - height) / 2;
  glfwSetWindowPos(window.get(), center_x, center_y);

  return window;
}

}  // namespace

namespace gfx {

Window::Window(const char* const title, const int width, const int height)
    : window_{CreateGlfwWindow(title, width, height)} {}

vk::Extent2D Window::GetExtent() const noexcept {
  int width = 0;
  int height = 0;
  glfwGetWindowSize(window_.get(), &width, &height);
  return vk::Extent2D{.width = static_cast<std::uint32_t>(width), .height = static_cast<std::uint32_t>(height)};
}

vk::Extent2D Window::GetFramebufferExtent() const noexcept {
  int width = 0;
  int height = 0;
  glfwGetFramebufferSize(window_.get(), &width, &height);
  return vk::Extent2D{.width = static_cast<std::uint32_t>(width), .height = static_cast<std::uint32_t>(height)};
}

float Window::GetAspectRatio() const noexcept {
  const auto [width, height] = GetExtent();
  return height == 0 ? 0.0f : static_cast<float>(width) / static_cast<float>(height);
}

glm::vec2 Window::GetCursorPosition() const noexcept {
  double x = 0.0;
  double y = 0.0;
  glfwGetCursorPos(window_.get(), &x, &y);
  return glm::vec2{static_cast<float>(x), static_cast<float>(y)};
}

std::span<const char* const> Window::GetInstanceExtensions() {
  std::uint32_t required_extension_count = 0;
  const auto* const* required_extensions = glfwGetRequiredInstanceExtensions(&required_extension_count);
  if (required_extensions == nullptr) throw std::runtime_error{"No window surface instance extensions"};
  return std::span{required_extensions, required_extension_count};  // pointer lifetime managed by GLFW
}

vk::UniqueSurfaceKHR Window::CreateSurface(const vk::Instance instance) const {
  VkSurfaceKHR surface = nullptr;
  const auto result = glfwCreateWindowSurface(instance, window_.get(), nullptr, &surface);
  vk::resultCheck(static_cast<vk::Result>(result), "Window surface creation failed");
  const vk::ObjectDestroy<vk::Instance, VULKAN_HPP_DEFAULT_DISPATCHER_TYPE> deleter{instance};
  return vk::UniqueSurfaceKHR{vk::SurfaceKHR{surface}, deleter};
}

}  // namespace gfx
