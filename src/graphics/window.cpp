#include "graphics/window.h"

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
#ifdef GLFW_INCLUDE_VULKAN
    glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
    glfwWindowHint(GLFW_RESIZABLE, GLFW_FALSE);  // TODO(#50): enable after implementing swapchain recreation
#endif
  }
};

using UniqueGlfwWindow = std::unique_ptr<GLFWwindow, decltype(&glfwDestroyWindow)>;

UniqueGlfwWindow CreateGlfwWindow(const char* const title, int width, int height) {
  [[maybe_unused]] const auto& glfw_context = GlfwContext::Get();

  auto* monitor = glfwGetPrimaryMonitor();
  const auto* video_mode = glfwGetVideoMode(monitor);

  glfwWindowHint(GLFW_RED_BITS, video_mode->redBits);
  glfwWindowHint(GLFW_GREEN_BITS, video_mode->greenBits);
  glfwWindowHint(GLFW_BLUE_BITS, video_mode->blueBits);
  glfwWindowHint(GLFW_REFRESH_RATE, video_mode->refreshRate);

  width = std::min(width, video_mode->width);
  height = std::min(height, video_mode->height);

  auto* window = glfwCreateWindow(width, height, title, nullptr, nullptr);
  if (window == nullptr) throw std::runtime_error{"GLFW window creation failed"};

  const auto center_x = (video_mode->width - width) / 2;
  const auto center_y = (video_mode->height - height) / 2;
  glfwSetWindowPos(window, center_x, center_y);

  return UniqueGlfwWindow{window, glfwDestroyWindow};
}

}  // namespace

namespace gfx {

Window::Window(const char* const title, const int width, const int height)
    : window_{CreateGlfwWindow(title, width, height)} {}

std::pair<int, int> Window::GetSize() const noexcept {
  int width{}, height{};
  glfwGetWindowSize(window_.get(), &width, &height);
  return std::pair{width, height};
}

std::pair<int, int> Window::GetFramebufferSize() const noexcept {
  int width{}, height{};
  glfwGetFramebufferSize(window_.get(), &width, &height);
  return std::pair{width, height};
}

float Window::GetAspectRatio() const noexcept {
  const auto [width, height] = GetSize();
  return height == 0 ? 0.0f : static_cast<float>(width) / static_cast<float>(height);
}

std::pair<float, float> Window::GetCursorPosition() const noexcept {
  double x{}, y{};
  glfwGetCursorPos(window_.get(), &x, &y);
  return std::pair{static_cast<float>(x), static_cast<float>(y)};
}

#ifdef GLFW_INCLUDE_VULKAN

std::span<const char* const> Window::GetInstanceExtensions() {
  std::uint32_t required_extension_count{};
  const auto* const* required_extensions = glfwGetRequiredInstanceExtensions(&required_extension_count);
  if (required_extensions == nullptr) throw std::runtime_error{"No window surface instance extensions"};
  return std::span{required_extensions, required_extension_count};  // pointer lifetime managed by GLFW
}

vk::UniqueSurfaceKHR Window::CreateSurface(const vk::Instance instance) const {
  VkSurfaceKHR surface{};
  const auto result = glfwCreateWindowSurface(instance, window_.get(), nullptr, &surface);
  vk::resultCheck(static_cast<vk::Result>(result), "Window surface creation failed");
  const vk::ObjectDestroy<vk::Instance, VULKAN_HPP_DEFAULT_DISPATCHER_TYPE> deleter{instance};
  return vk::UniqueSurfaceKHR{vk::SurfaceKHR{surface}, deleter};
}

#endif

}  // namespace gfx
