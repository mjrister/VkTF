module;

#include <cstdint>
#include <format>
#include <memory>
#include <stdexcept>
#include <vector>

#include <GLFW/glfw3.h>
#include <glm/glm.hpp>
#include <vulkan/vulkan.hpp>

export module window;

import log;

namespace vktf {

using UniqueGlfwWindow = std::unique_ptr<GLFWwindow, decltype(&glfwDestroyWindow)>;

export class [[nodiscard]] Window {
public:
  explicit Window(const char* title);

  [[nodiscard]] vk::Extent2D GetFramebufferExtent() const noexcept;
  [[nodiscard]] glm::vec2 GetCursorPosition() const noexcept;

  [[nodiscard]] bool IsKeyPressed(const int key) const noexcept {
    return glfwGetKey(glfw_window_.get(), key) == GLFW_PRESS;
  }

  [[nodiscard]] bool IsMouseButtonPressed(const int button) const noexcept {
    return glfwGetMouseButton(glfw_window_.get(), button) == GLFW_PRESS;
  }

  [[nodiscard]] bool IsClosed() const noexcept { return glfwWindowShouldClose(glfw_window_.get()) == GLFW_TRUE; }
  void Close() const noexcept { glfwSetWindowShouldClose(glfw_window_.get(), GLFW_TRUE); }

  static void Update() noexcept { glfwPollEvents(); }

  [[nodiscard]] static std::vector<const char*> GetInstanceExtensions();
  [[nodiscard]] vk::UniqueSurfaceKHR CreateSurface(vk::Instance instance) const;

private:
  UniqueGlfwWindow glfw_window_;
};

}  // namespace vktf

module :private;

namespace vktf {

namespace {

class GlfwContext {
public:
  static const GlfwContext& Instance() {
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
      using Severity = Log::Severity;
      auto& log = Log::Default();
      log(Severity::kError) << std::format("GLFW error {}: {}", error_code, description);
    });
#endif
    if (glfwInit() == GLFW_FALSE) throw std::runtime_error{"GLFW initialization failed"};

    glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);  // disable OpenGL context creation
  }
};

UniqueGlfwWindow CreateGlfwWindow(const char* const title) {
  [[maybe_unused]] const auto& glfw_context = GlfwContext::Instance();

  auto* primary_monitor = glfwGetPrimaryMonitor();
  if (primary_monitor == nullptr) throw std::runtime_error{"Failed to locate the primary monitor"};

  const auto* const video_mode = glfwGetVideoMode(primary_monitor);
  if (video_mode == nullptr) throw std::runtime_error{"Failed to get the primary monitor video mode"};

#ifndef NDEBUG
  primary_monitor = nullptr;  // avoid creating the window in full-screen mode when debugging
#endif

  UniqueGlfwWindow glfw_window{glfwCreateWindow(video_mode->width, video_mode->height, title, primary_monitor, nullptr),
                               glfwDestroyWindow};
  if (glfw_window == nullptr) {
    throw std::runtime_error{"GLFW window creation failed"};
  }
  return glfw_window;
}

}  // namespace

Window::Window(const char* const title) : glfw_window_{CreateGlfwWindow(title)} {}

std::vector<const char*> Window::GetInstanceExtensions() {
  std::uint32_t required_extension_count = 0;
  const auto** required_extensions = glfwGetRequiredInstanceExtensions(&required_extension_count);
  if (required_extensions == nullptr) throw std::runtime_error{"No window surface instance extensions"};
  return std::vector(required_extensions, required_extensions + required_extension_count);
}

vk::UniqueSurfaceKHR Window::CreateSurface(const vk::Instance instance) const {
  VkSurfaceKHR surface = nullptr;
  const auto result = glfwCreateWindowSurface(instance, glfw_window_.get(), nullptr, &surface);
  vk::detail::resultCheck(static_cast<vk::Result>(result), "Window surface creation failed");
  return vk::UniqueSurfaceKHR{surface, instance};
}

vk::Extent2D Window::GetFramebufferExtent() const noexcept {
  auto width = 0;
  auto height = 0;
  glfwGetFramebufferSize(glfw_window_.get(), &width, &height);
  return vk::Extent2D{.width = static_cast<std::uint32_t>(width), .height = static_cast<std::uint32_t>(height)};
}

glm::vec2 Window::GetCursorPosition() const noexcept {
  auto x = 0.0;
  auto y = 0.0;
  glfwGetCursorPos(glfw_window_.get(), &x, &y);
  return glm::vec2{static_cast<float>(x), static_cast<float>(y)};
}

}  // namespace vktf
