module;

#include <algorithm>
#include <cassert>
#include <concepts>
#include <cstdint>
#include <format>
#include <functional>
#include <iostream>
#include <memory>
#include <print>
#include <stdexcept>

#include <GLFW/glfw3.h>
#include <glm/glm.hpp>
#include <vulkan/vulkan.hpp>

export module window;

namespace gfx {

export class Window {
public:
  explicit Window(const char* title);

  [[nodiscard]] static std::span<const char* const> GetInstanceExtensions();
  [[nodiscard]] vk::UniqueSurfaceKHR CreateSurface(vk::Instance instance) const;

  void OnResize(std::invocable<vk::Extent2D> auto&& on_resize_fn) noexcept {
    on_resize_ = std::forward<decltype(on_resize_fn)>(on_resize_fn);
  }

  [[nodiscard]] vk::Extent2D GetFramebufferExtent() const noexcept;
  [[nodiscard]] float GetAspectRatio() const noexcept;
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

private:
  using UniqueGlfwWindow = std::unique_ptr<GLFWwindow, decltype(&glfwDestroyWindow)>;

  UniqueGlfwWindow glfw_window_{nullptr, nullptr};
  std::function<void(vk::Extent2D)> on_resize_;
};

}  // namespace gfx

module :private;

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
    if (glfwInit() == GLFW_FALSE) throw std::runtime_error{"GLFW initialization failed"};

    glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);  // disable OpenGL context creation
  }
};

}  // namespace

namespace gfx {

Window::Window(const char* const title) {
  [[maybe_unused]] const auto& glfw_context = GlfwContext::Get();

  auto* const primary_monitor = glfwGetPrimaryMonitor();
  if (primary_monitor == nullptr) throw std::runtime_error{"Failed to locate the primary monitor"};

  const auto* const video_mode = glfwGetVideoMode(primary_monitor);
  if (video_mode == nullptr) throw std::runtime_error{"Failed to get the primary monitor video mode"};

  glfw_window_ = UniqueGlfwWindow{glfwCreateWindow(video_mode->width, video_mode->height, title, nullptr, nullptr),
                                  glfwDestroyWindow};

  if (glfw_window_ == nullptr) throw std::runtime_error{"GLFW window creation failed"};

  glfwSetWindowUserPointer(glfw_window_.get(), this);
  glfwSetFramebufferSizeCallback(glfw_window_.get(),
                                 [](GLFWwindow* const glfw_window, const int width, const int height) {
                                   auto* const window = static_cast<Window*>(glfwGetWindowUserPointer(glfw_window));
                                   assert(window != nullptr);
                                   if (window->on_resize_) {
                                     window->on_resize_(vk::Extent2D{.width = static_cast<std::uint32_t>(width),
                                                                     .height = static_cast<std::uint32_t>(height)});
                                   }
                                 });
}

std::span<const char* const> Window::GetInstanceExtensions() {
  std::uint32_t required_extension_count = 0;
  const auto* const* required_extensions = glfwGetRequiredInstanceExtensions(&required_extension_count);
  if (required_extensions == nullptr) throw std::runtime_error{"No window surface instance extensions"};
  return std::span{required_extensions, required_extension_count};  // pointer lifetime is managed by GLFW
}

vk::UniqueSurfaceKHR Window::CreateSurface(const vk::Instance instance) const {
  VkSurfaceKHR surface = nullptr;
  const auto result = glfwCreateWindowSurface(instance, glfw_window_.get(), nullptr, &surface);
  vk::detail::resultCheck(static_cast<vk::Result>(result), "Window surface creation failed");
  const vk::ObjectDestroy<vk::Instance, VULKAN_HPP_DEFAULT_DISPATCHER_TYPE> deleter{instance};
  return vk::UniqueSurfaceKHR{vk::SurfaceKHR{surface}, deleter};
}

vk::Extent2D Window::GetFramebufferExtent() const noexcept {
  auto width = 0;
  auto height = 0;
  glfwGetFramebufferSize(glfw_window_.get(), &width, &height);
  return vk::Extent2D{.width = static_cast<std::uint32_t>(width), .height = static_cast<std::uint32_t>(height)};
}

float Window::GetAspectRatio() const noexcept {
  const auto [width, height] = GetFramebufferExtent();
  return height == 0 ? 0.0f : static_cast<float>(width) / static_cast<float>(height);
}

glm::vec2 Window::GetCursorPosition() const noexcept {
  auto x = 0.0;
  auto y = 0.0;
  glfwGetCursorPos(glfw_window_.get(), &x, &y);
  return glm::vec2{static_cast<float>(x), static_cast<float>(y)};
}

}  // namespace gfx
