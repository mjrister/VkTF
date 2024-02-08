#ifndef SRC_GRAPHICS_INCLUDE_GRAPHICS_WINDOW_H_
#define SRC_GRAPHICS_INCLUDE_GRAPHICS_WINDOW_H_

#include <functional>
#include <memory>
#include <span>
#include <utility>

#include <GLFW/glfw3.h>
#if GLFW_INCLUDE_VULKAN
#include <vulkan/vulkan.hpp>
#endif

namespace gfx {

class Window {
public:
  struct Size {
    int width{};
    int height{};
  };

  Window(const char* title, Size size);

  [[nodiscard]] Size GetSize() const noexcept;
  [[nodiscard]] Size GetFramebufferSize() const noexcept;
  [[nodiscard]] float GetAspectRatio() const noexcept;

  void OnKeyEvent(std::function<void(int, int)> fn) { key_event_handler_ = std::move(fn); }
  void OnCursorEvent(std::function<void(float, float)> fn) { cursor_event_handler_ = std::move(fn); }
  void OnScrollEvent(std::function<void(float)> fn) { scroll_event_handler_ = std::move(fn); }

  [[nodiscard]] bool IsClosed() const noexcept { return glfwWindowShouldClose(window_.get()) == GLFW_TRUE; }
  void Close() const noexcept { glfwSetWindowShouldClose(window_.get(), GLFW_TRUE); }

  [[nodiscard]] bool IsMouseButtonPressed(const int button) const noexcept {
    return glfwGetMouseButton(window_.get(), button) == GLFW_PRESS;
  }

  static void Update() noexcept { glfwPollEvents(); }

#ifdef GLFW_INCLUDE_VULKAN
  [[nodiscard]] static std::span<const char* const> GetInstanceExtensions();
  [[nodiscard]] vk::UniqueSurfaceKHR CreateSurface(vk::Instance instance) const;
#endif

private:
  std::unique_ptr<GLFWwindow, void (*)(GLFWwindow*)> window_{nullptr, nullptr};
  std::function<void(int, int)> key_event_handler_{};
  std::function<void(float, float)> cursor_event_handler_{};
  std::function<void(float)> scroll_event_handler_{};
};

}  // namespace gfx

#endif  // SRC_GRAPHICS_INCLUDE_GRAPHICS_WINDOW_H_"
