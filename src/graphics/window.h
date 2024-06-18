#ifndef GRAPHICS_WINDOW_H_
#define GRAPHICS_WINDOW_H_

#include <memory>
#include <utility>

#include <GLFW/glfw3.h>
#ifdef GLFW_INCLUDE_VULKAN
#include <vulkan/vulkan.hpp>
#endif

namespace gfx {

class Window {
public:
  Window(const char* title, int width, int height);

  [[nodiscard]] std::pair<int, int> GetSize() const noexcept;
  [[nodiscard]] std::pair<int, int> GetFramebufferSize() const noexcept;
  [[nodiscard]] float GetAspectRatio() const noexcept;
  [[nodiscard]] std::pair<float, float> GetCursorPosition() const noexcept;

  [[nodiscard]] bool IsKeyPressed(const int key) const noexcept { return glfwGetKey(window_.get(), key) == GLFW_PRESS; }
  [[nodiscard]] bool IsMouseButtonPressed(const int button) const noexcept {
    return glfwGetMouseButton(window_.get(), button) == GLFW_PRESS;
  }

  [[nodiscard]] bool IsClosed() const noexcept { return glfwWindowShouldClose(window_.get()) == GLFW_TRUE; }
  void Close() const noexcept { glfwSetWindowShouldClose(window_.get(), GLFW_TRUE); }

  static void Update() noexcept { glfwPollEvents(); }

#ifdef GLFW_INCLUDE_VULKAN
  [[nodiscard]] static std::span<const char* const> GetInstanceExtensions();
  [[nodiscard]] vk::UniqueSurfaceKHR CreateSurface(vk::Instance instance) const;
#endif

private:
  std::unique_ptr<GLFWwindow, decltype(&glfwDestroyWindow)> window_{nullptr, nullptr};
};

}  // namespace gfx

#endif  // GRAPHICS_WINDOW_H_
