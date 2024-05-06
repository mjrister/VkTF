#ifndef SRC_GRAPHICS_INCLUDE_GRAPHICS_WINDOW_H_
#define SRC_GRAPHICS_INCLUDE_GRAPHICS_WINDOW_H_

#include <memory>

#include <GLFW/glfw3.h>
#include <glm/vec2.hpp>
#include <vulkan/vulkan.hpp>

namespace gfx {

class Window {
public:
  Window(const char* const title, const int width, const int height);

  [[nodiscard]] vk::Extent2D GetExtent() const noexcept;
  [[nodiscard]] vk::Extent2D GetFramebufferExtent() const noexcept;
  [[nodiscard]] float GetAspectRatio() const noexcept;
  [[nodiscard]] glm::vec2 GetCursorPosition() const noexcept;

  [[nodiscard]] bool IsKeyPressed(const int key) const noexcept { return glfwGetKey(window_.get(), key) == GLFW_PRESS; }
  [[nodiscard]] bool IsMouseButtonPressed(const int button) const noexcept {
    return glfwGetMouseButton(window_.get(), button) == GLFW_PRESS;
  }

  [[nodiscard]] bool IsClosed() const noexcept { return glfwWindowShouldClose(window_.get()) == GLFW_TRUE; }
  void Close() const noexcept { glfwSetWindowShouldClose(window_.get(), GLFW_TRUE); }

  static void Update() noexcept { glfwPollEvents(); }

  [[nodiscard]] static std::span<const char* const> GetInstanceExtensions();
  [[nodiscard]] vk::UniqueSurfaceKHR CreateSurface(const vk::Instance instance) const;

private:
  std::unique_ptr<GLFWwindow, decltype(&glfwDestroyWindow)> window_;
};

}  // namespace gfx

#endif  // SRC_GRAPHICS_INCLUDE_GRAPHICS_WINDOW_H_
