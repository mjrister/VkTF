#include "graphics/window.h"

#include <cassert>
#include <format>
#include <iostream>
#include <stdexcept>

namespace {

void InitializeGlfw(const std::pair<int, int>& opengl_version) {
  if (glfwInit() == GLFW_FALSE) {
    throw std::runtime_error{"GLFW initialization failed"};
  }
#ifndef NDEBUG
  glfwSetErrorCallback([](const int error_code, const char* const description) {
    std::cerr << std::format("GLFW Error ({}): {}\n", error_code, description);
  });
  glfwWindowHint(GLFW_OPENGL_DEBUG_CONTEXT, GLFW_TRUE);
#endif
  const auto [major_version, minor_version] = opengl_version;
  glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, major_version);
  glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, minor_version);
  glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
  glfwWindowHint(GLFW_SAMPLES, 4);
}

}  // namespace

qem::Window::Window(const char* const title,
                    const std::pair<int, int>& window_dimensions,
                    const std::pair<int, int>& opengl_version) {
  InitializeGlfw(opengl_version);

  const auto [width, height] = window_dimensions;
  // NOLINTNEXTLINE(cppcoreguidelines-prefer-member-initializer)
  window_ = glfwCreateWindow(width, height, title, nullptr, nullptr);
  if (window_ == nullptr) throw std::runtime_error{"Window creation failed"};

  glfwSetWindowUserPointer(window_, this);
  glfwMakeContextCurrent(window_);
  glfwSwapInterval(1);

  glfwSetFramebufferSizeCallback(window_, [](GLFWwindow* /*window*/, const int new_width, const int new_height) {
    glViewport(0, 0, new_width, new_height);
  });

  glfwSetKeyCallback(
      window_,
      [](GLFWwindow* const window, const int key, const int /*scancode*/, const int action, const int /*modifiers*/) {
        if (key == GLFW_KEY_ESCAPE && action == GLFW_PRESS) {
          glfwSetWindowShouldClose(window, GLFW_TRUE);
        }
        if (action == GLFW_PRESS) {
          const auto* const self = static_cast<Window*>(glfwGetWindowUserPointer(window));
          assert(self != nullptr);
          if (self->on_key_press_) self->on_key_press_(key);
        }
      });

  glfwSetScrollCallback(window_, [](GLFWwindow* const window, const double x_offset, const double y_offset) {
    const auto* const self = static_cast<Window*>(glfwGetWindowUserPointer(window));
    assert(self != nullptr);
    if (self->on_scroll_) self->on_scroll_(x_offset, y_offset);
  });
}

qem::Window::~Window() noexcept {
  if (window_ != nullptr) {
    glfwDestroyWindow(window_);
  }
  glfwTerminate();
}
