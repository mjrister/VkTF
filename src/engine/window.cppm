module;

#include <concepts>
#include <cstdint>
#include <format>
#include <functional>
#include <memory>
#include <stdexcept>
#include <utility>
#include <vector>

#include <GLFW/glfw3.h>
#include <glm/glm.hpp>
#include <vulkan/vulkan.hpp>

export module window;

import log;

namespace vktf {

using UniqueGlfwWindow = std::unique_ptr<GLFWwindow, decltype(&glfwDestroyWindow)>;

/**
 * @brief An abstraction for a GLFW window.
 * @see https://www.glfw.org/documentation GLFW Documentation
 */
export class [[nodiscard]] Window {
public:
  /**
   * @brief Creates a @ref Window.
   * @param title The UTF-8 window title.
   * @throws std::runtime_error Thrown if GLFW initialization fails.
   */
  explicit Window(const char* title);

  /**
   * @brief Registers an event listener to handle keyboard input events.
   * @tparam KeyEventListener The callable function type for @p key_event_listener that accepts a GLFW key
   *                          (e.g., GLFW_KEY_ESCAPE) and action (e.g., GLFW_PRESS) as arguments.
   * @param key_event_listener The key event listener to register.
   */
  template <std::invocable<int, int> KeyEventListener>
  void AddKeyEventListener(KeyEventListener&& key_event_listener) {
    key_event_listeners_.emplace_back(std::forward<KeyEventListener>(key_event_listener));
  }

  /**
   * @brief Gets the size of the framebuffer in pixels.
   * @return An extent containing the width and height of the framebuffer.
   */
  [[nodiscard]] vk::Extent2D GetFramebufferExtent() const noexcept;

  /**
   * @brief Gets the cursor position in screen coordinates relative to the top-left window corner.
   * @return A 2D vector representing the (x,y) cursor position.
   */
  [[nodiscard]] glm::vec2 GetCursorPosition() const noexcept;

  /**
   * @brief Checks if a key is currently pressed.
   * @param key The GLFW key to check (e.g., @c GLFW_KEY_ESCAPE).
   * @return @c true if @ref key is pressed, otherwise @c false.
   */
  [[nodiscard]] bool IsKeyPressed(const int key) const noexcept {
    return glfwGetKey(glfw_window_.get(), key) == GLFW_PRESS;
  }

  /**
   * @brief Checks if a mouse button is currently pressed.
   * @param button The GLFW mouse button to check (e.g., @c GLFW_MOUSE_BUTTON_LEFT).
   * @return @c true if @ref button is pressed, otherwise @c false.
   */
  [[nodiscard]] bool IsMouseButtonPressed(const int button) const noexcept {
    return glfwGetMouseButton(glfw_window_.get(), button) == GLFW_PRESS;
  }

  /**
   * @brief Checks if the window close flag has been set.
   * @return @c true if the window close flag has been set, otherwise @c false.
   */
  [[nodiscard]] bool IsClosed() const noexcept { return glfwWindowShouldClose(glfw_window_.get()) == GLFW_TRUE; }

  /** @brief Sets the window close flag. */
  void Close() const noexcept { glfwSetWindowShouldClose(glfw_window_.get(), GLFW_TRUE); }

  /** @brief Polls window events and processes registered callbacks. */
  static void Update() noexcept { glfwPollEvents(); }

  /**
   * @brief Gets the Vulkan instance extensions required for creating a window surface.
   * @return A list of C-strings for required instance extensions whose lifetimes are managed by GLFW.
   * @throws std::runtime_error Thrown if no window instance extensions are found.
   */
  [[nodiscard]] static std::vector<const char*> GetRequiredInstanceExtensions();

  /**
   * @brief Creates a Vulkan surface.
   * @param instance The instance for creating the Vulkan surface.
   * @return A unique smart handle for a Vulkan surface.
   */
  [[nodiscard]] vk::UniqueSurfaceKHR CreateSurface(vk::Instance instance) const;

private:
  using KeyEventListener = std::function<void(int key, int action)>;

  UniqueGlfwWindow glfw_window_;
  std::vector<KeyEventListener> key_event_listeners_;
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

Window::Window(const char* const title) : glfw_window_{CreateGlfwWindow(title)} {
  glfwSetWindowUserPointer(glfw_window_.get(), this);

  static constexpr auto kGlfwKeyCallback =
      [](auto* glfw_window, const auto key, const auto /*scancode*/, const auto action, const auto /*modifiers*/) {
        const auto* const window = static_cast<const Window*>(glfwGetWindowUserPointer(glfw_window));
        assert(window != nullptr);  // the window user pointer is guaranteed to be set

        for (const auto& key_event_handler : window->key_event_listeners_) {
          key_event_handler(key, action);
        }
      };
  glfwSetKeyCallback(glfw_window_.get(), kGlfwKeyCallback);
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

std::vector<const char*> Window::GetRequiredInstanceExtensions() {
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

}  // namespace vktf
