#include "app.h"

#include <GLFW/glfw3.h>
#include <glm/glm.hpp>

namespace {

constexpr auto kWindowWidth = 1920;
constexpr auto kWindowHeight = 1080;

gfx::ArcCamera CreateCamera(const float aspect_ratio) {
  static constexpr glm::vec3 kTarget{0.0f};
  static constexpr glm::vec3 kPosition{0.0f, 0.0f, 4.0f};
  return gfx::ArcCamera{kTarget,
                        kPosition,
                        gfx::ArcCamera::ViewFrustum{.field_of_view_y = glm::radians(45.0f),
                                                    .aspect_ratio = aspect_ratio,
                                                    .z_near = 0.1f,
                                                    .z_far = 10'000.0f}};
}

}  // namespace

gfx::App::App()
    : window_{"VkRender", Window::Extent{.width = kWindowWidth, .height = kWindowHeight}},
      engine_{window_},
      camera_{CreateCamera(window_.GetAspectRatio())},
      model_{engine_.device(), "assets/models/damaged_helmet/DamagedHelmet.gltf"} {
  window_.OnKeyEvent([this](const auto key, const auto action) { HandleKeyEvent(key, action); });
  window_.OnCursorEvent([this](const auto x, const auto y) { HandleCursorEvent(x, y); });
  window_.OnScrollEvent([this](const auto y) { HandleScrollEvent(y); });
}

void gfx::App::Run() {
  while (!window_.IsClosed()) {
    Window::Update();
    engine_.Render(camera_, model_);
  }
  engine_.device()->waitIdle();
}

void gfx::App::HandleKeyEvent(const int key, const int action) const {
  if (key == GLFW_KEY_ESCAPE && action == GLFW_PRESS) {
    window_.Close();
  }
}

void gfx::App::HandleCursorEvent(const float x, const float y) {
  static std::optional<glm::vec2> prev_cursor_position;
  const glm::vec2 cursor_position{x, y};

  if (window_.IsMouseButtonPressed(GLFW_MOUSE_BUTTON_LEFT)) {
    if (prev_cursor_position.has_value()) {
      static constexpr auto kRotationSpeed = 0.0078125f;
      const auto delta_cursor_position = cursor_position - *prev_cursor_position;
      const auto rotation = kRotationSpeed * -delta_cursor_position;
      camera_.Rotate(rotation.x, rotation.y);
    }
    prev_cursor_position = cursor_position;
  } else if (prev_cursor_position.has_value()) {
    prev_cursor_position = std::nullopt;
  }
}

void gfx::App::HandleScrollEvent(const float y) {
  static constexpr auto kZoomSpeed = 0.03125f;
  camera_.Zoom(kZoomSpeed * -y);
}
