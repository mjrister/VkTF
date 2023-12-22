#include "app.h"  // NOLINT(build/include_subdir)

#include <GLFW/glfw3.h>
#include <glm/glm.hpp>

#include "graphics/model.h"

namespace {

constexpr auto kWindowWidth = 1920;
constexpr auto kWindowHeight = 1080;

gfx::ArcCamera CreateCamera(const float aspect_ratio) {
  static constexpr glm::vec3 kTarget{0.0f};
  static constexpr glm::vec3 kPosition{0.0f, 0.0f, 1024.0f};
  return gfx::ArcCamera{kTarget,
                        kPosition,
                        gfx::ViewFrustum{.field_of_view_y = glm::radians(45.0f),
                                         .aspect_ratio = aspect_ratio,
                                         .z_near = 0.1f,
                                         .z_far = 10'000.0f}};
}

}  // namespace

gfx::App::App()
    : window_{"VkRender", Window::Extent{.width = kWindowWidth, .height = kWindowHeight}},
      engine_{window_},
      camera_{CreateCamera(window_.GetAspectRatio())},
      model_{engine_.device(), "assets/models/survival_backpack.glb"} {
  window_.OnKeyEvent([this](const auto key, const auto action) { HandleKeyEvent(key, action); });
  window_.OnCursorEvent([this](const auto& x, const auto y) { HandleCursorEvent(x, y); });
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

  if (window_.IsMouseButtonPressed(GLFW_MOUSE_BUTTON_LEFT)) {
    const glm::vec2 cursor_position{x, y};
    if (prev_cursor_position.has_value()) {
      static constexpr auto kRotationSpeed = 0.0078125f;
      const auto delta_cursor_position = cursor_position - *prev_cursor_position;
      const auto rotation = kRotationSpeed * -delta_cursor_position;
      camera_.Rotate(rotation.x, rotation.y);
    }
    prev_cursor_position = cursor_position;
  } else {
    prev_cursor_position = std::nullopt;
  }
}
