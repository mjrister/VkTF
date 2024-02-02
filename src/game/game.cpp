#include "game/game.h"

#include <GLFW/glfw3.h>
#include <glm/glm.hpp>

namespace {
constexpr auto kWindowWidth = 1920;
constexpr auto kWindowHeight = 1080;

gfx::Camera CreateCamera(const float aspect_ratio) {
  static constexpr glm::vec3 kPosition{0.0f, 1.0f, 0.0f};
  static constexpr glm::vec3 kDirection{1.0f, 0.0f, 0.0f};
  return gfx::Camera{kPosition,
                     kDirection,
                     gfx::Camera::ViewFrustum{.field_of_view_y = glm::radians(45.0f),
                                              .aspect_ratio = aspect_ratio,
                                              .z_near = 0.1f,
                                              .z_far = 10'000.0f}};
}

}  // namespace

gfx::Game::Game()
    : window_{"VkRender", Window::Size{.width = kWindowWidth, .height = kWindowHeight}},
      engine_{window_},
      camera_{CreateCamera(window_.GetAspectRatio())},
      model_{engine_.device(), "assets/models/sponza/Sponza.gltf"} {
  window_.OnKeyEvent([this](const auto key, const auto action) { HandleKeyEvent(key, action); });
  window_.OnCursorEvent([this](const auto x, const auto y) { HandleCursorEvent(x, y); });
  window_.OnScrollEvent([this](const auto y) { HandleScrollEvent(y); });
}

void gfx::Game::Run() {
  while (!window_.IsClosed()) {
    Window::Update();
    engine_.Render(camera_, model_);
  }
  engine_.device()->waitIdle();
}

void gfx::Game::HandleKeyEvent(const int key, const int action) const {
  if (key == GLFW_KEY_ESCAPE && action == GLFW_PRESS) {
    window_.Close();
  }
}

void gfx::Game::HandleCursorEvent(const float x, const float y) {
  static constexpr auto kCursorSpeed = 0.00390625f;
  static std::optional<glm::vec2> previous_cursor_position;
  const glm::vec2 cursor_position{x, y};

  if (window_.IsMouseButtonPressed(GLFW_MOUSE_BUTTON_LEFT)) {
    if (previous_cursor_position.has_value()) {
      const auto delta_cursor_position = cursor_position - *previous_cursor_position;
      const auto rotation = kCursorSpeed * -delta_cursor_position;
      camera_.Rotate(rotation.x, rotation.y);
    }
    previous_cursor_position = cursor_position;
  } else if (window_.IsMouseButtonPressed(GLFW_MOUSE_BUTTON_RIGHT)) {
    if (previous_cursor_position.has_value()) {
      const auto delta_cursor_position = cursor_position - *previous_cursor_position;
      const auto translation = kCursorSpeed * glm::vec2{delta_cursor_position.x, -delta_cursor_position.y};
      camera_.Translate(translation.x, translation.y, 0.0f);
    }
    previous_cursor_position = cursor_position;
  } else if (previous_cursor_position.has_value()) {
    previous_cursor_position = std::nullopt;
  }
}

void gfx::Game::HandleScrollEvent(const float y) {
  static constexpr auto kScrollSpeed = 0.0625f;
  camera_.Translate(0.0f, 0.0f, kScrollSpeed * -y);
}
