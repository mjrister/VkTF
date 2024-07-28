#include "game/game.h"

#include <optional>

#include <GLFW/glfw3.h>
#include <glm/glm.hpp>

namespace {
constexpr auto kWindowWidth4k = 3840;
constexpr auto kWindowHeight4k = 2160;

gfx::ViewFrustum CreateViewFrustum(const float aspect_ratio) {
  static constexpr auto kFieldOfViewY = glm::radians(45.0f);
  static constexpr auto kZNear = 0.1f;
  static constexpr auto kZFar = 1.0e6f;
  return gfx::ViewFrustum{.field_of_view_y = kFieldOfViewY,
                          .aspect_ratio = aspect_ratio,
                          .z_near = kZNear,
                          .z_far = kZFar};
};

gfx::Camera CreateCamera(const float aspect_ratio) {
  static constexpr glm::vec3 kPosition{0.0f, 1.0f, 0.0f};
  static constexpr glm::vec3 kDirection{1.0f, 0.0f, 0.0f};
  const auto view_frustum = CreateViewFrustum(aspect_ratio);
  return gfx::Camera{kPosition, kDirection, view_frustum};
}

void HandleKeyEvents(const gfx::Window& window, gfx::Camera& camera, const gfx::DeltaTime& delta_time) {
  if (window.IsKeyPressed(GLFW_KEY_ESCAPE)) {
    window.Close();
    return;
  }

  static constexpr auto kTranslationSpeed = 6.0f;
  if (window.IsKeyPressed(GLFW_KEY_W)) camera.Translate(0.0f, 0.0f, -kTranslationSpeed * delta_time);
  if (window.IsKeyPressed(GLFW_KEY_A)) camera.Translate(-kTranslationSpeed * delta_time, 0.0f, 0.0f);
  if (window.IsKeyPressed(GLFW_KEY_S)) camera.Translate(0.0f, 0.0f, kTranslationSpeed * delta_time);
  if (window.IsKeyPressed(GLFW_KEY_D)) camera.Translate(kTranslationSpeed * delta_time, 0.0f, 0.0f);
}

void HandleMouseEvents(const gfx::Window& window, gfx::Camera& camera) {
  if (static std::optional<glm::vec2> maybe_previous_cursor_position;
      window.IsMouseButtonPressed(GLFW_MOUSE_BUTTON_LEFT)) {
    const auto [x, y] = window.GetCursorPosition();
    const glm::vec2 cursor_position{x, y};
    if (maybe_previous_cursor_position.has_value()) {
      static constexpr auto kCursorSpeed = 0.00390625f;
      const auto delta_cursor_position = cursor_position - *maybe_previous_cursor_position;
      const auto rotation = kCursorSpeed * -delta_cursor_position;
      camera.Rotate(rotation.x, rotation.y);
    }
    maybe_previous_cursor_position = cursor_position;
  } else if (maybe_previous_cursor_position.has_value()) {
    maybe_previous_cursor_position = std::nullopt;
  }
}

}  // namespace

namespace gfx {

Game::Game()
    : window_{"VkRender", kWindowWidth4k, kWindowHeight4k},
      engine_{window_},
      camera_{CreateCamera(window_.GetAspectRatio())},
      model_{engine_.LoadModel("assets/models/sponza/Main.1_Sponza/NewSponza_Main_glTF_002.gltf")} {}

void Game::Start() {
  engine_.Run(window_, [this](const auto delta_time) {
    HandleKeyEvents(window_, camera_, delta_time);
    HandleMouseEvents(window_, camera_);
    engine_.Render(camera_, model_);
  });
}

}  // namespace gfx
