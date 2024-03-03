#include "game/game.h"

#include <optional>

#include <GLFW/glfw3.h>
#include <glm/glm.hpp>

#include "game/delta_time.h"

namespace {
constexpr auto kWindowWidth = 1920;
constexpr auto kWindowHeight = 1080;

gfx::Camera CreateCamera(const float aspect_ratio) {
  static constexpr glm::vec3 kPosition{0.0f, 1.0f, 0.0f};
  static constexpr glm::vec3 kDirection{1.0f, 0.0f, 0.0f};
  return gfx::Camera{
      kPosition,
      kDirection,
      // NOLINTBEGIN(*-magic-numbers)
      gfx::ViewFrustum{.field_of_view_y = glm::radians(45.0f),
                       .aspect_ratio = aspect_ratio,
                       .z_near = 0.1f,
                       .z_far = 10'000.0f}
      // NOLINTEND(*-magic-numbers)
  };
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
  static std::optional<glm::vec2> previous_cursor_position;
  if (window.IsMouseButtonPressed(GLFW_MOUSE_BUTTON_LEFT)) {
    const auto [x, y] = window.GetCursorPosition();
    const glm::vec2 cursor_position{x, y};
    if (previous_cursor_position.has_value()) {
      static constexpr auto kCursorSpeed = 0.00390625f;
      const auto delta_cursor_position = cursor_position - *previous_cursor_position;
      const auto rotation = kCursorSpeed * -delta_cursor_position;
      camera.Rotate(rotation.x, rotation.y);
    }
    previous_cursor_position = cursor_position;
  } else if (previous_cursor_position.has_value()) {
    previous_cursor_position = std::nullopt;
  }
}

}  // namespace

namespace gfx {

Game::Game()
    : window_{"VkRender", kWindowWidth, kWindowHeight},
      engine_{window_},
      camera_{CreateCamera(window_.GetAspectRatio())},
      model_{engine_.device(), engine_.allocator(), "assets/models/sponza/Main.1_Sponza/NewSponza_Main_glTF_002.gltf"} {
}

void Game::Run() {
  for (DeltaTime delta_time; !window_.IsClosed();) {
    delta_time.Update();
    Window::Update();
    HandleKeyEvents(window_, camera_, delta_time);
    HandleMouseEvents(window_, camera_);
    engine_.Render(camera_, model_);
  }
  engine_.device()->waitIdle();
}

}  // namespace gfx
