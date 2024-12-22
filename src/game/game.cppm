module;

#include <optional>

#include <GLFW/glfw3.h>
#include <glm/glm.hpp>

export module game;

import camera;
import engine;
import gltf_scene;
import window;

namespace gfx {

export class Game {
public:
  Game();

  void Start();

private:
  Window window_;
  Engine engine_;
  GltfScene gltf_scene_;
  Camera camera_;
};

}  // namespace gfx

module :private;

namespace {
constexpr auto kWindowWidth4k = 3840;
constexpr auto kWindowHeight4k = 2160;

gfx::Camera CreateCamera(const float aspect_ratio) {
  static constexpr glm::vec3 kPosition{0.0f, 1.0f, 0.0f};
  static constexpr glm::vec3 kDirection{1.0f, 0.0f, 0.0f};
  return gfx::Camera{kPosition,
                     kDirection,
                     gfx::ViewFrustum{.field_of_view_y = glm::radians(45.0f),
                                      .aspect_ratio = aspect_ratio,
                                      .z_near = 0.1f,
                                      .z_far = 1.0e6f}};
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
  static std::optional<glm::vec2> maybe_previous_cursor_position;

  if (window.IsMouseButtonPressed(GLFW_MOUSE_BUTTON_LEFT)) {
    const auto cursor_position = window.GetCursorPosition();

    if (maybe_previous_cursor_position.has_value()) {
      static constexpr auto kRotationSpeed = 0.00390625f;
      const auto delta_cursor_position = cursor_position - *maybe_previous_cursor_position;
      const auto rotation = kRotationSpeed * -delta_cursor_position;
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
      gltf_scene_{engine_.Load("assets/models/Main.1_Sponza/NewSponza_Main_glTF_002.gltf")},
      camera_{CreateCamera(window_.GetAspectRatio())} {}

void Game::Start() {
  engine_.Run(window_, [this](const auto delta_time) {
    HandleKeyEvents(window_, camera_, delta_time);
    HandleMouseEvents(window_, camera_);
    engine_.Render(gltf_scene_, camera_);
  });
}

}  // namespace gfx
