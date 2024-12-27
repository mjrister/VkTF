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

void HandleKeyEvents(const gfx::Window& window, gfx::Camera& camera, const gfx::DeltaTime delta_time) {
  if (window.IsKeyPressed(GLFW_KEY_ESCAPE)) {
    window.Close();
    return;
  }

  static constexpr auto kTranslationSpeed = 6.0f;
  const auto translation = kTranslationSpeed * delta_time;
  const auto dx = window.IsKeyPressed(GLFW_KEY_D) * translation - window.IsKeyPressed(GLFW_KEY_A) * translation;
  const auto dz = window.IsKeyPressed(GLFW_KEY_S) * translation - window.IsKeyPressed(GLFW_KEY_W) * translation;

  if (dx != 0.0f || dz != 0.0f) camera.Translate(dx, 0.0f, dz);
}

void HandleMouseEvents(const gfx::Window& window,
                       gfx::Camera& camera,
                       std::optional<glm::vec2>& prev_left_click_position) {
  if (!window.IsMouseButtonPressed(GLFW_MOUSE_BUTTON_LEFT)) {
    prev_left_click_position = std::nullopt;
    return;
  }

  const auto left_click_position = window.GetCursorPosition();
  if (prev_left_click_position.has_value()) {
    static constexpr auto kRotationSpeed = 0.00390625f;
    const auto drag_direction = left_click_position - *prev_left_click_position;
    const auto rotation = kRotationSpeed * glm::vec2{-drag_direction.y, -drag_direction.x};
    camera.Rotate(rotation.x, rotation.y);
  }

  prev_left_click_position = left_click_position;
}

}  // namespace

namespace gfx {

Game::Game()
    : window_{"VkRender"},
      engine_{window_},
      gltf_scene_{engine_.Load("assets/models/Main.1_Sponza/NewSponza_Main_glTF_002.gltf")},
      camera_{CreateCamera(window_.GetAspectRatio())} {}

void Game::Start() {
  engine_.Run(window_, [this, prev_left_click_position = std::optional<glm::vec2>{}](const auto delta_time) mutable {
    HandleKeyEvents(window_, camera_, delta_time);
    HandleMouseEvents(window_, camera_, prev_left_click_position);
    engine_.Render(gltf_scene_, camera_);
  });
}

}  // namespace gfx
