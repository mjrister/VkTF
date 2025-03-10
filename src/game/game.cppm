module;

#include <optional>

#include <GLFW/glfw3.h>
#include <glm/glm.hpp>

export module game;

import delta_time;
import camera;
import engine;
import gltf_scene;
import window;

namespace game {

export void Start();

}  // namespace game

module :private;

namespace {

constexpr auto* kProjectName = "VkTF";

vktf::Camera CreateCamera(const float aspect_ratio) {
  static constexpr glm::vec3 kPosition{0.0f, 1.0f, 0.0f};
  static constexpr glm::vec3 kDirection{1.0f, 0.0f, 0.0f};

  return vktf::Camera{kPosition,
                      kDirection,
                      vktf::ViewFrustum{.field_of_view_y = glm::radians(45.0f),
                                        .aspect_ratio = aspect_ratio,
                                        .z_near = 0.1f,
                                        .z_far = 1.0e6f}};
}

void HandleKeyEvents(const vktf::Window& window, vktf::Camera& camera, const vktf::DeltaTime delta_time) {
  if (window.IsKeyPressed(GLFW_KEY_ESCAPE)) {
    window.Close();
    return;
  }

  static constexpr auto kTranslateSpeed = 6.0f;
  const auto translation_step = kTranslateSpeed * delta_time;
  camera.Translate(glm::vec3{translation_step * (window.IsKeyPressed(GLFW_KEY_D) - window.IsKeyPressed(GLFW_KEY_A)),
                             0.0f,
                             translation_step * (window.IsKeyPressed(GLFW_KEY_S) - window.IsKeyPressed(GLFW_KEY_W))});
}

void HandleMouseEvents(const vktf::Window& window,
                       vktf::Camera& camera,
                       std::optional<glm::vec2>& prev_left_click_position) {
  if (!window.IsMouseButtonPressed(GLFW_MOUSE_BUTTON_LEFT)) {
    prev_left_click_position = std::nullopt;
    return;
  }

  const auto left_click_position = window.GetCursorPosition();
  if (prev_left_click_position.has_value()) {
    static constexpr auto kDragSpeed = 0.00390625f;
    const auto drag_direction = kDragSpeed * (left_click_position - *prev_left_click_position);
    camera.Rotate(vktf::EulerAngles{.pitch = -drag_direction.y, .yaw = -drag_direction.x});
  }

  prev_left_click_position = left_click_position;
}

}  // namespace
namespace game {

void Start() {
  const vktf::Window window{kProjectName};
  vktf::Engine engine{window};
  const auto gltf_scene = engine.Load("assets/Main.1_Sponza/NewSponza_Main_glTF_002.gltf");
  auto camera = CreateCamera(window.GetAspectRatio());

  engine.Run(window, [&, prev_left_click_position = std::optional<glm::vec2>{}](const auto delta_time) mutable {
    HandleKeyEvents(window, camera, delta_time);
    HandleMouseEvents(window, camera, prev_left_click_position);
    engine.Render(gltf_scene, camera);
  });
}

}  // namespace game
