module;

#include <array>
#include <filesystem>
#include <optional>

#include <GLFW/glfw3.h>
#include <glm/glm.hpp>

export module game;

import delta_time;
import camera;
import engine;
import scene;
import window;

namespace game {

/**
 * @brief Starts the application.
 * @details This application is a narrowly scoped Vulkan glTF Renderer (VkTF) for assets that support the PBR metallic-
 *          roughness workflow. It implements simple camera controls using mouse movement to orient the viewer and WASD
 *          keys to move through the scene. Press ESC to exit the application.
 * @note This file is primarily intended to demonstrate how to use core Engine APIs to load and render a scene composed
 *       of multiple glTF files.
 */
export void Start();  // TODO: Add support for loading arbitrary glTF files using command line arguments

}  // namespace game

module :private;

namespace {

vktf::Window CreateWindow() {
  static constexpr auto* kProjectTitle = "VkTF";
  vktf::Window window{kProjectTitle};

  window.AddKeyEventListener([&window](const auto key, const auto action) {
    switch (key) {
      case GLFW_KEY_ESCAPE:
        if (action == GLFW_PRESS) {
          window.Close();
        }
        break;
      default:
        break;
    }
  });

  return window;
}

void HandleKeyEvents(const vktf::Window& window, vktf::Camera& camera, const vktf::DeltaTime delta_time) {
  static constexpr auto kTranslateSpeed = 6.0f;
  const auto translation_step = kTranslateSpeed * delta_time.get();
  camera.Translate(glm::vec3{translation_step * (window.IsKeyPressed(GLFW_KEY_D) - window.IsKeyPressed(GLFW_KEY_A)),
                             0.0f,
                             translation_step * (window.IsKeyPressed(GLFW_KEY_S) - window.IsKeyPressed(GLFW_KEY_W))});
}

void HandleMouseEvents(const vktf::Window& window, vktf::Camera& camera) {
  static std::optional<glm::vec2> prev_left_click_position;
  if (!window.IsMouseButtonPressed(GLFW_MOUSE_BUTTON_LEFT)) {
    prev_left_click_position = std::nullopt;
    return;
  }

  const auto left_click_position = window.GetCursorPosition();
  if (prev_left_click_position.has_value()) {
    static constexpr auto kDragSpeed = 0.00390625f;
    const auto drag_direction = kDragSpeed * (left_click_position - *prev_left_click_position);
    camera.Rotate(-drag_direction.y, -drag_direction.x);
  }

  prev_left_click_position = left_click_position;
}

vktf::Scene LoadScene(vktf::Engine& engine) {
  const std::array asset_filepaths{std::filesystem::path{"assets/Main.1_Sponza/NewSponza_Main_glTF_002.gltf"},
                                   std::filesystem::path{"assets/PKG_A_Curtains/NewSponza_Curtains_glTF.gltf"},
                                   std::filesystem::path{"assets/PKG_B_Ivy/NewSponza_IvyGrowth_glTF.gltf"}};

  auto scene = engine.Load(asset_filepaths);
  assert(scene.has_value());  // default assets are guaranteed to be valid glTF files
  return std::move(*scene);
}

}  // namespace

namespace game {

void Start() {
  const auto window = CreateWindow();
  vktf::Engine engine{window};

  engine.Run(window, [&window, &engine, scene = LoadScene(engine)](const auto delta_time) mutable {
    auto& camera = scene.camera();
    HandleKeyEvents(window, camera, delta_time);
    HandleMouseEvents(window, camera);
    engine.Render(scene);
  });
}

}  // namespace game
