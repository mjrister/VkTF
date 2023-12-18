#include "app.h"  // NOLINT(build/include_subdir)

#include <GLFW/glfw3.h>
#include <glm/glm.hpp>

#include "graphics/model.h"

namespace {

constexpr auto kWindowWidth = 1920;
constexpr auto kWindowHeight = 1080;

gfx::Camera CreateCamera(const float aspect_ratio) {
  static constexpr glm::vec3 kLookFrom{0.0f, 0.0f, 1024.0f};
  static constexpr glm::vec3 kLookAt{0.0f};
  static constexpr glm::vec3 kUp{0.0f, 1.0f, 0.0f};
  const gfx::Camera::ViewFrustum view_frustum{.field_of_view_y = glm::radians(45.0f),
                                              .aspect_ratio = aspect_ratio,
                                              .z_near = 0.1f,
                                              .z_far = 10'000.0f};
  return gfx::Camera{kLookFrom, kLookAt, kUp, view_frustum};
}

}  // namespace

gfx::App::App()
    : window_{"VkRender", Window::Extent{.width = kWindowWidth, .height = kWindowHeight}},
      engine_{window_},
      camera_{CreateCamera(window_.GetAspectRatio())},
      model_{engine_.device(), "assets/models/survival_backpack.glb"} {
  window_.OnKeyEvent([this](const auto key, const auto action) { HandleKeyEvent(key, action); });
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
