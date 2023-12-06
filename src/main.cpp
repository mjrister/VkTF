#include <cstdlib>
#include <format>
#include <iostream>
#include <stdexcept>
#include <utility>

#include <GL/gl3w.h>

#include "graphics/delta_time.h"
#include "graphics/scene.h"
#include "graphics/window.h"

namespace {

[[maybe_unused]] void APIENTRY HandleDebugMessageReceived(const GLenum source,
                                                          const GLenum type,
                                                          const GLuint id,
                                                          const GLenum severity,
                                                          const GLsizei /*length*/,
                                                          const GLchar* const message,
                                                          const void* const /*user_param*/) {
  std::string message_source;
  switch (source) {
    case GL_DEBUG_SOURCE_API:
      message_source = "API";
      break;
    case GL_DEBUG_SOURCE_WINDOW_SYSTEM:
      message_source = "WINDOW SYSTEM";
      break;
    case GL_DEBUG_SOURCE_SHADER_COMPILER:
      message_source = "SHADER COMPILER";
      break;
    case GL_DEBUG_SOURCE_THIRD_PARTY:
      message_source = "THIRD PARTY";
      break;
    case GL_DEBUG_SOURCE_APPLICATION:
      message_source = "APPLICATION";
      break;
    default:
      message_source = "OTHER";
      break;
  }

  std::string message_type;
  switch (type) {
    case GL_DEBUG_TYPE_ERROR:
      message_type = "ERROR";
      break;
    case GL_DEBUG_TYPE_DEPRECATED_BEHAVIOR:
      message_type = "DEPRECATED BEHAVIOR";
      break;
    case GL_DEBUG_TYPE_UNDEFINED_BEHAVIOR:
      message_type = "UNDEFINED BEHAVIOR";
      break;
    case GL_DEBUG_TYPE_PORTABILITY:
      message_type = "PORTABILITY";
      break;
    case GL_DEBUG_TYPE_PERFORMANCE:
      message_type = "PERFORMANCE";
      break;
    default:
      message_type = "OTHER";
      break;
  }

  std::string message_severity;
  switch (severity) {
    case GL_DEBUG_SEVERITY_HIGH:
      message_severity = "HIGH";
      break;
    case GL_DEBUG_SEVERITY_MEDIUM:
      message_severity = "MEDIUM";
      break;
    case GL_DEBUG_SEVERITY_LOW:
      message_severity = "LOW";
      break;
    case GL_DEBUG_SEVERITY_NOTIFICATION:
      message_severity = "NOTIFICATION";
      break;
    default:
      message_severity = "OTHER";
      break;
  }

  std::clog << format("OpenGL Debug ({}): Source: {}, Type: {}, Severity: {}\n{}\n",
                      id,
                      message_source,
                      message_type,
                      message_severity,
                      message);
}

void InitializeGl3w(const std::pair<int, int>& opengl_version) {
  if (gl3wInit() != GL3W_OK) {
    throw std::runtime_error{"OpenGL initialization failed"};
  }
  if (const auto [major_version, minor_version] = opengl_version; gl3wIsSupported(major_version, minor_version) == 0) {
    throw std::runtime_error{std::format("OpenGL {}.{} not supported", major_version, minor_version)};
  }
#ifndef NDEBUG
  // NOLINTBEGIN(cppcoreguidelines-pro-type-reinterpret-cast)
  std::clog << std::format("OpenGL version: {}, GLSL version: {}\n",
                           reinterpret_cast<const char*>(glGetString(GL_VERSION)),
                           reinterpret_cast<const char*>(glGetString(GL_SHADING_LANGUAGE_VERSION)));
  // NOLINTEND(cppcoreguidelines-pro-type-reinterpret-cast)
  glEnable(GL_DEBUG_OUTPUT_SYNCHRONOUS);
  glDebugMessageCallback(HandleDebugMessageReceived, nullptr);
#endif
}

}  // namespace

int main() {
  try {
    constexpr auto* kProjectTitle = "Mesh Simplification";
    constexpr auto kWindowDimensions = std::make_pair(1920, 1080);
    constexpr auto kOpenGlVersion = std::make_pair(4, 1);
    qem::Window window{kProjectTitle, kWindowDimensions, kOpenGlVersion};

    InitializeGl3w(kOpenGlVersion);
    glEnable(GL_DEPTH_TEST);
    glEnable(GL_CULL_FACE);
    glEnable(GL_MULTISAMPLE);

    qem::DeltaTime delta_time;
    qem::Scene scene{&window};

    while (!window.IsClosed()) {
      delta_time.Update();
      window.Update();
      scene.Render(delta_time.get());
    }
  } catch (const std::exception& e) {
    std::cerr << e.what() << std::endl;
    return EXIT_FAILURE;
  } catch (...) {
    std::cerr << "An unknown error occurred" << std::endl;
    return EXIT_FAILURE;
  }
  return EXIT_SUCCESS;
}
