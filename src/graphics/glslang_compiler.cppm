module;

#include <concepts>
#include <cstdint>
#include <format>
#include <iostream>
#include <memory>
#include <print>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include <glslang/Include/glslang_c_interface.h>
#include <glslang/Public/resource_limits_c.h>

export module glslang_compiler;

namespace gfx::glslang {

export std::vector<std::uint32_t> Compile(glslang_stage_t glslang_stage, const std::string& glsl_source);

}  // namespace gfx::glslang

module :private;

template <>
struct std::formatter<glslang_stage_t> : std::formatter<std::string_view> {
  [[nodiscard]] auto format(const glslang_stage_t glslang_stage, auto& format_context) const {
    return std::formatter<std::string_view>::format(to_string(glslang_stage), format_context);
  }

private:
  static std::string_view to_string(const glslang_stage_t glslang_stage) noexcept {
    switch (glslang_stage) {
      // clang-format off
#define CASE(kValue) case kValue: return #kValue;  // NOLINT(cppcoreguidelines-macro-usage)
      CASE(GLSLANG_STAGE_VERTEX)
      CASE(GLSLANG_STAGE_TESSCONTROL)
      CASE(GLSLANG_STAGE_TESSEVALUATION)
      CASE(GLSLANG_STAGE_GEOMETRY)
      CASE(GLSLANG_STAGE_FRAGMENT)
      CASE(GLSLANG_STAGE_COMPUTE)
      CASE(GLSLANG_STAGE_RAYGEN)
      CASE(GLSLANG_STAGE_INTERSECT)
      CASE(GLSLANG_STAGE_ANYHIT)
      CASE(GLSLANG_STAGE_CLOSESTHIT)
      CASE(GLSLANG_STAGE_MISS)
      CASE(GLSLANG_STAGE_CALLABLE)
      CASE(GLSLANG_STAGE_TASK)
      CASE(GLSLANG_STAGE_MESH)
      CASE(GLSLANG_STAGE_COUNT)
#undef CASE
      // clang-format on
      default:
        std::unreachable();
    }
  }
};

namespace {

class GlslangProcess {
public:
  [[nodiscard]] static const GlslangProcess& Get() {
    static const GlslangProcess kInstance;
    return kInstance;
  }

  GlslangProcess(const GlslangProcess&) = delete;
  GlslangProcess& operator=(const GlslangProcess&) = delete;

  GlslangProcess(GlslangProcess&&) noexcept = delete;
  GlslangProcess& operator=(GlslangProcess&&) noexcept = delete;

  ~GlslangProcess() noexcept { glslang_finalize_process(); }

private:
  GlslangProcess() {
    if (glslang_initialize_process() == 0) {
      throw std::runtime_error{"glslang initialization failed"};
    }
  }
};

using GlslangShader = std::unique_ptr<glslang_shader_t, decltype(&glslang_shader_delete)>;
using GlslangProgram = std::unique_ptr<glslang_program_t, decltype(&glslang_program_delete)>;

constexpr auto kGlslangMessages =
// NOLINTBEGIN(hicpp-signed-bitwise): glslang bit flags use signed integers
#ifndef NDEBUG
    GLSLANG_MSG_DEBUG_INFO_BIT |
#endif
    GLSLANG_MSG_SPV_RULES_BIT | GLSLANG_MSG_VULKAN_RULES_BIT;
// NOLINTEND(hicpp-signed-bitwise)

template <typename Fn, typename T>
  requires requires(Fn glslang_get_fn, T* glslang_element) {
    { glslang_get_fn(glslang_element) } -> std::same_as<const char*>;
  }
void Print(std::ostream& os, Fn glslang_get_fn, T* const glslang_element) {
  if (const auto* const message = glslang_get_fn(glslang_element); message != nullptr) {
    if (const std::string_view message_view = message; !message_view.empty()) {
      std::println(os, "{}", message_view);
    }
  }
}

GlslangShader CreateGlslangShader(const glslang_stage_t glslang_stage, const std::string& glsl_source) {
  const glslang_input_t glslang_input{.language = GLSLANG_SOURCE_GLSL,
                                      .stage = glslang_stage,
                                      .client = GLSLANG_CLIENT_VULKAN,
                                      .client_version = GLSLANG_TARGET_VULKAN_1_3,
                                      .target_language = GLSLANG_TARGET_SPV,
                                      .target_language_version = GLSLANG_TARGET_SPV_1_6,
                                      .code = glsl_source.c_str(),
                                      .default_version = 460,
                                      .default_profile = GLSLANG_NO_PROFILE,
                                      .force_default_version_and_profile = 0,
                                      .forward_compatible = 0,
                                      .messages = static_cast<glslang_messages_t>(kGlslangMessages),
                                      .resource = glslang_default_resource()};

  auto glslang_shader = GlslangShader{glslang_shader_create(&glslang_input), glslang_shader_delete};
  if (glslang_shader == nullptr) {
    throw std::runtime_error{
        std::format("Shader creation failed at {} with GLSL source:\n{}", glslang_stage, glsl_source)};
  }

  const auto glslang_shader_preprocess_result = glslang_shader_preprocess(glslang_shader.get(), &glslang_input);
#ifndef NDEBUG
  Print(std::clog, glslang_shader_get_info_log, glslang_shader.get());
  Print(std::clog, glslang_shader_get_info_debug_log, glslang_shader.get());
#endif

  if (glslang_shader_preprocess_result == 0) {
    throw std::runtime_error{
        std::format("Shader preprocessing failed at {} with GLSL source:\n{}", glslang_stage, glsl_source)};
  }

  const auto glslang_shader_parse_result = glslang_shader_parse(glslang_shader.get(), &glslang_input);
#ifndef NDEBUG
  Print(std::clog, glslang_shader_get_info_log, glslang_shader.get());
  Print(std::clog, glslang_shader_get_info_debug_log, glslang_shader.get());
#endif

  if (glslang_shader_parse_result == 0) {
    throw std::runtime_error{std::format("Shader parsing failed at {} with GLSL source:\n{}",
                                         glslang_stage,
                                         glslang_shader_get_preprocessed_code(glslang_shader.get()))};
  }

  return glslang_shader;
}

GlslangProgram CreateGlslangProgram(const glslang_stage_t glslang_stage, glslang_shader_t& glslang_shader) {
  auto glslang_program = GlslangProgram{glslang_program_create(), glslang_program_delete};
  if (glslang_program == nullptr) {
    throw std::runtime_error{std::format("Shader program creation failed at {}", glslang_stage)};
  }
  glslang_program_add_shader(glslang_program.get(), &glslang_shader);

  const auto glslang_program_link_result = glslang_program_link(glslang_program.get(), kGlslangMessages);
#ifndef NDEBUG
  Print(std::clog, glslang_program_get_info_log, glslang_program.get());
  Print(std::clog, glslang_program_get_info_debug_log, glslang_program.get());
#endif

  if (glslang_program_link_result == 0) {
    throw std::runtime_error{std::format("Shader program linking failed at {} with GLSL source:\n{}",
                                         glslang_stage,
                                         glslang_shader_get_preprocessed_code(&glslang_shader))};
  }

  return glslang_program;
}

std::vector<std::uint32_t> GenerateSpirv(const glslang_stage_t glslang_stage, glslang_program_t& glslang_program) {
  glslang_program_SPIRV_generate(&glslang_program, glslang_stage);

  const auto spirv_size = glslang_program_SPIRV_get_size(&glslang_program);
  if (spirv_size == 0) throw std::runtime_error{std::format("SPIR-V generation failed at {}", glslang_stage)};

  std::vector<std::uint32_t> spirv(spirv_size);
  glslang_program_SPIRV_get(&glslang_program, spirv.data());
#ifndef NDEBUG
  Print(std::clog, glslang_program_SPIRV_get_messages, &glslang_program);
#endif

  return spirv;
}

}  // namespace

namespace gfx::glslang {

std::vector<std::uint32_t> Compile(const glslang_stage_t glslang_stage, const std::string& glsl_source) {
  [[maybe_unused]] const auto& glslang_process = GlslangProcess::Get();
  const auto glslang_shader = CreateGlslangShader(glslang_stage, glsl_source);
  const auto glslang_program = CreateGlslangProgram(glslang_stage, *glslang_shader);
  return GenerateSpirv(glslang_stage, *glslang_program);
}

}  // namespace gfx::glslang
