module;

#include <concepts>
#include <cstdint>
#include <cstring>
#include <format>
#include <memory>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include <glslang/Include/glslang_c_interface.h>
#include <glslang/Public/resource_limits_c.h>

export module glslang_compiler;

import log;
namespace vktf {

/** @brief A type alias for a standard four-byte word in a SPIR-V binary. */
export using SpirvWord = std::uint32_t;

namespace glslang {

/**
 * @brief Compiles a GLSL shader to a SPIR-V binary.
 * @param glsl_shader The GLSL shader source code.
 * @param glslang_stage The GLSL shader stage (e.g., vertex, fragment).
 * @param log The log for writing shader compilation messages.
 * @return A vector of four-byte words representing the SPIR-V binary.
 * @throws std::runtime_error Thrown if shader compilation fails.
 */
export [[nodiscard]] std::vector<SpirvWord> Compile(const std::string& glsl_shader,
                                                    glslang_stage_t glslang_stage,
                                                    Log& log);

}  // namespace glslang
}  // namespace vktf

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

namespace vktf::glslang {

namespace {

class GlslangProcess {
public:
  [[nodiscard]] static const GlslangProcess& Instance() {
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

using UniqueGlslangShader = std::unique_ptr<glslang_shader_t, decltype(&glslang_shader_delete)>;
using UniqueGlslangProgram = std::unique_ptr<glslang_program_t, decltype(&glslang_program_delete)>;

using Severity = Log::Severity;

constexpr auto kGlslangMessages =
#ifndef NDEBUG
    GLSLANG_MSG_DEBUG_INFO_BIT |
#endif
    GLSLANG_MSG_SPV_RULES_BIT | GLSLANG_MSG_VULKAN_RULES_BIT;

template <typename Fn, typename T>
  requires std::same_as<std::invoke_result_t<Fn, T* const>, const char*>
void Print(Log& log, const Severity severity, Fn glslang_get_message, T* const glslang_element) {
  if (const auto* const message = glslang_get_message(glslang_element); message != nullptr) {
    if (const auto length = std::strlen(message); length > 0) {
      log(severity) << std::string_view{message, length};
    }
  }
}

UniqueGlslangShader CreateGlslangShader(const std::string& glsl_shader,
                                        const glslang_stage_t glslang_stage,
                                        [[maybe_unused]] Log& log) {
  const glslang_input_t glslang_input{.language = GLSLANG_SOURCE_GLSL,
                                      .stage = glslang_stage,
                                      .client = GLSLANG_CLIENT_VULKAN,
                                      .client_version = GLSLANG_TARGET_VULKAN_1_4,
                                      .target_language = GLSLANG_TARGET_SPV,
                                      .target_language_version = GLSLANG_TARGET_SPV_1_6,
                                      .code = glsl_shader.c_str(),
                                      .default_version = 460,
                                      .default_profile = GLSLANG_NO_PROFILE,
                                      .force_default_version_and_profile = 0,
                                      .forward_compatible = 0,
                                      .messages = static_cast<glslang_messages_t>(kGlslangMessages),
                                      .resource = glslang_default_resource()};

  auto glslang_shader = UniqueGlslangShader{glslang_shader_create(&glslang_input), glslang_shader_delete};
  if (glslang_shader == nullptr) {
    throw std::runtime_error{
        std::format("Shader creation failed at {} with GLSL source:\n{}", glslang_stage, glsl_shader)};
  }

  const auto glslang_shader_preprocess_result = glslang_shader_preprocess(glslang_shader.get(), &glslang_input);
#ifndef NDEBUG
  Print(log, Severity::kInfo, glslang_shader_get_info_log, glslang_shader.get());
  Print(log, Severity::kInfo, glslang_shader_get_info_debug_log, glslang_shader.get());
#endif

  if (glslang_shader_preprocess_result == 0) {
    throw std::runtime_error{
        std::format("Shader preprocessing failed at {} with GLSL source:\n{}", glslang_stage, glsl_shader)};
  }

  const auto glslang_shader_parse_result = glslang_shader_parse(glslang_shader.get(), &glslang_input);
#ifndef NDEBUG
  Print(log, Severity::kInfo, glslang_shader_get_info_log, glslang_shader.get());
  Print(log, Severity::kInfo, glslang_shader_get_info_debug_log, glslang_shader.get());
#endif

  if (glslang_shader_parse_result == 0) {
    throw std::runtime_error{std::format("Shader parsing failed at {} with GLSL source:\n{}",
                                         glslang_stage,
                                         glslang_shader_get_preprocessed_code(glslang_shader.get()))};
  }

  return glslang_shader;
}

UniqueGlslangProgram CreateGlslangProgram(glslang_shader_t& glslang_shader,
                                          const glslang_stage_t glslang_stage,
                                          [[maybe_unused]] Log& log) {
  auto glslang_program = UniqueGlslangProgram{glslang_program_create(), glslang_program_delete};
  if (glslang_program == nullptr) {
    throw std::runtime_error{std::format("Shader program creation failed at {}", glslang_stage)};
  }
  glslang_program_add_shader(glslang_program.get(), &glslang_shader);

  const auto glslang_program_link_result = glslang_program_link(glslang_program.get(), kGlslangMessages);
#ifndef NDEBUG
  Print(log, Severity::kInfo, glslang_program_get_info_log, glslang_program.get());
  Print(log, Severity::kInfo, glslang_program_get_info_debug_log, glslang_program.get());
#endif

  if (glslang_program_link_result == 0) {
    throw std::runtime_error{std::format("Shader program linking failed at {} with GLSL source:\n{}",
                                         glslang_stage,
                                         glslang_shader_get_preprocessed_code(&glslang_shader))};
  }

  return glslang_program;
}

std::vector<SpirvWord> GenerateSpirvBinary(glslang_program_t& glslang_program,
                                           const glslang_stage_t glslang_stage,
                                           [[maybe_unused]] Log& log) {
  glslang_spv_options_t glslang_spirv_options{
#ifndef NDEBUG
      .generate_debug_info = true,
      .disable_optimizer = true,
      .validate = true
#else
      .optimize_size = true
#endif
  };
  glslang_program_SPIRV_generate_with_options(&glslang_program, glslang_stage, &glslang_spirv_options);

  const auto spirv_size = glslang_program_SPIRV_get_size(&glslang_program);
  if (spirv_size == 0) throw std::runtime_error{std::format("SPIR-V generation failed at {}", glslang_stage)};

  std::vector<SpirvWord> spirv_binary(spirv_size);
  glslang_program_SPIRV_get(&glslang_program, spirv_binary.data());
#ifndef NDEBUG
  Print(log, Severity::kInfo, glslang_program_SPIRV_get_messages, &glslang_program);
#endif
  return spirv_binary;
}

}  // namespace

std::vector<SpirvWord> Compile(const std::string& glsl_shader, const glslang_stage_t glslang_stage, Log& log) {
  [[maybe_unused]] const auto& glslang_process = GlslangProcess::Instance();
  const auto glslang_shader = CreateGlslangShader(glsl_shader, glslang_stage, log);
  const auto glslang_program = CreateGlslangProgram(*glslang_shader, glslang_stage, log);
  return GenerateSpirvBinary(*glslang_program, glslang_stage, log);
}

}  // namespace vktf::glslang
