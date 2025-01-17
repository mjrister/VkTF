module;

#include <cassert>
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
#include <spirv-tools/optimizer.hpp>

export module glslang_compiler;

namespace gfx::glslang {

export enum class SpirvOptimization : std::uint8_t { kNone, kSize, kSpeed };

export std::vector<std::uint32_t> Compile(const std::string& glsl_shader,
                                          glslang_stage_t glslang_stage,
                                          SpirvOptimization spirv_optimization);

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

using UniqueGlslangShader = std::unique_ptr<glslang_shader_t, decltype(&glslang_shader_delete)>;
using UniqueGlslangProgram = std::unique_ptr<glslang_program_t, decltype(&glslang_program_delete)>;

constexpr auto kGlslangMessages =
#ifndef NDEBUG
    GLSLANG_MSG_DEBUG_INFO_BIT |
#endif
    GLSLANG_MSG_SPV_RULES_BIT | GLSLANG_MSG_VULKAN_RULES_BIT;

template <typename Fn, typename T>
  requires requires(Fn glslang_get_fn, T* glslang_element) {
    { glslang_get_fn(glslang_element) } -> std::same_as<const char*>;
  }
void Print(std::ostream& ostream, Fn glslang_get_fn, T* const glslang_element) {
  if (const auto* const message = glslang_get_fn(glslang_element); message != nullptr) {
    if (const std::string_view message_view = message; !message_view.empty()) {
      std::println(ostream, "{}", message_view);
    }
  }
}

UniqueGlslangShader CreateGlslangShader(const std::string& glsl_shader, const glslang_stage_t glslang_stage) {
  const glslang_input_t glslang_input{.language = GLSLANG_SOURCE_GLSL,
                                      .stage = glslang_stage,
                                      .client = GLSLANG_CLIENT_VULKAN,
                                      .client_version = GLSLANG_TARGET_VULKAN_1_3,
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
  Print(std::clog, glslang_shader_get_info_log, glslang_shader.get());
  Print(std::clog, glslang_shader_get_info_debug_log, glslang_shader.get());
#endif

  if (glslang_shader_preprocess_result == 0) {
    throw std::runtime_error{
        std::format("Shader preprocessing failed at {} with GLSL source:\n{}", glslang_stage, glsl_shader)};
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

UniqueGlslangProgram CreateGlslangProgram(glslang_shader_t& glslang_shader, const glslang_stage_t glslang_stage) {
  auto glslang_program = UniqueGlslangProgram{glslang_program_create(), glslang_program_delete};
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

std::vector<std::uint32_t> OptimizeSpirvBinary(const std::vector<std::uint32_t>& spirv_binary,
                                               const gfx::glslang::SpirvOptimization spirv_optimization) {
  spvtools::Optimizer spirv_optimizer{SPV_ENV_VULKAN_1_3};

  switch (spirv_optimization) {
    using enum gfx::glslang::SpirvOptimization;
    case kNone:
      assert(false);  // avoid redundant copy when SPIR-V optimization is disabled
      return spirv_binary;
    case kSize:
      spirv_optimizer.RegisterSizePasses();
      break;
    case kSpeed:
      spirv_optimizer.RegisterPerformancePasses();
      break;
    default:
      std::unreachable();
  }

  std::vector<uint32_t> spirv_optimized_binary;
  if (!spirv_optimizer.Run(spirv_binary.data(), spirv_binary.size(), &spirv_optimized_binary)) {
    std::println(std::cerr, "SPIR-V optimization failed");
    return spirv_binary;
  }

  return spirv_optimized_binary;
}

std::vector<std::uint32_t> GenerateSpirvBinary(glslang_program_t& glslang_program,
                                               const glslang_stage_t glslang_stage,
                                               const gfx::glslang::SpirvOptimization spirv_optimization) {
  glslang_program_SPIRV_generate(&glslang_program, glslang_stage);

  const auto spirv_size = glslang_program_SPIRV_get_size(&glslang_program);
  if (spirv_size == 0) throw std::runtime_error{std::format("SPIR-V generation failed at {}", glslang_stage)};

  std::vector<std::uint32_t> spirv_binary(spirv_size);
  glslang_program_SPIRV_get(&glslang_program, spirv_binary.data());
#ifndef NDEBUG
  Print(std::clog, glslang_program_SPIRV_get_messages, &glslang_program);
#endif

  return spirv_optimization == gfx::glslang::SpirvOptimization::kNone
             ? spirv_binary
             : OptimizeSpirvBinary(spirv_binary, spirv_optimization);
}

}  // namespace

namespace gfx::glslang {

std::vector<std::uint32_t> Compile(const std::string& glsl_shader,
                                   const glslang_stage_t glslang_stage,
                                   const SpirvOptimization spirv_optimization) {
  [[maybe_unused]] const auto& glslang_process = GlslangProcess::Get();
  const auto glslang_shader = CreateGlslangShader(glsl_shader, glslang_stage);
  const auto glslang_program = CreateGlslangProgram(*glslang_shader, glslang_stage);
  return GenerateSpirvBinary(*glslang_program, glslang_stage, spirv_optimization);
}

}  // namespace gfx::glslang
