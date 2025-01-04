module;

#include <filesystem>
#include <format>
#include <fstream>
#include <ios>
#include <spanstream>
#include <stdexcept>
#include <string>

#include <glslang/Include/glslang_c_interface.h>
#include <vulkan/vulkan.hpp>

export module shader_module;

import glslang_compiler;

namespace gfx {

export class ShaderModule {
public:
  ShaderModule(vk::Device device, const std::filesystem::path& shader_filepath, vk::ShaderStageFlagBits shader_stage);

  [[nodiscard]] vk::ShaderModule operator*() const noexcept { return *shader_module_; }

private:
  vk::UniqueShaderModule shader_module_;
};

}  // namespace gfx

module :private;

namespace {

using SpirvWord = std::uint32_t;
constexpr auto kSpirvWordSize = sizeof(SpirvWord);

std::vector<SpirvWord> ReadSpirvFile(const std::filesystem::path& spirv_filepath) {
  std::ifstream spirv_ifstream;
  spirv_ifstream.exceptions(std::ios::failbit | std::ios::badbit);
  spirv_ifstream.open(spirv_filepath, std::ios::ate | std::ios::binary);

  const std::size_t spirv_size_bytes = spirv_ifstream.tellg();
  if (spirv_size_bytes % kSpirvWordSize != 0) {  // SPIR-V file size must be a multiple of 4 bytes
    throw std::runtime_error{std::format("Invalid SPIR-V file {}", spirv_filepath.string())};
  }

  std::vector<SpirvWord> spirv(spirv_size_bytes / kSpirvWordSize);
  std::span spirv_char_span{reinterpret_cast<char*>(spirv.data()), spirv_size_bytes};
  std::ospanstream spirv_osstream{spirv_char_span};
  spirv_ifstream.seekg(0, std::ios::beg);
  spirv_osstream << spirv_ifstream.rdbuf();

  return spirv;
}

std::string ReadGlslFile(const std::filesystem::path& glsl_filepath) {
  std::ifstream glsl_ifstream;
  glsl_ifstream.exceptions(std::ios::failbit | std::ios::badbit);
  glsl_ifstream.open(glsl_filepath, std::ios::ate);

  std::string glsl_shader(glsl_ifstream.tellg(), '\0');
  std::ospanstream glsl_osstream{glsl_shader};
  glsl_ifstream.seekg(0, std::ios::beg);
  glsl_osstream << glsl_ifstream.rdbuf();

  return glsl_shader;
}

glslang_stage_t GetGlslangStage(const vk::ShaderStageFlagBits shader_stage) {
  switch (shader_stage) {  // NOLINT(clang-diagnostic-switch-enum)
    case vk::ShaderStageFlagBits::eVertex:
      return GLSLANG_STAGE_VERTEX;
    case vk::ShaderStageFlagBits::eTessellationControl:
      return GLSLANG_STAGE_TESSCONTROL;
    case vk::ShaderStageFlagBits::eTessellationEvaluation:
      return GLSLANG_STAGE_TESSEVALUATION;
    case vk::ShaderStageFlagBits::eGeometry:
      return GLSLANG_STAGE_GEOMETRY;
    case vk::ShaderStageFlagBits::eFragment:
      return GLSLANG_STAGE_FRAGMENT;
    case vk::ShaderStageFlagBits::eCompute:
      return GLSLANG_STAGE_COMPUTE;
    case vk::ShaderStageFlagBits::eRaygenKHR:
      return GLSLANG_STAGE_RAYGEN;
    case vk::ShaderStageFlagBits::eAnyHitKHR:
      return GLSLANG_STAGE_ANYHIT;
    case vk::ShaderStageFlagBits::eClosestHitKHR:
      return GLSLANG_STAGE_CLOSESTHIT;
    case vk::ShaderStageFlagBits::eMissKHR:
      return GLSLANG_STAGE_MISS;
    case vk::ShaderStageFlagBits::eIntersectionKHR:
      return GLSLANG_STAGE_INTERSECT;
    case vk::ShaderStageFlagBits::eCallableKHR:
      return GLSLANG_STAGE_CALLABLE;
    default:
      throw std::runtime_error{std::format("Unsupported shader stage {}", vk::to_string(shader_stage))};
  }
}

std::vector<SpirvWord> GetSpirv(const std::filesystem::path& shader_filepath,
                                const vk::ShaderStageFlagBits shader_stage) {
  try {
    if (shader_filepath.extension() == ".spv") return ReadSpirvFile(shader_filepath);

    const auto glslang_stage = GetGlslangStage(shader_stage);
    const auto glsl_shader = ReadGlslFile(shader_filepath);
    return gfx::glslang::Compile(glsl_shader, glslang_stage);

  } catch (const std::ios::failure&) {
    std::throw_with_nested(std::runtime_error{std::format("Failed to read {}", shader_filepath.string())});
  }
}

}  // namespace

namespace gfx {

ShaderModule::ShaderModule(const vk::Device device,
                           const std::filesystem::path& shader_filepath,
                           const vk::ShaderStageFlagBits shader_stage) {
  const auto spirv = GetSpirv(shader_filepath, shader_stage);

  shader_module_ = device.createShaderModuleUnique(
      vk::ShaderModuleCreateInfo{.codeSize = spirv.size() * kSpirvWordSize, .pCode = spirv.data()});
}

}  // namespace gfx
