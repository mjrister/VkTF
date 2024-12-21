module;

#include <filesystem>
#include <format>
#include <fstream>
#include <ios>
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
constexpr auto kSpirvWordSizeBytes = sizeof(SpirvWord);

std::vector<SpirvWord> ReadSpirvFile(const std::filesystem::path& spirv_filepath) {
  if (std::ifstream ifstream{spirv_filepath, std::ios::ate | std::ios::binary}) {
    const std::streamsize size = ifstream.tellg();
    if (size % kSpirvWordSizeBytes != 0) {
      throw std::runtime_error{std::format("Invalid SPIR-V file {} with size {}", spirv_filepath.string(), size)};
    }
    std::vector<SpirvWord> spirv(static_cast<std::size_t>(size) / kSpirvWordSizeBytes);
    ifstream.seekg(0, std::ios::beg);
    ifstream.read(reinterpret_cast<char*>(spirv.data()), size);  // NOLINT(cppcoreguidelines-pro-type-reinterpret-cast)
    return spirv;
  }
  throw std::runtime_error{std::format("Failed to open {}", spirv_filepath.string())};
}

std::string ReadGlslFile(const std::filesystem::path& glsl_filepath) {
  if (std::ifstream ifstream{glsl_filepath, std::ios::ate}) {
    const std::streamsize size = ifstream.tellg();
    std::string glsl_shader(static_cast<std::size_t>(size), '\0');
    ifstream.seekg(0, std::ios::beg);
    ifstream.read(glsl_shader.data(), size);
    return glsl_shader;
  }
  throw std::runtime_error{std::format("Failed to open {}", glsl_filepath.string())};
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
  if (shader_filepath.extension() == ".spv") {
    return ReadSpirvFile(shader_filepath);
  }
  const auto glslang_stage = GetGlslangStage(shader_stage);
  const auto glsl_shader = ReadGlslFile(shader_filepath);
  return gfx::glslang::Compile(glsl_shader, glslang_stage);
}

}  // namespace

namespace gfx {

ShaderModule::ShaderModule(const vk::Device device,
                           const std::filesystem::path& shader_filepath,
                           const vk::ShaderStageFlagBits shader_stage) {
  const auto spirv = GetSpirv(shader_filepath, shader_stage);

  shader_module_ = device.createShaderModuleUnique(
      vk::ShaderModuleCreateInfo{.codeSize = spirv.size() * kSpirvWordSizeBytes, .pCode = spirv.data()});
}

}  // namespace gfx
