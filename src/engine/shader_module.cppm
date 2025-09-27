module;

#include <exception>
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
import log;

namespace vktf {

/**
 * @brief An abstraction for a Vulkan shader module.
 * @details This class handles creating a SPIR-V shader module from a file on disk. Files ending in @c .spv are loaded
 *          as SPIR-V binaries. Otherwise the file is treated as GLSL source code that is compiled at runtime.
 * @see https://registry.khronos.org/vulkan/specs/latest/man/html/VkShaderModule.html VkShaderModule
 */
export class [[nodiscard]] ShaderModule {
public:
  /** @brief The parameters for creating a @ref ShaderModule. */
  struct [[nodiscard]] CreateInfo {
    /** @brief The filepath to the shader representing either a SPIR-V binary or GLSL source code. */
    const std::filesystem::path& shader_filepath;

    /** @brief The stage the shader module will be used for. */
    vk::ShaderStageFlagBits shader_stage{};

    /** @brief The log for writing messages when creating a shader module. */
    Log& log;
  };

  /**
   * @brief Creates a @ref ShaderModule.
   * @param device The device for creating the shader module.
   * @param create_info @copybrief ShaderModule::CreateInfo
   * @throws std::runtime_error Thrown if the file at @ref ShaderModule::CreateInfo::shader_filepath is not a valid
   *                            SPIR-V binary or GLSL shader.
   */
  ShaderModule(vk::Device device, const CreateInfo& create_info);

  /** @brief Gets the underlying Vulkan shader module handle. */
  [[nodiscard]] vk::ShaderModule operator*() const noexcept { return *shader_module_; }

private:
  vk::UniqueShaderModule shader_module_;
};

}  // namespace vktf

module :private;

namespace vktf {

namespace {

constexpr std::size_t kSpirvWordSize = sizeof(SpirvWord);

std::vector<SpirvWord> ReadSpirvFile(const std::filesystem::path& spirv_filepath) {
  std::ifstream spirv_ifstream;
  spirv_ifstream.exceptions(std::ios::failbit | std::ios::badbit);
  spirv_ifstream.open(spirv_filepath, std::ios::ate | std::ios::binary);

  const std::size_t spirv_size_bytes = spirv_ifstream.tellg();
  if (spirv_size_bytes == 0 || spirv_size_bytes % kSpirvWordSize != 0) {
    throw std::runtime_error{
        std::format("Invalid SPIR-V file {} with size {}", spirv_filepath.string(), spirv_size_bytes)};
  }

  std::vector<SpirvWord> spirv_binary(spirv_size_bytes / kSpirvWordSize);
  const std::span spirv_char_span{reinterpret_cast<char*>(spirv_binary.data()), spirv_size_bytes};
  std::ospanstream spirv_osstream{spirv_char_span};
  spirv_ifstream.seekg(0, std::ios::beg);
  spirv_osstream << spirv_ifstream.rdbuf();

  return spirv_binary;
}

std::string ReadGlslFile(const std::filesystem::path& glsl_filepath) {
  std::ifstream glsl_ifstream;
  glsl_ifstream.exceptions(std::ios::failbit | std::ios::badbit);
  glsl_ifstream.open(glsl_filepath, std::ios::ate);

  const std::size_t glsl_size = glsl_ifstream.tellg();
  if (glsl_size == 0) throw std::runtime_error{std::format("Empty GLSL file {}", glsl_filepath.string())};

  std::string glsl_shader(glsl_size, '\0');
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
    case vk::ShaderStageFlagBits::eTaskEXT:
      return GLSLANG_STAGE_TASK;
    case vk::ShaderStageFlagBits::eMeshEXT:
      return GLSLANG_STAGE_MESH;
    default:
      throw std::runtime_error{std::format("Unsupported shader stage {}", vk::to_string(shader_stage))};
  }
}

std::vector<SpirvWord> GetSpirvBinary(const std::filesystem::path& shader_filepath,
                                      const vk::ShaderStageFlagBits shader_stage,
                                      Log& log) {
  try {
    if (shader_filepath.extension() == ".spv") return ReadSpirvFile(shader_filepath);

    const auto glsl_shader = ReadGlslFile(shader_filepath);
    const auto glslang_stage = GetGlslangStage(shader_stage);
    return glslang::Compile(glsl_shader, glslang_stage, log);

  } catch (const std::ios::failure&) {
    std::throw_with_nested(std::runtime_error{std::format("Failed to read {}", shader_filepath.string())});
  }
}

vk::UniqueShaderModule CreateShaderModule(const vk::Device device,
                                          const std::filesystem::path& shader_filepath,
                                          const vk::ShaderStageFlagBits shader_stage,
                                          Log& log) {
  const auto spirv_binary = GetSpirvBinary(shader_filepath, shader_stage, log);

  return device.createShaderModuleUnique(
      vk::ShaderModuleCreateInfo{.codeSize = spirv_binary.size() * kSpirvWordSize, .pCode = spirv_binary.data()});
}

}  // namespace

ShaderModule::ShaderModule(const vk::Device device, const CreateInfo& create_info)
    : shader_module_{
          CreateShaderModule(device, create_info.shader_filepath, create_info.shader_stage, create_info.log)} {}

}  // namespace vktf
