#pragma once

#include <cstdint>
#include <vector>

#include <glslang/Include/glslang_c_shader_types.h>

namespace gfx {

class GlslangCompiler {
public:
  [[nodiscard]] static const GlslangCompiler& Get() {
    static const GlslangCompiler instance;
    return instance;
  }

  GlslangCompiler(const GlslangCompiler&) = delete;
  GlslangCompiler& operator=(const GlslangCompiler&) = delete;

  GlslangCompiler(GlslangCompiler&&) noexcept = delete;
  GlslangCompiler& operator=(GlslangCompiler&&) noexcept = delete;

  ~GlslangCompiler() noexcept;

  [[nodiscard]] std::vector<std::uint32_t> Compile(glslang_stage_t stage, const char* glsl_source) const;

private:
  GlslangCompiler();
};

}  // namespace gfx
