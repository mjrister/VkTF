#ifndef SRC_GRAPHICS_INCLUDE_GRAPHICS_GLSLANG_COMPILER_H_
#define SRC_GRAPHICS_INCLUDE_GRAPHICS_GLSLANG_COMPILER_H_

#include <cstdint>
#include <stdexcept>
#include <vector>

#include <glslang/Include/glslang_c_interface.h>

namespace gfx {

class GlslangCompiler {
public:
  [[nodiscard]] static const GlslangCompiler& Get() {
    static const GlslangCompiler kInstance;
    return kInstance;
  }

  GlslangCompiler(const GlslangCompiler&) = delete;
  GlslangCompiler& operator=(const GlslangCompiler&) = delete;

  GlslangCompiler(GlslangCompiler&&) noexcept = delete;
  GlslangCompiler& operator=(GlslangCompiler&&) noexcept = delete;

  ~GlslangCompiler() noexcept { glslang_finalize_process(); }

  [[nodiscard]] std::vector<std::uint32_t> Compile(glslang_stage_t stage, const char* glsl_source) const;

private:
  GlslangCompiler() {
    if (glslang_initialize_process() == 0) {
      throw std::runtime_error{"glslang initialization failed"};
    }
  }
};

}  // namespace gfx

#endif  // SRC_GRAPHICS_INCLUDE_GRAPHICS_GLSLANG_COMPILER_H_
