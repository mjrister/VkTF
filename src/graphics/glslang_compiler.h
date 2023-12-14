#ifndef SRC_GRAPHICS_GLSLANG_COMPILER_H_
#define SRC_GRAPHICS_GLSLANG_COMPILER_H_

#include <cstdint>
#include <vector>

#include <glslang/Include/glslang_c_shader_types.h>

namespace gfx {

/** \brief A GLSL to SPIR-V compiler using glslang. */
class GlslangCompiler {
public:
  /**
   * \brief Gets a reference to a static instance of the GLSL compiler.
   * \remark Access to the GLSL compiler is enforced through this method to ensure initialization and termination of a
   *         glslang process occur only once.
   */
  [[nodiscard]] static const GlslangCompiler& Get() {
    static const GlslangCompiler kInstance;
    return kInstance;
  }

  GlslangCompiler(const GlslangCompiler&) = delete;
  GlslangCompiler& operator=(const GlslangCompiler&) = delete;

  GlslangCompiler(GlslangCompiler&&) noexcept = delete;
  GlslangCompiler& operator=(GlslangCompiler&&) noexcept = delete;

  /** \brief Terminates the glslang process. */
  ~GlslangCompiler() noexcept;

  /**
   * \brief Compilers GLSL source code to SPIR-V.
   * \param stage The target GLSL shader stage.
   * \param glsl_source The GLSL source code to compile.
   * \return A vector containing the compiled SPIR-V bytecode.
   */
  [[nodiscard]] std::vector<std::uint32_t> Compile(glslang_stage_t stage, const char* glsl_source) const;

private:
  /** \brief Initializes the glslang process. */
  GlslangCompiler();
};

}  // namespace gfx

#endif  // SRC_GRAPHICS_GLSLANG_COMPILER_H_
