find_package(glslang CONFIG REQUIRED)
get_target_property(GLSLANG_STANDALONE glslang::glslang-standalone LOCATION)

if(CMAKE_BUILD_TYPE STREQUAL Debug)
  # TODO: use -gVS and enable VK_KHR_shader_non_semantic_info when optional device extension support is added
  set(GLSLANG_FLAGS -Od -g --enhanced-msgs --error-column)
else()
  set(GLSLANG_FLAGS -Os)
endif()

function(compile_shader GLSL_FILENAME GLSL_STAGE)
  set(GLSL_SOURCE_FILEPATH ${SHADERS_SOURCE_DIR}/${GLSL_FILENAME})
  set(GLSL_BINARY_FILEPATH ${SHADERS_BINARY_DIR}/${GLSL_FILENAME})
  set(SPIRV_BINARY_FILEPATH ${GLSL_BINARY_FILEPATH}.spv)

  add_custom_command(OUTPUT  ${SPIRV_BINARY_FILEPATH} ${GLSL_BINARY_FILEPATH}
                     COMMAND ${GLSLANG_STANDALONE} -S ${GLSL_STAGE}
                                                   # TODO: use vulkan1.4 when glslang 15.3 is available in vcpkg
                                                   --target-env vulkan1.3
                                                   -o ${SPIRV_BINARY_FILEPATH}
                                                   ${GLSLANG_FLAGS}
                                                   ${GLSL_SOURCE_FILEPATH}
                     # create a symlink to the original source file to enable runtime shader compilation
                     COMMAND ${CMAKE_COMMAND} -E create_symlink ${GLSL_SOURCE_FILEPATH} ${GLSL_BINARY_FILEPATH}
                     DEPENDS ${GLSL_SOURCE_FILEPATH}
                     COMMENT "Compiling ${GLSL_SOURCE_FILEPATH}")

  list(APPEND SPIRV_BINARY_FILEPATHS ${SPIRV_BINARY_FILEPATH})
  return(PROPAGATE SPIRV_BINARY_FILEPATHS)
endfunction()
