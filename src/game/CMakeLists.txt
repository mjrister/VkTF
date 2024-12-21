add_executable(game main.cpp)

target_sources(game PRIVATE FILE_SET CXX_MODULES
                            FILES game.cppm)

target_link_libraries(game PRIVATE graphics)

find_package(glslang CONFIG REQUIRED)
get_target_property(GLSLANG_EXECUTABLE glslang::glslang-standalone LOCATION)

set(SHADERS_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/assets/shaders)
set(SHADERS_BINARY_DIR ${CMAKE_CURRENT_BINARY_DIR}/assets/shaders)

function(compile_shader GLSL_FILENAME)
  set(GLSL_FILEPATH ${SHADERS_SOURCE_DIR}/${GLSL_FILENAME})
  set(SPIRV_FILEPATH ${SHADERS_BINARY_DIR}/${GLSL_FILENAME}.spv)
  add_custom_command(OUTPUT  ${SPIRV_FILEPATH}
                     COMMAND ${GLSLANG_EXECUTABLE} -V -o ${SPIRV_FILEPATH} ${GLSL_FILEPATH}
                     DEPENDS ${GLSL_FILEPATH}
                     COMMENT "Compiling ${GLSL_FILEPATH}")
  list(APPEND SPIRV_FILEPATHS ${SPIRV_FILEPATH})
  return(PROPAGATE SPIRV_FILEPATHS)
endfunction()

compile_shader(mesh.vert)
compile_shader(mesh.frag)

add_custom_target(compile_shaders DEPENDS ${SPIRV_FILEPATHS})
add_dependencies(game compile_shaders)

set(MODELS_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/assets/models)
set(MODELS_BINARY_DIR ${CMAKE_CURRENT_BINARY_DIR}/assets/models)

# create a symlink to avoid copying large assets to the CMake binary directory
add_custom_target(create_models_symlink ALL
                  COMMAND ${CMAKE_COMMAND} -E create_symlink ${MODELS_SOURCE_DIR} ${MODELS_BINARY_DIR})
