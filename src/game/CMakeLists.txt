add_executable(game)

target_sources(game PRIVATE FILE_SET CXX_MODULES
                            FILES game.cppm
                    PUBLIC main.cpp)

target_link_libraries(game PRIVATE graphics)

find_package(glslang CONFIG REQUIRED)
get_target_property(GLSLANG_EXECUTABLE glslang::glslang-standalone LOCATION)

set(ASSETS_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/assets)
set(ASSETS_BINARY_DIR ${CMAKE_CURRENT_BINARY_DIR}/assets)

# create a symlink to the assets directory to avoid copying large source files to the binary directory
add_custom_target(create_assets_symlink ALL
                  COMMAND ${CMAKE_COMMAND} -E create_symlink ${ASSETS_SOURCE_DIR} ${ASSETS_BINARY_DIR}
                  COMMENT "Creating symlink from ${ASSETS_SOURCE_DIR} to ${ASSETS_BINARY_DIR}")

set(SHADERS_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/shaders)
set(SHADERS_BINARY_DIR ${CMAKE_CURRENT_BINARY_DIR}/shaders)

# compile GLSL shaders in the source directory to SPIR-V modules in the binary directory
function(compile_shader GLSL_FILENAME)
  set(GLSL_SOURCE_FILEPATH ${SHADERS_SOURCE_DIR}/${GLSL_FILENAME})
  set(GLSL_BINARY_FILEPATH ${SHADERS_BINARY_DIR}/${GLSL_FILENAME})
  set(SPIRV_BINARY_FILEPATH ${GLSL_BINARY_FILEPATH}.spv)

  add_custom_command(OUTPUT  ${SPIRV_BINARY_FILEPATH} ${GLSL_BINARY_FILEPATH}
                     COMMAND ${GLSLANG_EXECUTABLE} -V -o ${SPIRV_BINARY_FILEPATH} ${GLSL_SOURCE_FILEPATH}
                     # create a symlink to the original source file to enable runtime glslang compilation
                     COMMAND ${CMAKE_COMMAND} -E create_symlink ${GLSL_SOURCE_FILEPATH} ${GLSL_BINARY_FILEPATH}
                     DEPENDS ${GLSL_SOURCE_FILEPATH}
                     COMMENT "Compiling ${GLSL_SOURCE_FILEPATH}")

  list(APPEND SPIRV_BINARY_FILEPATHS ${SPIRV_BINARY_FILEPATH})
  return(PROPAGATE SPIRV_BINARY_FILEPATHS)
endfunction()

compile_shader(mesh.vert)
compile_shader(mesh.frag)

add_custom_target(compile_shaders DEPENDS ${SPIRV_BINARY_FILEPATHS})
add_dependencies(game compile_shaders)
