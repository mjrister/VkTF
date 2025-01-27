add_executable(game main.cpp)

target_sources(game PRIVATE FILE_SET CXX_MODULES
                            FILES game.cppm)

target_link_libraries(game PRIVATE graphics)

# create a symlink to the assets directory to avoid copying large source files to the binary directory
set(ASSETS_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/assets)
set(ASSETS_BINARY_DIR ${CMAKE_CURRENT_BINARY_DIR}/assets)
add_custom_target(create_assets_symlink ALL
                  COMMAND ${CMAKE_COMMAND} -E create_symlink ${ASSETS_SOURCE_DIR} ${ASSETS_BINARY_DIR}
                  COMMENT "Creating symlink ${ASSETS_BINARY_DIR}")

# compile GLSL shaders from the source directory to SPIR-V modules in the binary directory
include(${CMAKE_SOURCE_DIR}/cmake/compile_shader.cmake)
set(SHADERS_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/shaders)
set(SHADERS_BINARY_DIR ${CMAKE_CURRENT_BINARY_DIR}/shaders)
compile_shader(vertex.glsl vert)
compile_shader(fragment.glsl frag)

add_custom_target(compile_shaders DEPENDS ${SPIRV_BINARY_FILEPATHS})
add_dependencies(game compile_shaders)
