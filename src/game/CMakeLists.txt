add_executable(game main.cpp)

target_sources(game PRIVATE FILE_SET CXX_MODULES
                            FILES game.cppm)

target_link_libraries(game PRIVATE graphics)

set(ASSETS_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/assets)
set(ASSETS_BINARY_DIR ${CMAKE_CURRENT_BINARY_DIR}/assets)

add_custom_target(create_symlink ALL
                  COMMAND ${CMAKE_COMMAND} -E create_symlink ${ASSETS_SOURCE_DIR} ${ASSETS_BINARY_DIR})
