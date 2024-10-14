add_executable(game main.cpp)

target_sources(game PRIVATE FILE_SET CXX_MODULES FILES game.cppm)
target_link_libraries(game PRIVATE graphics)

set(ASSETS_SRC_DIR ${CMAKE_CURRENT_SOURCE_DIR}/assets)
set(ASSETS_DST_DIR ${CMAKE_CURRENT_BINARY_DIR}/assets)

add_custom_target(copy_assets ALL
                  COMMAND ${CMAKE_COMMAND} -E copy_directory_if_different ${ASSETS_SRC_DIR} ${ASSETS_DST_DIR}
                  COMMENT "Copying assets from ${ASSETS_SRC_DIR} to ${ASSETS_DST_DIR}")
