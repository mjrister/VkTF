add_executable(game)

target_sources(
  game
  PUBLIC FILE_SET HEADERS BASE_DIRS ${SRC_DIR} FILES game.h
  PRIVATE game.cpp main.cpp)

target_link_libraries(game PRIVATE graphics)

set(ASSETS_SRC_DIR ${CMAKE_CURRENT_SOURCE_DIR}/assets)
set(ASSETS_DST_DIR ${CMAKE_CURRENT_BINARY_DIR}/assets)

add_custom_target(
  copy_assets ALL
  COMMAND ${CMAKE_COMMAND} -E copy_directory_if_different ${ASSETS_SRC_DIR} ${ASSETS_DST_DIR}
  COMMENT "Copying assets from ${ASSETS_SRC_DIR} to ${ASSETS_DST_DIR}")
