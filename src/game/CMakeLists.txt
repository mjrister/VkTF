add_executable(game main.cpp)

target_sources(game PUBLIC FILE_SET CXX_MODULES FILES delta_time.cppm game.cppm)

target_link_libraries(game PRIVATE graphics)
target_include_directories(game PRIVATE include)

add_custom_target(copy_assets ALL COMMAND ${CMAKE_COMMAND} -E copy_directory_if_different ${CMAKE_CURRENT_SOURCE_DIR}/assets
                                                                                          ${CMAKE_CURRENT_BINARY_DIR}/assets)
