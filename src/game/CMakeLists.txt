add_executable(game game.cpp main.cpp)

target_link_libraries(game PRIVATE graphics)
target_include_directories(game PRIVATE ${CMAKE_SOURCE_DIR}/src)

add_custom_target(copy_assets ALL COMMAND ${CMAKE_COMMAND} -E copy_directory_if_different ${CMAKE_CURRENT_SOURCE_DIR}/assets
                                                                                          ${CMAKE_CURRENT_BINARY_DIR}/assets)
