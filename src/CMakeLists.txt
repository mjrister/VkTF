add_executable(app app.cpp main.cpp)

target_link_libraries(app PRIVATE asan fast_math graphics warnings)

add_subdirectory(graphics)
add_subdirectory(math)

add_custom_target(copy_assets ALL COMMAND ${CMAKE_COMMAND} -E copy_directory ${CMAKE_CURRENT_SOURCE_DIR}/assets
                                                                             ${CMAKE_CURRENT_BINARY_DIR}/assets)
