add_executable(game main.cpp game.cpp window.cpp)

find_package(glfw3 CONFIG REQUIRED)
target_link_libraries(game PRIVATE glfw)

target_include_directories(game PRIVATE include)
target_link_libraries(game PRIVATE engine math)
