cmake_minimum_required(VERSION 3.31)
project(VkTF VERSION 0.1.1 LANGUAGES CXX)

# Defining CMAKE_CXX_FLAGS and CMAKE_EXE_LINKER_FLAGS in CMake presets will overwrite default values provided by CMake.
# To prevent this, compile and link options for a preset are defined in cache/environment variables and applied here.
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} $ENV{CXX_FLAGS} $CACHE{CXX_FLAGS}")
set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} $CACHE{CXX_LINKER_FLAGS}")

add_subdirectory(src)

enable_testing()
add_subdirectory(tests)
