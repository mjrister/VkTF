cmake_minimum_required(VERSION 3.30)
project(
  vkrender
  VERSION 0.0.3
  LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CXX_SCAN_FOR_MODULES OFF)

if(MSVC)
  add_compile_options(/W4 /WX)
  if(CMAKE_BUILD_TYPE STREQUAL Debug)
    add_compile_options(/fsanitize=address)
  else()
    add_compile_options(/GL /Ob3 /fp:fast)
    add_link_options(/LTCG:INCREMENTAL)
  endif()
else()
  add_compile_options(-Wall -Werror -Wextra -Wpedantic)
  if(CMAKE_BUILD_TYPE STREQUAL Debug)
    add_compile_options(-fsanitize=address)
    add_link_options(-fsanitize=address)
  else()
    add_compile_options(-O3 -ffast-math -flto)
    add_link_options(-flto)
  endif()
endif()

add_subdirectory(src)

enable_testing()
add_subdirectory(tests)
