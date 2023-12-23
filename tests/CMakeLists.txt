add_executable(tests graphics/data_view_test.cpp math/spherical_coordinates_test.cpp)

find_package(GTest CONFIG REQUIRED)

target_link_libraries(tests PRIVATE GTest::gtest_main asan graphics warnings
                            INTERFACE math)
target_include_directories(tests PRIVATE ${CMAKE_SOURCE_DIR})

include(GoogleTest)
gtest_discover_tests(tests)
