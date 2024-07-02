add_executable(tests graphics/data_view_tests.cpp math/spherical_coordinates_test.cpp)

find_package(GTest CONFIG REQUIRED)

target_link_libraries(tests PRIVATE GTest::gtest_main graphics math)

include(GoogleTest)
gtest_discover_tests(tests)
