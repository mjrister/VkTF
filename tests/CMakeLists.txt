add_executable(tests graphics/data_view_tests.cpp)

find_package(GTest CONFIG REQUIRED)

target_link_libraries(tests PRIVATE GTest::gtest_main graphics)

include(GoogleTest)
gtest_discover_tests(tests)
