add_executable(tests graphics/camera_test.cpp
                     graphics/data_view_test.cpp
                     graphics/log_test.cpp)

find_package(GTest CONFIG REQUIRED)

target_link_libraries(tests PRIVATE GTest::gtest_main graphics)

include(GoogleTest)
gtest_discover_tests(tests)
