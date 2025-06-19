add_executable(tests engine/camera_test.cpp
                     engine/data_view_test.cpp
                     engine/log_test.cpp)

find_package(GTest CONFIG REQUIRED)

target_link_libraries(tests PRIVATE GTest::gtest_main engine)

include(GoogleTest)
gtest_discover_tests(tests)
