add_executable(tests graphics/arcball_test.cpp
                     graphics/data_view_test.cpp
                     graphics/obj_loader_test.cpp)

find_package(GTest CONFIG REQUIRED)

target_link_libraries(tests PRIVATE GTest::gtest_main asan graphics warnings)

target_include_directories(tests PRIVATE ${CMAKE_SOURCE_DIR})

include(GoogleTest)
gtest_discover_tests(tests)
