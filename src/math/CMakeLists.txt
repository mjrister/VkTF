add_library(math STATIC)

target_sources(math PUBLIC FILE_SET CXX_MODULES
                           FILES spherical_coordinates.cppm)

find_package(glm CONFIG REQUIRED)

target_link_libraries(math PUBLIC glm::glm)

target_compile_definitions(math PUBLIC GLM_FORCE_DEFAULT_ALIGNED_GENTYPES
                                       GLM_FORCE_XYZW_ONLY)
