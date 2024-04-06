add_library(graphics STATIC allocator.cpp
                            buffer.cpp
                            camera.cpp
                            device.cpp
                            engine.cpp
                            glslang_compiler.cpp
                            image.cpp
                            instance.cpp
                            model.cpp
                            shader_module.cpp
                            swapchain.cpp
                            window.cpp)

find_package(VulkanHeaders CONFIG REQUIRED)
find_package(VulkanMemoryAllocator CONFIG REQUIRED)
find_package(glfw3 CONFIG REQUIRED)
find_package(glslang CONFIG REQUIRED)
find_package(glm CONFIG REQUIRED)

target_link_libraries(graphics PUBLIC GPUOpen::VulkanMemoryAllocator
                                      Vulkan::Headers
                                      glfw
                                      glm::glm
                                      glslang::glslang
                                      glslang::glslang-default-resource-limits
                                      glslang::SPIRV
                                      math)

target_compile_definitions(graphics PUBLIC GLFW_INCLUDE_VULKAN
                                           GLM_FORCE_DEFAULT_ALIGNED_GENTYPES
                                           GLM_FORCE_DEPTH_ZERO_TO_ONE
                                           GLM_FORCE_SIZE_T_LENGTH
                                           GLM_FORCE_XYZW_ONLY
                                           VMA_STATIC_VULKAN_FUNCTIONS=0
                                           VMA_DYNAMIC_VULKAN_FUNCTIONS=0
                                           VULKAN_HPP_DISPATCH_LOADER_DYNAMIC=1
                                           VULKAN_HPP_NO_CONSTRUCTORS)

target_include_directories(graphics PUBLIC ${CGLTF_INCLUDE_DIRS} include)
