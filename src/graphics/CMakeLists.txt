add_library(graphics STATIC)

target_sources(graphics PUBLIC FILE_SET CXX_MODULES
                               FILES allocator.cppm
                                     buffer.cppm
                                     camera.cppm
                                     data_view.cppm
                                     delta_time.cppm
                                     device.cppm
                                     engine.cppm
                                     glslang_compiler.cppm
                                     image.cppm
                                     instance.cppm
                                     ktx_texture.cppm
                                     physical_device.cppm
                                     scene.cppm
                                     shader_module.cppm
                                     swapchain.cppm
                                     window.cppm)

find_package(Ktx CONFIG REQUIRED)
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
                                      KTX::ktx)

target_compile_definitions(graphics PUBLIC GLFW_INCLUDE_VULKAN
                                           GLM_FORCE_DEFAULT_ALIGNED_GENTYPES
                                           GLM_FORCE_DEPTH_ZERO_TO_ONE
                                           GLM_FORCE_XYZW_ONLY
                                           VMA_STATIC_VULKAN_FUNCTIONS=0
                                           VMA_DYNAMIC_VULKAN_FUNCTIONS=0
                                           VULKAN_HPP_DISPATCH_LOADER_DYNAMIC=1
                                           VULKAN_HPP_NO_CONSTRUCTORS)
