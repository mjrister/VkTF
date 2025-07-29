add_library(engine STATIC)

target_sources(engine PUBLIC FILE_SET CXX_MODULES
                             FILES bounding_box.cppm
                                   buffer.cppm
                                   camera.cppm
                                   command_pool.cppm
                                   data_view.cppm
                                   delta_time.cppm
                                   descriptor_pool.cppm
                                   device.cppm
                                   engine.cppm
                                   glslang_compiler.cppm
                                   gltf_asset.cppm
                                   graphics_pipeline.cppm
                                   image.cppm
                                   instance.cppm
                                   ktx_texture.cppm
                                   log.cppm
                                   material.cppm
                                   mesh.cppm
                                   model.cppm
                                   physical_device.cppm
                                   queue.cppm
                                   scene.cppm
                                   shader_module.cppm
                                   swapchain.cppm
                                   texture.cppm
                                   vma_allocator.cppm
                                   window.cppm)

find_package(Ktx CONFIG REQUIRED)
find_package(SPIRV-Tools-opt CONFIG REQUIRED)
find_package(VulkanHeaders CONFIG REQUIRED)
find_package(VulkanMemoryAllocator CONFIG REQUIRED)
find_package(glfw3 CONFIG REQUIRED)
find_package(glslang CONFIG REQUIRED)
find_package(glm CONFIG REQUIRED)

target_link_libraries(engine PUBLIC GPUOpen::VulkanMemoryAllocator
                                    KTX::ktx
                                    SPIRV-Tools-opt
                                    Vulkan::Headers
                                    glfw
                                    glm::glm
                                    glslang::glslang
                                    glslang::glslang-default-resource-limits)

target_compile_definitions(engine PUBLIC GLFW_INCLUDE_VULKAN
                                         GLM_FORCE_DEFAULT_ALIGNED_GENTYPES
                                         GLM_FORCE_DEPTH_ZERO_TO_ONE
                                         GLM_FORCE_XYZW_ONLY
                                         VMA_STATIC_VULKAN_FUNCTIONS=0
                                         VMA_DYNAMIC_VULKAN_FUNCTIONS=0
                                         VULKAN_HPP_DISPATCH_LOADER_DYNAMIC=1
                                         VULKAN_HPP_NO_CONSTRUCTORS)
