#ifndef SRC_GRAPHICS_INCLUDE_GRAPHICS_SCENE_H_
#define SRC_GRAPHICS_INCLUDE_GRAPHICS_SCENE_H_

#include <cstdint>
#include <filesystem>
#include <memory>
#include <unordered_map>
#include <vector>

#include <vk_mem_alloc.h>
#include <vulkan/vulkan.hpp>
#include <vulkan/vulkan_hash.hpp>

namespace gfx {
class Camera;
class Image;

class Scene {
public:
  Scene(const std::filesystem::path& gltf_filepath,
        const vk::PhysicalDeviceFeatures& physical_device_features,
        const vk::PhysicalDeviceLimits& physical_device_limits,
        vk::Device device,
        vk::Queue queue,
        std::uint32_t queue_family_index,
        vk::Extent2D viewport_extent,
        vk::SampleCountFlagBits msaa_sample_count,
        vk::RenderPass render_pass,
        VmaAllocator allocator);

  Scene(const Scene&) = delete;
  Scene(Scene&&) noexcept = default;

  Scene& operator=(const Scene&) = delete;
  Scene& operator=(Scene&&) noexcept = default;

  ~Scene() noexcept;

  void Render(const Camera& camera, vk::CommandBuffer command_buffer) const;

private:
  class Node;
  struct Material;

  vk::UniqueDescriptorPool descriptor_pool_;
  vk::UniqueDescriptorSetLayout descriptor_set_layout_;
  vk::UniquePipelineLayout pipeline_layout_;
  vk::UniquePipeline pipeline_;
  std::vector<Material> materials_;
  std::unordered_map<vk::SamplerCreateInfo, vk::UniqueSampler> samplers_;
  std::unique_ptr<const Node> root_node_;
};

}  // namespace gfx

#endif  // SRC_GRAPHICS_INCLUDE_GRAPHICS_SCENE_H_
