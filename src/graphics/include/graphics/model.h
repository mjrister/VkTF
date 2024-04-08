#ifndef SRC_GRAPHICS_INCLUDE_GRAPHICS_MODEL_H_
#define SRC_GRAPHICS_INCLUDE_GRAPHICS_MODEL_H_

#include <cstdint>
#include <filesystem>
#include <memory>

#include <vk_mem_alloc.h>
#include <vulkan/vulkan.hpp>

namespace gfx {
class Camera;

class Model {
public:
  Model(const std::filesystem::path& gltf_filepath,
        vk::Device device,
        vk::Queue transfer_queue,
        std::uint32_t transfer_queue_family_index,
        vk::Extent2D viewport_extent,
        vk::SampleCountFlagBits msaa_sample_count,
        vk::RenderPass render_pass,
        VmaAllocator allocator);

  Model(const Model&) = delete;
  Model(Model&&) noexcept = default;

  Model& operator=(const Model&) = delete;
  Model& operator=(Model&&) noexcept = default;

  ~Model() noexcept;

  void Render(const Camera& camera, vk::CommandBuffer command_buffer) const;

private:
  class Node;

  std::unique_ptr<Node> root_node_;
  vk::UniquePipelineLayout pipeline_layout_;
  vk::UniquePipeline pipeline_;
};

}  // namespace gfx

#endif  // SRC_GRAPHICS_INCLUDE_GRAPHICS_MODEL_H_
