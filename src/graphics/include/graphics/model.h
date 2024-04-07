#ifndef SRC_GRAPHICS_INCLUDE_GRAPHICS_MODEL_H_
#define SRC_GRAPHICS_INCLUDE_GRAPHICS_MODEL_H_

#include <filesystem>
#include <memory>

#include <vk_mem_alloc.h>
#include <glm/mat4x4.hpp>
#include <glm/vec3.hpp>
#include <vulkan/vulkan.hpp>

namespace gfx {
class Device;

struct Vertex {
  glm::vec3 position{0.0f};
  glm::vec3 normal{0.0f};
};

struct PushConstants {
  glm::mat4 model_transform{1.0f};
};

class Model {
public:
  Model(const std::filesystem::path& gltf_filepath, const Device& device, VmaAllocator allocator);

  Model(const Model&) = delete;
  Model(Model&&) noexcept = default;

  Model& operator=(const Model&) = delete;
  Model& operator=(Model&&) noexcept = default;

  ~Model() noexcept;

  void Render(vk::CommandBuffer command_buffer, vk::PipelineLayout pipeline_layout) const;

private:
  class Node;
  std::unique_ptr<Node> root_node_;
};

}  // namespace gfx

#endif  // SRC_GRAPHICS_INCLUDE_GRAPHICS_MODEL_H_
