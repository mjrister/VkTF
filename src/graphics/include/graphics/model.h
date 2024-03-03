#ifndef SRC_GRAPHICS_INCLUDE_GRAPHICS_MODEL_H_
#define SRC_GRAPHICS_INCLUDE_GRAPHICS_MODEL_H_

#include <filesystem>
#include <memory>
#include <vector>

#include <glm/mat4x4.hpp>
#include <vulkan/vulkan.hpp>

#include "graphics/mesh.h"

namespace gfx {
class Device;

class Model {
public:
  struct Node {
    std::vector<Mesh> meshes;
    std::vector<std::unique_ptr<Node>> children;
    glm::mat4 transform{1.0f};
  };

  struct PushConstants {
    glm::mat4 node_transform{1.0f};
  };

  Model(const Device& device, VmaAllocator allocator, const std::filesystem::path& filepath);

  void Translate(float dx, float dy, float dz) const;
  void Rotate(const glm::vec3& axis, float angle) const;
  void Scale(float sx, float sy, float sz) const;

  void Render(vk::CommandBuffer command_buffer, vk::PipelineLayout pipeline_layout) const;

private:
  std::unique_ptr<Node> root_node_;
};

}  // namespace gfx

#endif  // SRC_GRAPHICS_INCLUDE_GRAPHICS_MODEL_H_
