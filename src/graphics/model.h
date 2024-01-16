#ifndef SRC_GRAPHICS_MODEL_H_
#define SRC_GRAPHICS_MODEL_H_

#include <filesystem>
#include <memory>
#include <vector>

#include <glm/gtc/matrix_transform.hpp>
#include <glm/mat4x4.hpp>
#include <vulkan/vulkan.hpp>

#include "graphics/material.h"
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

  Model(const Device& device, const std::filesystem::path& filepath);

  void Render(vk::CommandBuffer command_buffer, vk::PipelineLayout pipeline_layout) const;

  void Translate(const float dx, const float dy, const float dz) const {
    auto& transform = root_node_->transform;
    transform = glm::translate(transform, glm::vec3{dx, dy, dz});
  }

  void Rotate(const glm::vec3& axis, const float angle) const {
    auto& transform = root_node_->transform;
    transform = glm::rotate(transform, angle, axis);
  }

  void Scale(const float sx, const float sy, const float sz) const {
    auto& transform = root_node_->transform;
    transform = glm::scale(transform, glm::vec3{sx, sy, sz});
  }

private:
  std::unique_ptr<Node> root_node_;
  Materials materials_;
};

}  // namespace gfx

#endif  // SRC_GRAPHICS_MODEL_H_
