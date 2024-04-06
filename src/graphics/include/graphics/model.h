#ifndef SRC_GRAPHICS_INCLUDE_GRAPHICS_MODEL_H_
#define SRC_GRAPHICS_INCLUDE_GRAPHICS_MODEL_H_

#include <filesystem>
#include <memory>
#include <vector>

#include <vk_mem_alloc.h>
#include <glm/mat4x4.hpp>
#include <glm/vec3.hpp>
#include <vulkan/vulkan.hpp>

#include "graphics/mesh.h"

namespace gfx {
class Device;

struct Vertex {
  glm::vec3 position{0.0f};
  glm::vec3 normal{0.0f};
};

struct PushConstants {
  glm::mat4 model_transform{1.0f};
};

// TODO(matthew-rister): avoid leaking internal implementation details
struct Node {
  std::vector<Mesh> meshes;
  std::vector<std::unique_ptr<Node>> children;
  glm::mat4 transform{1.0f};
};

class Model {
public:
  Model(const std::filesystem::path& gltf_filepath, const Device& device, VmaAllocator allocator);

  void Translate(float dx, float dy, float dz) const;
  void Rotate(const glm::vec3& axis, float angle) const;
  void Scale(float sx, float sy, float sz) const;

  void Render(vk::CommandBuffer command_buffer, vk::PipelineLayout pipeline_layout) const;

private:
  std::unique_ptr<Node> root_node_;
};

}  // namespace gfx

#endif  // SRC_GRAPHICS_INCLUDE_GRAPHICS_MODEL_H_
