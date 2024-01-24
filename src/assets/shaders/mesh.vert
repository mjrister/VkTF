#version 460

layout(binding = 0, set = 0) uniform CameraTransforms {
  mat4 view_transform;
  mat4 projection_transform;
} camera_transforms;

layout(push_constant) uniform PushConstants {
  mat4 model_transform;
} push_constants;

layout(location = 0) in vec3 position;
layout(location = 1) in vec2 texture_coordinates;
layout(location = 2) in vec3 normal;

layout(location = 0) out Vertex {
  vec3 position;
  vec2 texture_coordinates;
  vec3 normal;
} vertex;

void main() {
  const mat4 model_view_transform = camera_transforms.view_transform * push_constants.model_transform;
  const vec4 model_view_position = model_view_transform * vec4(position, 1.0);
  vertex.position = model_view_position.xyz;
  vertex.texture_coordinates = texture_coordinates;
  vertex.normal = normalize(mat3(model_view_transform) * normal);
  gl_Position = camera_transforms.projection_transform * model_view_position;
}
