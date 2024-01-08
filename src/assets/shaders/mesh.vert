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
layout(location = 3) in vec3 tangent;

layout(location = 0) out Vertex {
  vec3 position;
  vec2 texture_coordinates;
  mat3 normal_transform; // TODO(matthew-rister): transform lighting vectors by inverse TBN matrix
} vertex;

mat3 GetNormalTransform(const mat4 model_view_transform) {
  const vec3 bitangent = cross(normal, tangent);
  return mat3(model_view_transform) * mat3(tangent, bitangent, normal);
}

void main() {
  const mat4 model_view_transform = camera_transforms.view_transform * push_constants.model_transform;
  const vec4 model_view_position = model_view_transform * vec4(position, 1.0);
  vertex.position = model_view_position.xyz;
  vertex.texture_coordinates = texture_coordinates;
  vertex.normal_transform = GetNormalTransform(model_view_transform);
  gl_Position = camera_transforms.projection_transform * model_view_position;
}
