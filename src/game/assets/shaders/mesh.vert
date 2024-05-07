#version 460

layout(push_constant) uniform PushConstants {
  mat4 model_view_transform;
  mat4 projection_transform;
} push_constants;

layout(location = 0) in vec3 position;
layout(location = 1) in vec3 normal;
layout(location = 2) in vec2 texture_coordinates;

layout(location = 0) out Vertex {
  vec3 position;
  vec3 normal;
  vec2 texture_coordinates;
} vertex;

void main() {
  const vec4 model_view_position = push_constants.model_view_transform * vec4(position, 1.0);
  const mat3 normal_transform = mat3(push_constants.model_view_transform);
  vertex.position = model_view_position.xyz;
  vertex.normal = normalize(normal_transform * normal);
  vertex.texture_coordinates = texture_coordinates;
  gl_Position = push_constants.projection_transform * model_view_position;
}
