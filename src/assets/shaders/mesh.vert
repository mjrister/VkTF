#version 460

layout(push_constant) uniform PushConstants {
  mat4 model_transform;
} push_constants;

layout(set = 0, binding = 0) uniform CameraTransforms {
  mat4 view_transform;
  mat4 projection_transform;
} camera_transforms;

layout(location = 0) in vec3 position;
layout(location = 1) in vec3 normal;
layout(location = 2) in vec2 texture_coordinates;

layout(location = 0) out Vertex {
  vec3 position;
  vec3 normal;
  vec2 texture_coordinates;
} vertex;

void main() {
  const mat4 model_view_transform = camera_transforms.view_transform * push_constants.model_transform;
  const mat3 normal_transform = mat3(model_view_transform); // model-view transform is an orthogonal matrix
  const vec4 model_view_position = model_view_transform * vec4(position, 1.0);
  vertex.position = model_view_position.xyz;
  vertex.normal = normalize(normal_transform * normal);
  vertex.texture_coordinates = texture_coordinates;
  gl_Position = camera_transforms.projection_transform * model_view_position;
}
