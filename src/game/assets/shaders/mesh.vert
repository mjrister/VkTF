#version 460

layout(set = 0, binding = 0) uniform CameraTransforms {
  mat4 view_transform;
  mat4 projection_transform;
} camera_transforms;

layout(push_constant) uniform PushConstants {
  mat4 model_transform;
} push_constants;

layout(location = 0) in vec3 position;
layout(location = 1) in vec3 normal;
layout(location = 2) in vec4 tangent; // w-component indicates the signed handedness of the tangent basis
layout(location = 3) in vec2 texture_coordinates_0;

layout(location = 0) out Vertex {
  vec3 position;
  vec2 texture_coordinates_0;
  mat3 normal_transform;
 } vertex;

void main() {
  // model-view transform is assumed to consist of only translations, rotations, and uniform scaling
  const mat4 model_view_transform = camera_transforms.view_transform * push_constants.model_transform;
  const vec4 model_view_position = model_view_transform * vec4(position, 1.0);
  const vec3 bitangent = cross(normal, tangent.xyz) * tangent.w;
  const mat3 normal_transform = mat3(model_view_transform) * mat3(tangent.xyz, bitangent, normal);

  vertex.position = model_view_position.xyz;
  vertex.texture_coordinates_0 = texture_coordinates_0;
  vertex.normal_transform = normal_transform;

  gl_Position = camera_transforms.projection_transform * model_view_position;
}
