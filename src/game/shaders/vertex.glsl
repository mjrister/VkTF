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
layout(location = 2) in vec4 tangent;  // w-component indicates the signed handedness of the tangent basis
layout(location = 3) in vec2 texcoord_0;

layout(location = 0) out Fragment {
  vec3 world_position;
  vec3 world_normal;
  vec4 world_tangent;
  vec2 texcoord_0;
} fragment;

void main() {
  const mat4 model_transform = push_constants.model_transform;
  const mat3 model_rotation = mat3(model_transform);
  const mat4 view_transform = camera_transforms.view_transform;
  const mat4 projection_transform = camera_transforms.projection_transform;
  const vec4 world_position = model_transform * vec4(position, 1.0);

  fragment.world_position = world_position.xyz;
  fragment.world_normal = model_rotation * normal; // model rotation assumed to be an orthogonal matrix
  fragment.world_tangent = vec4(model_rotation * tangent.xyz, tangent.w);
  fragment.texcoord_0 = texcoord_0;

  gl_Position = projection_transform * view_transform * world_position;
}
