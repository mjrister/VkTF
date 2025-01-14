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
layout(location = 2) in vec4 tangent; // w-component indicates the signed handedness of the tangent basis
layout(location = 3) in vec2 texture_coordinates_0;
layout(location = 4) in vec4 color_0;

layout(location = 0) out Fragment {
  vec3 position;
  mat3 normal_transform;
  vec2 texture_coordinates_0;
  vec4 color_0;
} fragment;

mat3 GetNormalTransform(const mat4 model_transform) {
  const vec3 bitangent = cross(normal, tangent.xyz) * tangent.w;  // tangent and bitangent assumed to be unit length
  return mat3(model_transform) * mat3(tangent.xyz, bitangent, normal);  // model transform assumed to be orthogonal
}

void main() {
  const mat4 model_transform = push_constants.model_transform;
  const mat4 view_transform = camera_transforms.view_transform;
  const mat4 projection_transform = camera_transforms.projection_transform;
  const vec4 model_position = model_transform * vec4(position, 1.0);

  fragment.position = model_position.xyz;
  fragment.normal_transform = GetNormalTransform(model_transform);
  fragment.texture_coordinates_0 = texture_coordinates_0;
  fragment.color_0 = color_0;

  gl_Position = projection_transform * view_transform * model_position;
}
