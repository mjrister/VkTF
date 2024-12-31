#version 460

const float kPi = 3.141592653589793;
const int kBaseColorSamplerIndex = 0;
const int kMetallicRoughnessSamplerIndex = 1;
const int kNormalSamplerIndex = 2;
const int kSamplerCount = 3;

layout(set = 1, binding = 0) uniform Material {
  vec4 base_color_factor;
  float metallic_factor;
  float roughness_factor;
  float normal_scale;
} material;

layout(set = 1, binding = 1) uniform sampler2D material_samplers[kSamplerCount];

layout(push_constant, std430) uniform PushConstants {
  layout(offset = 64) vec3 camera_position;
} push_constants;

layout(location = 0) in Fragment {
  vec3 position;
  mat3 normal_transform;  // TODO(matthew-rister): prefer normal transform matrix multplication in the vertex shader
  vec2 texture_coordinates_0;
  vec4 color;
} fragment;

layout(location = 0) out vec4 fragment_color;

vec4 GetImageColor(const uint sampler_index) {
  return texture(material_samplers[sampler_index], fragment.texture_coordinates_0);
}

vec3 GetNormal() {
  vec3 normal = 2.0 * GetImageColor(kNormalSamplerIndex).rgb - 1.0;  // convert sampled RGB values to the range [-1, 1]
  normal.xy *= vec2(material.normal_scale);  // glTF supports scaling sampled normals in the x/y directions
  return normalize(fragment.normal_transform * normal);
}

vec3 GetViewDirection() {
  return normalize(push_constants.camera_position - fragment.position);
}

vec3 GetLightDirection(out float light_distance) {
  // TODO(matthew-rister): import lights from a glTF scene using the KHR_lights_punctual extension
  const vec3 light_direction = push_constants.camera_position - fragment.position;
  light_distance = length(light_direction);
  return light_direction / light_distance;
}

  // D: Trowbridge-Reitz GGX normal distribution function
float GetMicrofacetDistribution(const vec3 normal, const vec3 halfway_direction, const float alpha) {
  const float alpha2 = alpha * alpha;
  const float n_dot_h = dot(normal, halfway_direction);
  const float d = n_dot_h * n_dot_h * (alpha2 - 1.0) + 1.0;
  return step(0.0, n_dot_h) * alpha2 / (kPi * d * d);
}

// G: Smith's joint masking-shadowing function
// V = G / (4 * N·L * N·V)
float GetMicrofacetVisibility(const vec3 normal_vector, const vec3 view_direction, const vec3 light_direction,
                              const vec3 halfway_direction, const float alpha) {
  const float alpha2 = alpha * alpha;
  const float h_dot_l = dot(halfway_direction, light_direction);
  const float h_dot_v = dot(halfway_direction, view_direction);
  const float n_dot_l = max(dot(normal_vector, light_direction), 0.0);
  const float n_dot_v = max(dot(normal_vector, view_direction), 0.0);
  return step(0.0, h_dot_l) / (n_dot_l + sqrt(alpha2 + (1.0 - alpha2) * n_dot_l * n_dot_l)) *
         step(0.0, h_dot_v) / (n_dot_v + sqrt(alpha2 + (1.0 - alpha2) * n_dot_v * n_dot_v));
}

// F: Schlick's approximation for the Fresnel term
vec3 GetFresnelApproximation(const vec3 view_direction, const vec3 halfway_direction, const vec3 f0) {
  const float h_dot_v = max(dot(halfway_direction, view_direction), 0.0);
  return f0 + (1.0 - f0) * pow(1.0 - h_dot_v, 5.0);
}

// implementation based on https://registry.khronos.org/glTF/specs/2.0/glTF-2.0.html#implementation
vec3 GetMaterialBrdf(const vec3 normal, const vec3 view_direction, const vec3 light_direction,
                     const vec3 halfway_direction, out vec4 base_color) {
  const vec2 metallic_roughness = GetImageColor(kMetallicRoughnessSamplerIndex).bg;
  const float metallic_factor = material.metallic_factor * metallic_roughness[0];
  const float roughness_factor = material.roughness_factor * metallic_roughness[1];

  const float alpha = roughness_factor * roughness_factor;
  const float D = GetMicrofacetDistribution(normal, halfway_direction, alpha);
  const float V = GetMicrofacetVisibility(normal, view_direction, light_direction, halfway_direction, alpha);

  base_color = fragment.color * material.base_color_factor * GetImageColor(kBaseColorSamplerIndex);
  const vec3 f0 = mix(vec3(0.04), base_color.rgb, metallic_factor);
  const vec3 F = GetFresnelApproximation(view_direction, halfway_direction, f0);

  const vec3 specular_brdf = D * V * F;
  const vec3 diffuse_brdf = (1.0 - F) / kPi * mix(base_color.rgb, vec3(0.0), metallic_factor);
  return diffuse_brdf + specular_brdf;
}

void main() {
  float light_distance = 0.0;
  const vec3 normal = GetNormal();
  const vec3 view_direction = GetViewDirection();
  const vec3 light_direction = GetLightDirection(light_distance);
  const vec3 halfway_direction = normalize(light_direction + view_direction);

  vec4 base_color = vec4(0.0);
  const vec3 material_brdf = GetMaterialBrdf(normal, view_direction, light_direction, halfway_direction, base_color);

  const float light_attenuation = 1.0 / light_distance;  // TODO(matthew-rister): use quadratic attenuation
  const vec3 kLightColor = vec3(1.0);
  const vec3 radiance_in = kLightColor * light_attenuation;

  const float cos_theta = max(dot(normal, light_direction), 0.0);
  const vec3 radiance_out = material_brdf * radiance_in * cos_theta;

  const vec3 kAmbiance = 0.05 * base_color.rgb;  // TODO(matthew-rister): use a more sophisticated ambient model
  fragment_color = vec4(kAmbiance + radiance_out, base_color.a);  // TODO(matthew-rister): add alpha-mode support
}
