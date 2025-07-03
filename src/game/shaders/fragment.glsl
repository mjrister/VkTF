#version 460

struct WorldLight {
  vec4 position;  // represents normalized light direction when w-component is zero
  vec4 color;
};

const float kPi = 3.1415927;
const float kEpsilon = 1.0e-7;
const uint kBaseColorSamplerIndex = 0;
const uint kMetallicRoughnessSamplerIndex = 1;
const uint kNormalSamplerIndex = 2;
const uint kMaterialSamplerCount = 3;

layout(push_constant) uniform PushConstants {
  layout(offset = 64) vec3 view_position;
} push_constants;

layout (constant_id = 0) const uint kLightCount = 1;

layout(set = 0, binding = 1) uniform WorldLights {
  WorldLight data[kLightCount];
} world_lights;

layout(set = 1, binding = 0) uniform MaterialProperties {
  vec4 base_color_factor;
  vec2 metallic_roughness_factor;
  float normal_scale;  // glTF allows scaling sampled normals in the x/y directions
} material_properties;

layout(set = 1, binding = 1) uniform sampler2D material_samplers[kMaterialSamplerCount];

layout(location = 0) in Fragment {
  vec3 world_position;
  vec3 world_normal;
  vec4 world_tangent;
  vec2 texcoord_0;
} fragment;

layout(location = 0) out vec4 fragment_color;

vec3 GetViewDirection() {
  return normalize(push_constants.view_position - fragment.world_position);
}

vec4 GetSampledImageColor(const uint sampler_index) {
  return texture(material_samplers[sampler_index], fragment.texcoord_0);
}

vec4 GetBaseColor() {
  return material_properties.base_color_factor * GetSampledImageColor(kBaseColorSamplerIndex);
}

vec2 GetMetallicRoughness() {
  return material_properties.metallic_roughness_factor * GetSampledImageColor(kMetallicRoughnessSamplerIndex).bg;
}

mat3 GetTbnTransform() {
  // the TBN transform is constructed here to ensure basis vectors remain orthonormal which can otherwise
  // break down due to fragment interpolation during rasterization if constructed in the vertex shader
  const vec3 normal = normalize(fragment.world_normal);
  const vec3 tangent = normalize(fragment.world_tangent.xyz);
  const vec3 bitangent = normalize(cross(normal, tangent)) * fragment.world_tangent.w;
  return mat3(tangent, bitangent, normal);
}

vec3 GetNormal() {
  vec3 normal = 2.0 * GetSampledImageColor(kNormalSamplerIndex).rgb - 1.0;  // convert RGB values from [0, 1] to [-1, 1]
  normal.xy *= vec2(material_properties.normal_scale);
  const mat3 tbn_transform = GetTbnTransform();
  return normalize(tbn_transform * normal);
}

float GetLightAttenuation(const float light_distance, const float has_position) {
  const float kDirectionalLightAttenuation = 1.0;  // do not attenuate directional lights
  const float point_light_attenuation = 1.0 / (light_distance * light_distance);
  return mix(kDirectionalLightAttenuation, point_light_attenuation, has_position);
}

vec3 GetLightDirection(const WorldLight world_light, out float light_attenuation) {
  const float kPointLightRadius = 0.1;
  const float has_position = float(world_light.position.w != 0.0);
  const vec3 light_direction = world_light.position.xyz - (has_position * fragment.world_position);
  const float light_distance = max(length(light_direction), kPointLightRadius);
  light_attenuation = GetLightAttenuation(light_distance, has_position);
  return light_direction / light_distance;
}

vec3 GetFresnelApproximation(const vec3 f0, const vec3 view_direction, const vec3 halfway_direction) {
  const float h_dot_v = dot(halfway_direction, view_direction);
  return f0 + (1.0 - f0) * pow(1.0 - abs(h_dot_v), 5.0);
}

float GetMicrofacetVisibility(const float alpha2, const vec3 light_direction, const vec3 normal,
                              const vec3 view_direction, const vec3 halfway_direction) {
  const float h_dot_l = dot(halfway_direction, light_direction);
  const float h_dot_v = dot(halfway_direction, view_direction);
  const float n_dot_l = dot(normal, light_direction);
  const float n_dot_v = dot(normal, view_direction);
  return step(0.0, h_dot_l) / (abs(n_dot_l) + sqrt(alpha2 + (1.0 - alpha2) * n_dot_l * n_dot_l) + kEpsilon) *
         step(0.0, h_dot_v) / (abs(n_dot_v) + sqrt(alpha2 + (1.0 - alpha2) * n_dot_v * n_dot_v) + kEpsilon);
}

float GetMicrofacetDistribution(const float alpha2, const vec3 normal, const vec3 halfway_direction) {
  const float n_dot_h = dot(normal, halfway_direction);
  const float d = n_dot_h * n_dot_h * (alpha2 - 1.0) + 1.0;
  return step(0.0, n_dot_h) * alpha2 / (kPi * d * d + kEpsilon);
}

// implementation based on https://registry.khronos.org/glTF/specs/2.0/glTF-2.0.html#implementation
vec3 GetMaterialBrdf(const vec4 base_color, const vec2 metallic_roughness, const vec3 light_direction,
                     const vec3 normal, const vec3 view_direction) {
  const vec3 halfway_direction = normalize(light_direction + view_direction);
  const float metallic_factor = metallic_roughness.x;
  const float roughness_factor = metallic_roughness.y;
  const float alpha = roughness_factor * roughness_factor;
  const float alpha2 = alpha * alpha;

  const vec3 f0 = mix(vec3(0.04), base_color.rgb, metallic_factor);
  const vec3 F = GetFresnelApproximation(f0, view_direction, halfway_direction);
  const float V = GetMicrofacetVisibility(alpha2, light_direction, normal, view_direction, halfway_direction);
  const float D = GetMicrofacetDistribution(alpha2, normal, halfway_direction);

  const vec3 diffuse_brdf = (1.0 - F) / kPi * mix(base_color.rgb, vec3(0.0), metallic_factor);
  const vec3 specular_brdf = F * V * D;
  return diffuse_brdf + specular_brdf;
}

void main() {
  const vec3 view_direction = GetViewDirection();
  const vec3 normal = GetNormal();
  const vec4 base_color = GetBaseColor();
  const vec2 metallic_roughness = GetMetallicRoughness();
  vec3 radiance_out = vec3(0.0);

  for (int i = 0; i < kLightCount; ++i) {
    const WorldLight world_light = world_lights.data[i];
    float light_attenuation = 0.0;
    const vec3 light_direction = GetLightDirection(world_light, light_attenuation);
    const vec3 radiance_in = light_attenuation * world_light.color.rgb;
    const vec3 material_brdf = GetMaterialBrdf(base_color, metallic_roughness, light_direction, normal, view_direction);
    const float cos_theta = max(dot(normal, light_direction), 0.0);
    radiance_out.rgb += radiance_in * material_brdf * cos_theta;
  }

  fragment_color = vec4(radiance_out, base_color.a);  // TODO: add alpha-mode support
}
