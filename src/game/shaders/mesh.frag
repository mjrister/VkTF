#version 460

// TODO: use specialized constants to avoid hardcoding values known when creating graphics pipelines
const uint kBaseColorSamplerIndex = 0;
const uint kMetallicRoughnessSamplerIndex = 1;
const uint kNormalSamplerIndex = 2;
const uint kSamplerCount = 3;
const uint kLightCount = 24;
const float kMinLightDistance = 0.1;
const float kPi = 3.141592653589793;

layout(push_constant) uniform PushConstants {
  layout(offset = 64) vec3 camera_position;
} push_constants;

struct Light {
  vec4 position;  // represents light direction when w-component is zero
  vec4 color;
};

layout(set = 0, binding = 1) uniform Lights {
  Light data[kLightCount];
} lights;

layout(set = 1, binding = 0) uniform MaterialProperties {
  vec4 base_color_factor;
  vec2 metallic_roughness_factor;
  float normal_scale;
} material_properties;

layout(set = 1, binding = 1) uniform sampler2D material_samplers[kSamplerCount];

layout(location = 0) in Fragment {
  vec3 position;
  mat3 normal_transform;  // TODO: prefer normal transform matrix multplication in the vertex shader
  vec2 texture_coordinates_0;
  vec4 color_0;
} fragment;

layout(location = 0) out vec4 fragment_color;

vec4 GetSampledImageColor(const uint sampler_index) {
  return texture(material_samplers[sampler_index], fragment.texture_coordinates_0);
}

vec4 GetBaseColor() {
  return fragment.color_0 * material_properties.base_color_factor * GetSampledImageColor(kBaseColorSamplerIndex);
}

vec2 GetMetallicRoughness() {
  return material_properties.metallic_roughness_factor * GetSampledImageColor(kMetallicRoughnessSamplerIndex).bg;
}

vec3 GetViewDirection() {
  return normalize(push_constants.camera_position - fragment.position);
}

vec3 GetNormal() {
  vec3 normal = 2.0 * GetSampledImageColor(kNormalSamplerIndex).rgb - 1.0;  // convert RGB values from [0, 1] to [-1, 1]
  normal.xy *= vec2(material_properties.normal_scale);  // glTF allows scaling sampled normals in the x/y directions
  return normalize(fragment.normal_transform * normal);
}

float GetLightAttenuation(const float light_distance, const float has_position) {
  const float kDirectionalLightAttenuation = 1.0;  // do not attenuate directional lights
  const float point_light_attenuation = 1.0 / (light_distance * light_distance);
  return mix(kDirectionalLightAttenuation, point_light_attenuation, has_position);
}

vec3 GetLightDirection(const Light light, out float light_attenuation) {
  const float has_position = light.position.w;
  const vec3 light_direction = light.position.xyz - (has_position * fragment.position);
  const float light_distance = max(length(light_direction), kMinLightDistance); // avoid division by zero
  light_attenuation = GetLightAttenuation(light_distance, has_position);
  return light_direction / light_distance;
}

float GetMicrofacetVisibility(const vec3 light_direction, const vec3 normal, const vec3 view_direction,
                              const vec3 halfway_direction, const float alpha2) {
  const float h_dot_l = dot(halfway_direction, light_direction);
  const float h_dot_v = dot(halfway_direction, view_direction);
  const float n_dot_l = dot(normal, light_direction);
  const float n_dot_v = dot(normal, view_direction);
  return step(0.0, h_dot_l) / (abs(n_dot_l) + sqrt(alpha2 + (1.0 - alpha2) * n_dot_l * n_dot_l)) *
         step(0.0, h_dot_v) / (abs(n_dot_v) + sqrt(alpha2 + (1.0 - alpha2) * n_dot_v * n_dot_v));
}

float GetMicrofacetDistribution(const vec3 normal, const vec3 halfway_direction, const float alpha2) {
  const float n_dot_h = dot(normal, halfway_direction);
  const float d = n_dot_h * n_dot_h * (alpha2 - 1.0) + 1.0;
  return step(0.0, n_dot_h) * alpha2 / (kPi * d * d);
}

vec3 GetFresnelApproximation(const vec3 view_direction, const vec3 halfway_direction, const vec3 f0) {
  const float h_dot_v = dot(halfway_direction, view_direction);
  return f0 + (1.0 - f0) * pow(1.0 - abs(h_dot_v), 5.0);
}

// implementation based on https://registry.khronos.org/glTF/specs/2.0/glTF-2.0.html#implementation
vec3 GetMaterialBrdf(const vec3 light_direction, const vec3 normal, const vec3 view_direction,
                     const vec4 base_color, const vec2 metallic_roughness) {
  const vec3 halfway_direction = normalize(light_direction + view_direction);

  const float metallic_factor = metallic_roughness.x;
  const vec3 f0 = mix(vec3(0.04), base_color.rgb, metallic_factor);
  const vec3 fresnel_approximation = GetFresnelApproximation(view_direction, halfway_direction, f0);
  const vec3 diffuse_brdf = (1.0 - fresnel_approximation) / kPi * mix(base_color.rgb, vec3(0.0), metallic_factor);

  const float roughness_factor = metallic_roughness.y;
  const float alpha = roughness_factor * roughness_factor;
  const float alpha2 = alpha * alpha;
  const float microfacet_visibility = GetMicrofacetVisibility(light_direction, normal, view_direction,
                                                              halfway_direction, alpha2);
  const float microfacet_distribution = GetMicrofacetDistribution(normal, halfway_direction, alpha2);
  const vec3 specular_brdf = fresnel_approximation * microfacet_visibility * microfacet_distribution;

  return diffuse_brdf + specular_brdf;
}

void main() {
  const vec3 view_direction = GetViewDirection();
  const vec3 normal = GetNormal();
  const vec4 base_color = GetBaseColor();
  const vec2 metallic_roughness = GetMetallicRoughness();
  vec3 radiance_out = vec3(0.0);

  for (int i = 0; i < kLightCount; ++i) {
    const Light light = lights.data[i];
    float light_attenuation = 0.0;
    const vec3 light_direction = GetLightDirection(light, light_attenuation);
    const vec3 radiance_in = light_attenuation * light.color.rgb;
    const vec3 material_brdf = GetMaterialBrdf(light_direction, normal, view_direction, base_color, metallic_roughness);
    const float cos_theta = max(dot(normal, light_direction), 0.0);
    radiance_out.rgb += radiance_in * material_brdf * cos_theta;
  }

  fragment_color = vec4(radiance_out, base_color.a); // TODO: add alpha-mode support
}
