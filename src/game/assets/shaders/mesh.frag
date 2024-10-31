#version 460

const float kPi = 3.141592653589793;

const int kBaseColorSamplerIndex = 0;
const int kMetallicRoughnessSamplerIndex = 1;
const int kNormalSamplerIndex = 2;
const int kSamplerCount = 3;

layout(set = 1, binding = 0) uniform sampler2D material_samplers[kSamplerCount];

layout(location = 0) in Vertex {
  vec3 position;
  vec2 texture_coordinates_0;
  mat3 normal_transform;  // TODO(matthew-rister): prefer normal transform matrix multplication in the vertex shader
} vertex;

layout(location = 0) out vec4 fragment_color;

struct PointLight {
  vec3 position;
  vec3 color;
} kPointLight = {
  // TODO(matthew-rister): import lights from a glTF scene using the KHR_lights_punctual extension
  vec3(0.0, 0.0, 0.0),
  vec3(1.0, 1.0, 1.0)
};

vec4 GetImageColor(const uint sampler_index) {
  return texture(material_samplers[sampler_index], vertex.texture_coordinates_0);
}

vec3 GetNormal() {
  vec3 normal = GetImageColor(kNormalSamplerIndex).rgb;
  normal = 2.0 * normal - 1.0;  // convert sampled RGB values from [0, 1] to [-1, 1]
  return normalize(vertex.normal_transform * normal);
}

vec3 GetViewDirection() {
  return normalize(-vertex.position);  // position assumed to be in view space
}

vec3 GetLightDirection(out float light_distance) {
  const vec3 light_direction = kPointLight.position - vertex.position;
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
                     const vec3 halfway_direction, out vec3 base_color) {
  const vec2 metallic_roughness = GetImageColor(kMetallicRoughnessSamplerIndex).bg;
  const float metallic_factor = metallic_roughness.x;
  const float roughness_factor = metallic_roughness.y;

  const float alpha = roughness_factor * roughness_factor;
  const float D = GetMicrofacetDistribution(normal, halfway_direction, alpha);
  const float V = GetMicrofacetVisibility(normal, view_direction, light_direction, halfway_direction, alpha);

  base_color = GetImageColor(kBaseColorSamplerIndex).rgb;
  const vec3 f0 = mix(vec3(0.04), base_color, metallic_factor);
  const vec3 F = GetFresnelApproximation(view_direction, halfway_direction, f0);

  const vec3 specular_brdf = D * V * F;
  const vec3 diffuse_brdf = (1.0 - F) / kPi * mix(base_color, vec3(0.0), metallic_factor);
  return diffuse_brdf + specular_brdf;
}

void main() {
  float light_distance = 0.0;
  const vec3 normal = GetNormal();
  const vec3 view_direction = GetViewDirection();
  const vec3 light_direction = GetLightDirection(light_distance);
  const vec3 halfway_direction = normalize(light_direction + view_direction);

  vec3 base_color = vec3(0.0);
  const vec3 material_brdf = GetMaterialBrdf(normal, view_direction, light_direction, halfway_direction, base_color);

  const float light_attenuation = 1.0 / max(light_distance * light_distance, 1.0);
  const vec3 radiance_in = kPointLight.color * light_attenuation;

  const float cos_theta = max(dot(normal, light_direction), 0.0);
  const vec3 radiance_out = material_brdf * radiance_in * cos_theta;

  const vec3 kAmbiance = vec3(0.03) * base_color;  // HACK
  fragment_color = vec4(kAmbiance + radiance_out, 1.0);
}
