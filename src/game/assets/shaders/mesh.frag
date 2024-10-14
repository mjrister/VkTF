#version 460

const int kBaseColorSamplerIndex = 0;
const int kMetallicRougnessSamplerIndex = 1;
const int kMaxSamplers = 2;

layout(set = 0, binding = 0) uniform sampler2D material_samplers[kMaxSamplers];

layout(location = 0) in Vertex {
  vec3 position;
  vec3 normal;
  vec2 texture_coordinates_0;
} vertex;

layout(location = 0) out vec4 fragment_color;

struct PointLight {
  vec3 position;
  vec3 color;
} kPointLight = {
  vec3(0.0, 0.0, 0.0),
  vec3(1.0, 1.0, 1.0)
};

const vec3 kAmbientColor = vec3(0.001);
const float kShininess = 64.0;

void main() {
  vec3 light_direction = kPointLight.position - vertex.position;
  const float light_distance = length(light_direction);
  const float attenuation = 1.0 / max(light_distance, 1.0); // TODO(matthew-rister): use quadratic attenuation
  light_direction /= light_distance;

  const vec3 normal = normalize(vertex.normal);
  const float diffuse_intensity = max(dot(normal, light_direction), 0.0);

  const vec3 view_direction = normalize(-vertex.position);
  const vec3 reflect_direction = reflect(-light_direction, normal);
  const float specular_intensity = pow(max(dot(reflect_direction, view_direction), 0.0), kShininess);

  const vec3 light_color = (diffuse_intensity + specular_intensity) * attenuation * kPointLight.color;
  const vec4 base_color = texture(material_samplers[kBaseColorSamplerIndex], vertex.texture_coordinates_0);
  fragment_color = vec4(kAmbientColor, 0.0) + vec4(light_color * base_color.rgb, base_color.a);
}
