#version 460

layout (binding = 1, set = 1) uniform sampler2D base_color_sampler;

layout(location = 0) in Vertex {
  vec3 position;
  vec2 texture_coordinates;
  vec3 normal;
} vertex;

layout(location = 0) out vec4 fragment_color;

// TODO(#64): store light data in a uniform buffer
struct PointLight {
  vec3 position;
  vec3 color;
} kPointLights[] = {
 {{ 2.0,  2.0, -2.0}, {1.0, 1.0, 1.0}},
 {{-1.0, -1.0, -1.0}, {1.0, 1.0, 1.0}},
 {{ 0.0,  4.0, -5.0}, {1.0, 1.0, 1.0}},
};

const vec3 kAmbientColor = vec3(0.001953125);
const float kReflectance = 128.0;
const float kAlphaCutoff = 0.5; // TODO(matthew-rister): avoid hardcoding mask value specified in glTF model

void main() {
  const vec4 base_color = texture(base_color_sampler, vertex.texture_coordinates);
  if (base_color.a < kAlphaCutoff) discard;

  const vec3 normal = normalize(vertex.normal);
  const vec3 view_direction = normalize(-vertex.position);
  vec3 light_color = kAmbientColor;

  for (int i = 0; i < kPointLights.length(); ++i) {
    const PointLight point_light = kPointLights[i];

    vec3 light_direction = point_light.position - vertex.position;
    const float light_distance = length(light_direction);
    const float attenuation = 1.0 / max(light_distance * light_distance, 1.0);
    light_direction = light_direction / light_distance;

    const vec3 reflect_direction = reflect(-light_direction, normal);
    const float diffuse_intensity = max(dot(light_direction, normal), 0.0);
    const float specular_intensity = pow(max(dot(reflect_direction, view_direction), 0.0), kReflectance);

    light_color += (diffuse_intensity + specular_intensity) * attenuation * point_light.color;
  }

  fragment_color = base_color * vec4(light_color, 1.0);
}
