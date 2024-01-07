#version 460

layout (binding = 1, set = 1) uniform sampler2D diffuse_map;

layout(location = 0) in Vertex {
  vec3 position;
  vec3 normal;
  vec2 texture_coordinates;
} vertex;

layout(location = 0) out vec4 fragment_color;

struct PointLight {
  vec3 position;
  vec3 color;
} kPointLights[] = {
 {{ 2.0,  2.0, -2.0}, {1.0, 1.0, 1.0}},
 {{-1.0, -1.0, -1.0}, {1.0, 1.0, 1.0}},
 {{ 0.0,  4.0, -5.0}, {1.0, 1.0, 1.0}},
};

const float kReflectance = 128.0;
const vec3 kAmbientColor = vec3(0.01);

void main() {
  const vec3 normal = normalize(vertex.normal);
  const vec3 view_direction = normalize(-vertex.position);
  fragment_color = vec4(kAmbientColor, 1.0f);

  for (int i = 0; i < kPointLights.length(); ++i) {
    const PointLight point_light = kPointLights[i];

    vec3 light_direction = point_light.position - vertex.position;
    const float light_distance = length(light_direction);
    const float attenuation = 1.0 / max(light_distance * light_distance, 1.0);
    light_direction = light_direction / light_distance;

    const vec3 reflect_direction = reflect(-light_direction, normal);
    const float diffuse_intensity = max(dot(light_direction, normal), 0.0);
    const float specular_intensity = pow(max(dot(reflect_direction, view_direction), 0.0), kReflectance);

    fragment_color += vec4((diffuse_intensity + specular_intensity) * attenuation * point_light.color, 0.0);
  }

  fragment_color *= texture(diffuse_map, vertex.texture_coordinates);
}
