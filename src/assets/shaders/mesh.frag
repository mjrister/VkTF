#version 460

layout(location = 0) in Vertex {
  vec3 position;
  vec3 normal;
  vec2 texture_coordinates;
} vertex;

layout(location = 0) out vec4 fragment_color;

void main() {
  const vec3 normal = normalize(vertex.normal);
  fragment_color = vec4(normal, 1.0);
}
