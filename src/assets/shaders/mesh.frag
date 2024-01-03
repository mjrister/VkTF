#version 460

layout (binding = 1, set = 1) uniform sampler2D diffuse_map;

layout(location = 0) in Vertex {
  vec3 position;
  vec3 normal;
  vec2 texture_coordinates;
} vertex;

layout(location = 0) out vec4 fragment_color;

void main() {
  fragment_color = texture(diffuse_map, vertex.texture_coordinates);
}
