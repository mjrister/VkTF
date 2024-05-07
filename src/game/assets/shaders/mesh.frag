#version 460

layout(binding = 0, set = 0) uniform sampler2D base_color_sampler;

layout(location = 0) in Vertex {
  vec3 position;
  vec3 normal;
  vec2 texture_coordinates;
} vertex;

layout(location = 0) out vec4 fragment_color;

void main() {
  fragment_color = texture(base_color_sampler, vertex.texture_coordinates);
}
