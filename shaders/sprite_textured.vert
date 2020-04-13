#version 450
#extension GL_ARB_separate_shader_objects : enable

layout(location = 0) in vec2 vertex_position;
layout(location = 1) in vec2 vertex_uv;
layout(location = 2) in vec4 sprite_position_1; // per-instance.
layout(location = 3) in vec4 sprite_position_2; // per-instance.

layout(location = 0) out vec2 uv;

void main() {
    mat2x3 sprite_position = mat2x3(sprite_position_1.xyz, sprite_position_2.xyz);
    gl_Position = vec4((vec3(vertex_position, 1) * sprite_position), 0.0, 1.0);
    uv = vertex_uv;
}
