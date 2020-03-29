#version 450
#extension GL_ARB_separate_shader_objects : enable

layout(location = 0) in vec2 vertex_position;
layout(location = 1) in vec4 sprite_position_1; // per-instance.
layout(location = 2) in vec4 sprite_position_2; // per-instance.

void main() {
    mat3x3 sprite_position = mat3x3(sprite_position_1.xyz, sprite_position_2.xyz, vec3(0,0,1));
    gl_Position = vec4((vec3(vertex_position, 1) * sprite_position).xy, 0.0, 1.0);
}
