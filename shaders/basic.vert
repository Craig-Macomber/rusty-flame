#version 450
#extension GL_ARB_separate_shader_objects : enable

layout(location = 0) in vec2 vertex_position;
layout(location = 0) out vec2 frag_uv;

void main() {
    gl_Position = vec4(vertex_position, 0.0, 1.0);
    frag_uv = vertex_position * 0.5 + 0.5;
}
