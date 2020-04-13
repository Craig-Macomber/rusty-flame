#version 450
#extension GL_ARB_separate_shader_objects : enable

layout(location = 0) in vec2 uv;
layout(set = 0, binding = 0) uniform sampler tex_sampler;
layout(set = 0, binding = 1) uniform texture2D tex;

layout(location = 0) out vec4 color;

void main() {
    color = texture(sampler2D(tex, tex_sampler), uv);
}