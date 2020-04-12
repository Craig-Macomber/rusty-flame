#version 450
#extension GL_ARB_separate_shader_objects : enable

layout(location = 0) in vec2 frag_uv;
layout(location = 0) out vec4 color;

layout(set = 0, binding = 0) uniform sampler tex_sampler;
layout(set = 0, binding = 1) uniform texture2D hdr_tex;

void main() {
    float v = texture(sampler2D(hdr_tex, tex_sampler), frag_uv.xy).r;
    float l = log2(v) / 5;
    color = vec4(l + 1, l, l - 1, 1.0);
}
