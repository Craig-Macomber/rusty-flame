#version 450
#extension GL_ARB_separate_shader_objects : enable

layout(location = 0) out vec4 color;

layout(set = 0, binding = 0) uniform sampler tex_sampler;
layout(set = 0, binding = 1) uniform texture2D hdr_tex;

void main() {
    ivec2 uv = ivec2(gl_FragCoord.xy);
    float v = texelFetch(sampler2D(hdr_tex, tex_sampler), uv, 0).r;
    float l = log2(v) / 3.0;
    if ( l > 3) {
        l = l - 3;
        color = vec4(1 - l, 2 - l, 3 - l, 1.0);
    } else {
        color = vec4(l, l - 1, l - 2, 1.0);
    }
}
