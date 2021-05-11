
struct VertexOutput {
    [[location(0)]]
    tex_coord: vec2<f32>;
    [[builtin(position)]]
    position: vec4<f32>;
};

[[stage(vertex)]]
fn vs_main(
    [[location(0)]]
    in_pos_vs: vec2<f32>,
    [[location(1)]]
    in_tex_coord_vs: vec2<f32>
) -> VertexOutput {
    var out: VertexOutput;
    out.tex_coord = in_tex_coord_vs;
    out.position = vec4<f32>(in_pos_vs, 0.0, 1.0);
    return out;
}


[[group(0), binding(0)]]
var r_color: texture_2d<f32>;
[[group(0), binding(1)]]
var r_sampler: sampler;

[[group(1), binding(0)]]
var r_gradient: texture_1d<f32>;
[[group(1), binding(1)]]
var r_gradient_sampler: sampler;

[[stage(fragment)]]
fn fs_main(in : VertexOutput) -> [[location(0)]] vec4<f32> {
    // TODO: non-filtered interger sampler?
    var tex: vec4<f32> = textureSample(r_color, r_sampler, in.tex_coord);
    var v: f32 = tex.x;
    var l: f32 = log2(v);

    return textureSample(r_gradient, r_gradient_sampler, l / 100.0);
}
