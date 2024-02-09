
struct VertexOutput {
    @location(0)
    tex_coord: vec2<f32>,
    @builtin(position)
    position: vec4<f32>,
}

@vertex
fn vs_main(
    @location(0)
    instance_matrix_row_0: vec4<f32>,
    @location(1)
    instance_matrix_row_1: vec4<f32>,
    @location(2)
    in_pos_vs: vec2<f32>,
    @location(3)
    in_tex_coord_vs: vec2<f32>,
) -> VertexOutput {
    var instance_matrix: mat2x3<f32> = mat2x3<f32>(instance_matrix_row_0.xyz, instance_matrix_row_1.xyz);

    var out: VertexOutput;
    out.tex_coord = in_tex_coord_vs;
    out.position = vec4<f32>((vec3<f32>(in_pos_vs, 1.0) * instance_matrix), 0.0, 1.0);
    return out;
}



@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    return vec4<f32>(1.0, 0.0, 0.0, 1.0);
}


@group(0) @binding(0)
var r_color: texture_2d<f32>;
@group(0) @binding(1)
var r_sampler: sampler;


@fragment
fn fs_main_textured(
    in: VertexOutput
) -> @location(0) vec4<f32> {
    return textureSample(r_color, r_sampler, in.tex_coord);
}
