
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
    // var instance_matrix: mat2x3<f32> = mat2x3<f32>(instance_matrix_row_0.xyz, instance_matrix_row_1.xyz);
    // out_position = vec4<f32>((vec3<f32>(in_pos_vs, 1.0) * instance_matrix), 0.0, 1.0);
    var out: VertexOutput;
    out.tex_coord = in_tex_coord_vs;
    out.position = vec4<f32>(
        instance_matrix_row_0.x * in_pos_vs.x + instance_matrix_row_0.y * in_pos_vs.y + instance_matrix_row_0.z,
        instance_matrix_row_1.x * in_pos_vs.x + instance_matrix_row_1.y * in_pos_vs.y + instance_matrix_row_1.z,
        0.0,
        1.0
    );
    return out;
}



@fragment
fn fs_main(in : VertexOutput) -> @location(0) vec4<f32> {
    return vec4<f32>(1.0, 0.0, 0.0, 1.0);
}


@group(0) @binding(0)
var r_color: texture_2d<f32>;
@group(0) @binding(1)
var r_sampler: sampler;


@fragment
fn fs_main_textured(
    in : VertexOutput
) -> @location(0) vec4<f32>  {
    var tex: vec4<f32> = textureSample(r_color, r_sampler, in.tex_coord);
    return tex;
}
