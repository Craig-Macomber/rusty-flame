// https://github.com/gfx-rs/wgpu-rs/blob/master/examples/hello-triangle/shader.wgsl
// https://github.com/gfx-rs/wgpu-rs/blob/master/examples/boids/draw.wgsl

[[location(0)]]
var<in> instance_matrix_row_0: vec4<f32>;
[[location(1)]]
var<in> instance_matrix_row_1: vec4<f32>;

[[location(2)]]
var<in> in_pos_vs: vec2<f32>;
[[location(3)]]
var<in> in_tex_coord_vs: vec2<f32>;

[[location(0)]]
var<out> out_tex_coord: vec2<f32>;
[[builtin(position)]]
var<out> out_position: vec4<f32>;

[[stage(vertex)]]
fn vs_main() {
    // var instance_matrix: mat2x3<f32> = mat2x3<f32>(instance_matrix_row_0.xyz, instance_matrix_row_1.xyz);
    // out_position = vec4<f32>((vec3<f32>(in_pos_vs, 1.0) * instance_matrix), 0.0, 1.0);

    out_tex_coord = in_tex_coord_vs;
    out_position = vec4<f32>(
        instance_matrix_row_0.x * in_pos_vs.x + instance_matrix_row_0.y * in_pos_vs.y + instance_matrix_row_0.z,
        instance_matrix_row_1.x * in_pos_vs.x + instance_matrix_row_1.y * in_pos_vs.y + instance_matrix_row_1.z,
        0.0,
        1.0
    );
}

[[location(0)]]
var<in> in_tex_coord_fs: vec2<f32>;
[[location(0)]]
var<out> out_color: vec4<f32>;
// [[group(0), binding(1)]]
// var r_color: texture_2d<f32>;
// [[group(0), binding(2)]]
// var r_sampler: sampler;

[[stage(fragment)]]
fn fs_main() {
    // var tex: vec4<f32> = textureSample(r_color, r_sampler, in_tex_coord_fs);
    // out_color = tex;
    out_color = vec4<f32>(0.01, 0.0, 0.0, 1.0);
}
