extern crate alga;
extern crate glutin_window;
extern crate graphics;
extern crate image as im;
extern crate nalgebra as na;
extern crate opengl_graphics;
extern crate piston;

use crate::flame::process_levels;
use crate::flame::AffineState;
use na::{Affine2, Rotation2, Similarity2, Translation2};

mod flame;
mod piston_render;
mod rendy_render;

fn main() {
    rendy_render::main()
}

pub fn process_scene<F: FnMut(&AffineState)>(
    cursor: [f64; 2],
    draw_size: [f64; 2],
    callback: &mut F,
) {
    let sm = Similarity2::from_scaling(0.5);

    let a: Affine2<f64> = na::convert(sm * Translation2::new(0.25, 0.25));
    let a2: Affine2<f64> = na::convert(sm * Translation2::new(-0.25, 0.25));

    let transforms = vec![
        na::convert(sm * Translation2::new(0.0, -0.25)),
        a2 * Rotation2::new(100.0 * (cursor[1] as f64) / draw_size[1]),
        a * Rotation2::new(100.0 * (cursor[0] as f64) / draw_size[0]),
    ];

    let mat_root = na::convert(
        Translation2::new(cursor[0], cursor[1]) * Similarity2::from_scaling(draw_size[0] as f64),
    );

    let state = AffineState::new(mat_root, &transforms);
    process_levels(8, &state, callback);
}
