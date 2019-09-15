extern crate alga;
extern crate glutin_window;
extern crate graphics;
extern crate image as im;
extern crate nalgebra as na;
extern crate opengl_graphics;
extern crate piston;
extern crate vecmath;

use crate::flame::{AffineState, Root, State};
use na::{Affine2, Point2, Rotation2, Similarity2, Translation2};

mod flame;
mod piston_render;
mod rendy_render;

fn main() {
    piston_render::main()
}

pub fn process_scene<F: FnMut(&AffineState)>(state: AffineState, callback: &mut F) {
    state.process_levels(3, callback);
}

pub fn get_state(cursor: [f64; 2], draw_size: [f64; 2]) -> Root {
    let n: u32 = 3;
    let shift = 0.5;
    let scale = 0.5; // + (cursor[1] as f64) / draw_size[1];
    let sm = Similarity2::from_scaling(scale);

    let mut va = (0..n)
        .map(|i| {
            let offset = Rotation2::new(std::f64::consts::PI * 2.0 * f64::from(i) / f64::from(n))
                * Point2::new(shift, 0.0);
            na::convert::<_, Affine2<f64>>(sm * Translation2::new(offset.x, offset.y))
                * Rotation2::new(20.0 * (cursor[0] as f64) / draw_size[0])
        })
        .collect::<Vec<Affine2<f64>>>();

    va[0] *= Rotation2::new(20.0 * (cursor[1] as f64) / draw_size[1]);
    va[1] *= Rotation2::new(20.0 * (cursor[0] as f64) / draw_size[0]);

    Root { storage: va }
}
