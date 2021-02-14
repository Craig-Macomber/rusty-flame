#![warn(unused_extern_crates)]

extern crate nalgebra as na;

use crate::flame::{AffineState, Root, State};
use na::{Affine2, Point2, Rotation2, Similarity2, Translation2};
use wgpu_render::run;
use winit::{dpi::Size, event_loop::EventLoop, window::WindowBuilder};

mod fixed_point;
mod flame;
pub mod geometry;
mod mesh;
pub mod wgpu_render;

pub fn main() {
    let event_loop = EventLoop::new();
    let window = WindowBuilder::new()
        .with_inner_size(Size::Physical((3000, 2000).into()))
        .with_title("Rusty Flame")
        .build(&event_loop)
        .unwrap();

    #[cfg(not(target_arch = "wasm32"))]
    {
        wgpu_subscriber::initialize_default_subscriber(None);
        // Temporarily avoid srgb formats for the swapchain on the web
        pollster::block_on(run(event_loop, window));
    }
    #[cfg(target_arch = "wasm32")]
    {
        std::panic::set_hook(Box::new(console_error_panic_hook::hook));
        console_log::init().expect("could not initialize logger");
        use winit::platform::web::WindowExtWebSys;
        // On wasm, append the canvas to the document body
        web_sys::window()
            .and_then(|win| win.document())
            .and_then(|doc| doc.body())
            .and_then(|body| {
                body.append_child(&web_sys::Element::from(window.canvas()))
                    .ok()
            })
            .expect("couldn't append canvas to document body");
        wasm_bindgen_futures::spawn_local(run(event_loop, window));
    }
}

pub const LEVELS: u32 = 10;
pub const SMALL_ACCUMULATION_BUFFER_SIZE: u32 = 256;

pub fn process_scene<F: FnMut(&AffineState)>(state: AffineState, callback: &mut F) {
    state.process_levels(LEVELS, callback);
}

pub fn get_state(cursor: [f64; 2], draw_size: [f64; 2]) -> Root<Affine2<f64>> {
    let n: u32 = 3;
    let shift = 0.5;
    let scale = 0.5 + f64::min((cursor[1] as f64) / draw_size[1] * 0.2, 0.2);
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

pub struct LevelSplit {
    mesh: u32,
    instance: u32,
}

pub fn split_levels() -> LevelSplit {
    let instance = LEVELS / 2;
    LevelSplit {
        instance,
        mesh: LEVELS - instance,
    }
}
