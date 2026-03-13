#![warn(unused_extern_crates)]

extern crate nalgebra as na;
#[cfg(target_arch = "wasm32")]
use wasm_bindgen::prelude::*;

use winit::event_loop::{ControlFlow, EventLoop};
#[cfg(target_arch = "wasm32")]
extern crate console_error_panic_hook;

mod accumulate;
pub mod fixed_point;
mod flame;
pub mod geometry;
mod mesh;
mod postprocess;
mod render_common;
mod ui;
mod util_types;
mod wgpu_render;

use crate::app::App;
use winit::error::EventLoopError;

mod app;
mod wgpu_ctx;

fn main() -> Result<(), EventLoopError> {
    let event_loop = EventLoop::new().unwrap();
    event_loop.set_control_flow(ControlFlow::Poll);
    let mut app = App::default();
    event_loop.run_app(&mut app)
}

#[cfg(target_arch = "wasm32")]
#[wasm_bindgen]
pub fn wasm_run() {
    std::panic::set_hook(Box::new(console_error_panic_hook::hook));
    console_log::init().expect("could not initialize logger");
    main().unwrap();
}
