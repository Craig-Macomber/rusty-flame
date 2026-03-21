use winit::event_loop::{ControlFlow, EventLoop};

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

use app::App;
use winit::error::EventLoopError;

mod app;
mod wgpu_ctx;

pub fn main() -> Result<(), EventLoopError> {
    let event_loop = EventLoop::new().unwrap();
    event_loop.set_control_flow(ControlFlow::Poll);
    let mut app = App::default();
    event_loop.run_app(&mut app)
}
