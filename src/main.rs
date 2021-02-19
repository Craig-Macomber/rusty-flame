#![warn(unused_extern_crates)]

extern crate nalgebra as na;

use std::rc::Rc;

use crate::flame::Root;
use na::{Affine2, Point2, Rotation2, Similarity2, Translation2};
use wgpu_render::{render, DebugIt, Renderer};
use winit::{
    dpi::{PhysicalSize, Size},
    event::{Event, WindowEvent},
    event_loop::{ControlFlow, EventLoop},
    window::Window,
    window::WindowBuilder,
};
mod fixed_point;
mod flame;
mod geometry;
mod mesh;
mod plan;
mod wgpu_render;

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

const MIN_SCALE: f64 = 0.5;
const MAX_SCALE: f64 = 0.8;

pub fn get_state(cursor: [f64; 2]) -> Root {
    let n: u32 = 3;
    let shift = 0.5;
    let max_grow = MAX_SCALE - MIN_SCALE;
    let x = f64::min(1.0, f64::max(0.0, cursor[0] as f64));
    let y = f64::min(1.0, f64::max(0.0, cursor[1] as f64));
    let scale = MIN_SCALE + y * max_grow;
    let sm = Similarity2::from_scaling(scale);

    let mut va = (0..n)
        .map(|i| {
            let offset = Rotation2::new(std::f64::consts::PI * 2.0 * f64::from(i) / f64::from(n))
                * Point2::new(shift, 0.0);
            na::convert::<_, Affine2<f64>>(sm * Translation2::new(offset.x, offset.y))
                * Rotation2::new(20.0 * x)
        })
        .collect::<Vec<Affine2<f64>>>();

    va[0] *= Rotation2::new(20.0 * y);
    va[1] *= Rotation2::new(20.0 * x);

    Root::new(va)
}

#[derive(PartialEq, Clone, Copy, Debug)]
pub struct SceneState {
    pub cursor: Point2<f64>,
}

pub async fn run(event_loop: EventLoop<()>, window: Window) {
    let mut started = std::time::Instant::now();
    let mut frame_count = 0u64;

    let size: PhysicalSize<u32> = window.inner_size();
    let instance = wgpu::Instance::new(wgpu::BackendBit::all());
    let surface = unsafe { instance.create_surface(&window) };
    let adapter = instance
        .request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::default(),
            // Request an adapter which can render to our surface
            compatible_surface: Some(&surface),
        })
        .await
        .expect("Failed to find an appropriate adapter");

    // Create the logical device and command queue
    let (device, queue) = adapter
        .request_device(
            &wgpu::DeviceDescriptor {
                label: None,
                features: wgpu::Features::empty(),
                limits: wgpu::Limits::default(),
            },
            None,
        )
        .await
        .expect("Failed to create device");

    let swapchain_format = adapter.get_swap_chain_preferred_format(&surface);

    let mut sc_desc = wgpu::SwapChainDescriptor {
        usage: wgpu::TextureUsage::RENDER_ATTACHMENT,
        format: swapchain_format,
        width: size.width,
        height: size.height,
        present_mode: wgpu::PresentMode::Fifo, // Mailbox
    };

    let mut swap_chain = device.create_swap_chain(&surface, &sc_desc);

    let mut db = wgpu_render::DatabaseStruct::default();
    db.set_cursor((), [0.0, 0.0]);
    db.set_size_with_durability((), size, salsa::Durability::HIGH);
    db.set_device_with_durability((), Rc::new(device), salsa::Durability::HIGH);
    db.set_queue_with_durability((), Rc::new(queue), salsa::Durability::HIGH);
    db.set_swapchain_format_with_durability((), DebugIt(swapchain_format), salsa::Durability::HIGH);

    event_loop.run(move |event, _, control_flow| {
        // Have the closure take ownership of the resources.
        // `event_loop.run` never returns, therefore we must do this to ensure
        // the resources are properly cleaned up.
        // let _ = (&device, &queue, &instance, &adapter, &renderer);

        *control_flow = ControlFlow::Wait;
        match event {
            Event::WindowEvent {
                event: WindowEvent::Resized(size),
                ..
            } => {
                // Recreate the swap chain with the new size
                sc_desc.width = size.width;
                sc_desc.height = size.height;
                swap_chain = db.device(()).create_swap_chain(&surface, &sc_desc);
                db.set_size_with_durability((), size, salsa::Durability::HIGH);
                window.request_redraw();
            }

            Event::WindowEvent {
                event: WindowEvent::CursorMoved { position, .. },
                ..
            } => {
                let size = window.inner_size();
                db.set_cursor(
                    (),
                    [
                        f64::from(position.x) / f64::from(size.width),
                        f64::from(position.y) / f64::from(size.height),
                    ],
                );
                window.request_redraw();
            }

            Event::RedrawRequested(_) => {
                render(
                    &db,
                    &swap_chain
                        .get_current_frame()
                        .expect("Failed to acquire next swap chain texture")
                        .output,
                );
                frame_count += 1;
                if frame_count % 100 == 0 {
                    let elapsed = started.elapsed();
                    log::error!(
                        "Frame Time: {:?}. FPS: {:.3}",
                        elapsed / frame_count as u32,
                        frame_count as f64 / elapsed.as_secs_f64()
                    );
                    started = std::time::Instant::now();
                    frame_count = 0;
                }
            }
            // Event::MainEventsCleared => {
            //     window.request_redraw(); // Enable to busy loop
            // }
            Event::WindowEvent {
                event: WindowEvent::CloseRequested,
                ..
            } => *control_flow = ControlFlow::Exit,
            _ => {}
        }
    });
}
