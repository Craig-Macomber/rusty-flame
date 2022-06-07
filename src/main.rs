#![warn(unused_extern_crates)]

extern crate nalgebra as na;

use std::time::Instant;

use egui::FontDefinitions;
use egui_wgpu_backend::{RenderPass, ScreenDescriptor};
use egui_winit_platform::{Platform, PlatformDescriptor};
use winit::event::Event::*;

use std::rc::Rc;

use crate::flame::Root;
use na::{Affine2, Point2, Rotation2, Similarity2, Translation2};
use util_types::DebugIt;
use wgpu_render::{render, Inputs, Inputs2};
use winit::{
    dpi::{PhysicalSize, Size},
    event::{Event, WindowEvent},
    event_loop::{ControlFlow, EventLoop},
    window::Window,
    window::WindowBuilder,
};
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

/// A custom event type for the winit app.
enum Event2 {
    RequestRedraw,
}

/// This is the repaint signal type that egui needs for requesting a repaint from another thread.
/// It sends the custom RequestRedraw event to the winit event loop.
struct ExampleRepaintSignal(std::sync::Mutex<winit::event_loop::EventLoopProxy<Event2>>);

impl epi::RepaintSignal for ExampleRepaintSignal {
    fn request_repaint(&self) {
        self.0
            .lock()
            .unwrap()
            .send_event(Event2::RequestRedraw)
            .ok();
    }
}

pub fn main() {
    let event_loop = winit::event_loop::EventLoop::with_user_event();
    let window = WindowBuilder::new()
        .with_inner_size(Size::Physical((3000, 2000).into()))
        .with_title("Rusty Flame")
        .build(&event_loop)
        .unwrap();

    #[cfg(not(target_arch = "wasm32"))]
    {
        // wgpu_subscriber::initialize_default_subscriber(None);
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

async fn run(event_loop: EventLoop<Event2>, window: Window) {
    let mut started = std::time::Instant::now();
    let mut frame_count = 0u64;

    let size: PhysicalSize<u32> = window.inner_size();
    let instance = wgpu::Instance::new(wgpu::Backends::all());
    let surface = unsafe { instance.create_surface(&window) };
    let adapter = instance
        .request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::default(),
            // Request an adapter which can render to our surface
            compatible_surface: Some(&surface),
            force_fallback_adapter: false,
        })
        .await
        .expect("Failed to find an appropriate adapter");

    // List features for R32Float (This app depends on R32Float blending)
    let r32features = adapter.get_texture_format_features(wgpu::TextureFormat::R32Float);
    if !r32features.filterable {
        panic!("This app depends on R32Float blending which is not supported")
    }

    // Create the logical device and command queue
    let (device, queue) = adapter
        .request_device(
            &wgpu::DeviceDescriptor {
                label: None,
                // Enable nonstandard features
                features: wgpu::Features::TEXTURE_ADAPTER_SPECIFIC_FORMAT_FEATURES,
                limits: wgpu::Limits::default(),
            },
            None,
        )
        .await
        .expect("Failed to create device");

    let surface_format = surface.get_preferred_format(&adapter).unwrap();

    let mut surface_config = wgpu::SurfaceConfiguration {
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
        format: surface_format,
        width: size.width,
        height: size.height,
        present_mode: wgpu::PresentMode::Mailbox,
    };

    surface.configure(&device, &surface_config);

    // We use the egui_winit_platform crate as the platform.
    let mut platform = Platform::new(PlatformDescriptor {
        physical_width: size.width as u32,
        physical_height: size.height as u32,
        scale_factor: window.scale_factor(),
        font_definitions: FontDefinitions::default(),
        style: Default::default(),
    });

    // We use the egui_wgpu_backend crate as the render backend.
    let mut egui_rpass = RenderPass::new(&device, surface_format, 1);

    // Display the demo application that ships with egui.
    // let mut demo_app = egui_demo_lib::ColorTest::default();

    let start_time = Instant::now();

    let mut db = wgpu_render::DatabaseStruct::default();
    db.set_cursor((), [0.0, 0.0]);
    db.set_window_size_with_durability((), size, salsa::Durability::MEDIUM);
    db.set_device_with_durability((), Rc::new(device), salsa::Durability::HIGH);
    db.set_queue_with_durability((), Rc::new(queue), salsa::Durability::HIGH);
    db.set_swapchain_format_with_durability((), DebugIt(surface_format), salsa::Durability::HIGH);

    let mut ui_settings = ui::Settings {
        angle: 0.5f32,
        levels: 5,
    };

    event_loop.run(move |event, _, control_flow| {
        // Pass the winit events to the platform integration.
        platform.handle_event(&event);
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
                // Reconfigure the surface with the new size
                surface_config.width = size.width;
                surface_config.height = size.height;
                surface.configure(&db.device(()), &surface_config);
                db.set_window_size_with_durability((), size, salsa::Durability::MEDIUM);
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
                        // position.x / f64::from(size.width),
                        //position.y / f64::from(size.height),
                        f64::from(ui_settings.angle),
                        f64::from(ui_settings.angle),
                    ],
                );
                window.request_redraw();
            }
            UserEvent(Event2::RequestRedraw) => {
                window.request_redraw();
            }
            Event::RedrawRequested(_) => {
                let device = &mut db.device(());
                let queue = &mut db.queue(());
                let output_texture = surface
                    .get_current_texture()
                    .expect("Failed to acquire next swap chain texture");
                let mut encoder =
                    device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
                {
                    render(&db, &output_texture, &mut encoder);

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

                    platform.update_time(start_time.elapsed().as_secs_f64());

                    let output_view = output_texture
                        .texture
                        .create_view(&wgpu::TextureViewDescriptor::default());

                    // Draw UI
                    platform.begin_frame();
                    ui::update(&platform.context(), &mut ui_settings);

                    // End the UI frame. We could now handle the output and draw the UI with the backend.
                    let (_output, paint_commands) = platform.end_frame(Some(&window));
                    let paint_jobs = platform.context().tessellate(paint_commands);

                    // Upload all resources for the GPU.
                    let screen_descriptor = ScreenDescriptor {
                        physical_width: surface_config.width,
                        physical_height: surface_config.height,
                        scale_factor: window.scale_factor() as f32,
                    };

                    egui_rpass.update_texture(device, queue, &platform.context().texture());
                    egui_rpass.update_user_textures(device, queue);

                    egui_rpass.update_buffers(device, queue, &paint_jobs, &screen_descriptor);

                    // Record all render passes.
                    egui_rpass
                        .execute(
                            &mut encoder,
                            &output_view,
                            &paint_jobs,
                            &screen_descriptor,
                            None,
                        )
                        .unwrap();
                }

                db.queue(()).submit(Some(encoder.finish()));

                output_texture.present()
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
