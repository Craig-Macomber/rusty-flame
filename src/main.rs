#![warn(unused_extern_crates)]

extern crate nalgebra as na;
use egui::{FontDefinitions, Style};
use egui_wgpu_backend::{RenderPass, ScreenDescriptor};
use egui_winit_platform::{Platform, PlatformDescriptor};
#[cfg(target_arch = "wasm32")]
use wasm_bindgen::prelude::*;

use std::rc::Rc;

use util_types::DebugIt;
use wgpu_render::{render, Inputs, Inputs2};
use winit::{
    dpi::{PhysicalSize, Size},
    event::Event,
    event_loop::{ControlFlow, EventLoop},
    window::Window,
    window::WindowBuilder,
};
#[cfg(target_arch = "wasm32")]
extern crate console_error_panic_hook;
use std::panic;

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

#[cfg(target_arch = "wasm32")]
#[wasm_bindgen]
pub fn wasm_run() {
    main();
}

pub fn main() {
    let event_loop = winit::event_loop::EventLoopBuilder::with_user_event().build();
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

async fn run(event_loop: EventLoop<()>, window: Window) {
    let mut started = wasm_timer::Instant::now();
    let mut frame_count = 0u64;
    let mut recent_frme_rate: f64 = 0.0;

    let size: PhysicalSize<u32> = window.inner_size();
    // Backend "all" does not appear to be preferring VULKAN in wgpu 0.13, so use VULKAN explicitly for now.
    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::default()); //  backend: wgpu::InstanceDescriptor::VULKAN
    dbg!(&instance);
    let surface = unsafe { instance.create_surface(&window) }.unwrap();
    let adapter = instance
        .request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            // Request an adapter which can render to our surface
            compatible_surface: Some(&surface),
            force_fallback_adapter: false,
        })
        .await
        .expect("Failed to find an appropriate adapter");

    dbg!(&adapter.get_info());

    // List features for R32Float (This app depends on R32Float blending)
    let r32features = adapter.get_texture_format_features(wgpu::TextureFormat::R32Float);
    if !r32features
        .flags
        .contains(wgpu::TextureFormatFeatureFlags::FILTERABLE)
    {
        panic!("This app depends on R32Float blending which is not supported")
    }

    let mut limits: wgpu::Limits = wgpu::Limits::default();
    // {
    //     max_texture_dimension_1d: 8192,
    //     max_texture_dimension_2d: 8192,
    //     max_texture_dimension_3d: 2048,
    //     max_texture_array_layers: 256,
    //     max_bind_groups: 4,
    //     max_bindings_per_bind_group: 1000,
    //     max_dynamic_uniform_buffers_per_pipeline_layout: 8,
    //     max_dynamic_storage_buffers_per_pipeline_layout: 4,
    //     max_sampled_textures_per_shader_stage: 16,
    //     max_samplers_per_shader_stage: 16,
    //     max_storage_buffers_per_shader_stage: 8,
    //     max_storage_textures_per_shader_stage: 4,
    //     max_uniform_buffers_per_shader_stage: 12,
    //     max_uniform_buffer_binding_size: 64 << 10,
    //     max_storage_buffer_binding_size: 128 << 20,
    //     max_vertex_buffers: 8,
    //     max_buffer_size: 256 << 20,
    //     max_vertex_attributes: 16,
    //     max_vertex_buffer_array_stride: 2048,
    //     min_uniform_buffer_offset_alignment: 256,
    //     min_storage_buffer_offset_alignment: 256,
    //     max_inter_stage_shader_components: 60,
    //     max_compute_workgroup_storage_size: 16384,
    //     max_compute_invocations_per_workgroup: 256,
    //     max_compute_workgroup_size_x: 256,
    //     max_compute_workgroup_size_y: 256,
    //     max_compute_workgroup_size_z: 64,
    //     max_compute_workgroups_per_dimension: 65535,
    //     max_push_constant_size: 0,
    //     max_non_sampler_bindings: 1_000_000,
    // };

    // Lower limits to work on webgl based on testing in Firefox
    {
        limits.max_compute_workgroups_per_dimension = 0; // 65535,
        limits.max_compute_workgroup_size_z = 0; // 64,
        limits.max_compute_workgroup_size_y = 0; // 256,
        limits.max_compute_workgroup_size_x = 0; //  256,
        limits.max_compute_invocations_per_workgroup = 0; // 256,
        limits.max_compute_workgroup_storage_size = 0; // 16384
        limits.max_storage_buffer_binding_size = 0; // 128 << 20,
        limits.max_storage_textures_per_shader_stage = 0; // 4,
        limits.max_storage_buffers_per_shader_stage = 0; // 8,
        limits.max_dynamic_storage_buffers_per_pipeline_layout = 0; // 4,
    }

    // Create the logical device and command queue
    let (device, queue) = adapter
        .request_device(
            &wgpu::DeviceDescriptor {
                label: None,
                // Enable nonstandard features
                features: wgpu::Features::TEXTURE_ADAPTER_SPECIFIC_FORMAT_FEATURES,
                limits,
            },
            None,
        )
        .await
        .expect("Failed to create device");

    let surface_caps = surface.get_capabilities(&adapter);
    let surface_format = surface_caps
        .formats
        .iter()
        .copied()
        .find(|f| f.is_srgb())
        .unwrap_or(surface_caps.formats[0]);

    let mut surface_config = wgpu::SurfaceConfiguration {
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
        format: surface_format,
        width: size.width,
        height: size.height,
        #[cfg(target_arch = "wasm32")]
        present_mode: wgpu::PresentMode::Fifo,
        #[cfg(not(target_arch = "wasm32"))]
        present_mode: wgpu::PresentMode::Mailbox,
        alpha_mode: wgpu::CompositeAlphaMode::Opaque,
        view_formats: vec![],
    };

    surface.configure(&device, &surface_config);

    // We use the `egui_winit_platform` crate to handle integration with wgpu, and create the runtime context
    let mut egui_platform = Platform::new(PlatformDescriptor {
        physical_width: window.inner_size().width,
        physical_height: window.inner_size().height,
        scale_factor: window.scale_factor(),
        font_definitions: FontDefinitions::default(),
        style: Style::default(),
    });

    // We use the egui_wgpu_backend crate as the render backend.
    let mut egui_rpass = RenderPass::new(&device, surface_format, 1);

    // Display the demo application that ships with egui.
    // let mut demo_app = egui_demo_lib::ColorTest::default();

    // let start_time = Instant::now();

    let mut ui_settings = ui::Settings::default();

    let mut db = wgpu_render::DatabaseStruct::default();
    db.set_config((), ui_settings.clone());
    db.set_window_size_with_durability((), size, salsa::Durability::MEDIUM);
    db.set_device_with_durability((), Rc::new(device), salsa::Durability::HIGH);
    db.set_queue_with_durability((), Rc::new(queue), salsa::Durability::HIGH);
    db.set_swapchain_format_with_durability((), DebugIt(surface_format), salsa::Durability::HIGH);

    event_loop.run(move |event, _, control_flow| {
        // Have the closure take ownership of the resources.
        // `event_loop.run` never returns, therefore we must do this to ensure
        // the resources are properly cleaned up.
        // let _ = (&device, &queue, &instance, &adapter, &renderer);

        *control_flow = ControlFlow::Wait;

        let exclusive = egui_platform.captures_event(&event);
        egui_platform.handle_event(&event);

        match event {
            // Event::WindowEvent {
            //     event: WindowEvent::CursorMoved { position, .. },
            //     ..
            // } => {
            //     let size = window.inner_size();
            //     db.set_cursor(
            //         (),
            //         [
            //             // position.x / f64::from(size.width),
            //             //position.y / f64::from(size.height),
            //             f64::from(ui_settings.angle),
            //             f64::from(ui_settings.angle),
            //         ],
            //     );
            //     window.request_redraw();
            // }
            Event::RedrawRequested(_) => {
                db.set_config((), ui_settings.clone());

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
                    let elapsed = started.elapsed();
                    if elapsed.as_secs_f64() > 0.25 {
                        recent_frme_rate = elapsed.as_secs_f64() / frame_count as f64;
                        started = wasm_timer::Instant::now();
                        frame_count = 0;
                    }

                    let output_view = output_texture
                        .texture
                        .create_view(&wgpu::TextureViewDescriptor::default());

                    // Draw UI
                    egui_platform.begin_frame();

                    ui::update(&egui_platform.context(), &mut ui_settings, recent_frme_rate);

                    // End the UI frame. We could now handle the output and draw the UI with the backend.
                    let output = egui_platform.end_frame(Some(&window));
                    let paint_jobs = egui_platform.context().tessellate(output.shapes);

                    // Upload all resources for the GPU.
                    let screen_descriptor = ScreenDescriptor {
                        physical_width: surface_config.width,
                        physical_height: surface_config.height,
                        scale_factor: window.scale_factor() as f32,
                    };

                    egui_rpass
                        .add_textures(device, queue, &output.textures_delta)
                        .unwrap();
                    egui_rpass.remove_textures(output.textures_delta).unwrap();

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
            Event::MainEventsCleared => {
                if ui_settings.busy_loop {
                    window.request_redraw(); // Enable to busy loop
                }
            }
            Event::WindowEvent { event, .. } => {
                // Ideally we would only request redraw if needed, not on every event,
                // but its unclear how to tell when its safe to skip this.
                window.request_redraw();

                // Pass the winit events to the platform integration.
                // state.on_event returns true when the event has already been handled by egui and shouldn't be passed further
                if !exclusive {
                    match event {
                        winit::event::WindowEvent::Resized(size) => {
                            // Resize with 0 width and height is used by winit to signal a minimize event on Windows.
                            // See: https://github.com/rust-windowing/winit/issues/208
                            // This solves an issue where the app would panic when minimizing on Windows.
                            if size.width > 0 && size.height > 0 {
                                surface_config.width = size.width;
                                surface_config.height = size.height;
                                surface.configure(&db.device(()), &surface_config);
                                db.set_window_size_with_durability(
                                    (),
                                    size,
                                    salsa::Durability::MEDIUM,
                                );
                            }
                        }
                        winit::event::WindowEvent::CloseRequested => {
                            *control_flow = ControlFlow::Exit;
                        }
                        _ => (),
                    }
                }
            }
            _ => {}
        }
    });
}
