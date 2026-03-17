// https://github.com/w4ngzhen/wgpu_winit_example/blob/main/ch01_render_in_window/src/wgpu_ctx.rs

use egui::{FontDefinitions, Style};
use egui_wgpu_backend::{RenderPass, ScreenDescriptor};
use egui_winit_platform::PlatformDescriptor;
use std::{rc::Rc, sync::Arc};
use wasm_timer::Instant;
use wgpu::MemoryHints::Performance;
use wgpu::Trace;
use winit::{
    event::{ElementState, KeyEvent, WindowEvent},
    event_loop::{ActiveEventLoop, ControlFlow},
    keyboard::{Key, NamedKey},
    window::{self, Window},
};

use crate::{
    ui,
    util_types::DebugIt,
    wgpu_render::{self, render, Inputs, Inputs2},
};

pub struct WgpuCtx {
    surface: wgpu::Surface<'static>,
    surface_config: wgpu::SurfaceConfiguration,
    // adapter: wgpu::Adapter,
    // device: wgpu::Device,
    // queue: wgpu::Queue,
    started: Instant,
    frame_count: u64,
    recent_frame_rate: f64,
    ui_settings: ui::Settings,
    egui_platform: egui_winit_platform::Platform,
    egui_rpass: RenderPass,

    window: Arc<Window>,

    db: wgpu_render::DatabaseStruct,
}

impl WgpuCtx {
    pub async fn new_async(window: Arc<Window>) -> WgpuCtx {
        let instance = wgpu::Instance::default();
        let surface = instance.create_surface(Arc::clone(&window)).unwrap();
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::default(),
                force_fallback_adapter: false,
                // Request an adapter which can render to our surface
                compatible_surface: Some(&surface),
            })
            .await
            .expect("Failed to find an appropriate adapter");

        // List features for R32Float (This app depends on R32Float blending)
        let r32features = adapter.get_texture_format_features(wgpu::TextureFormat::R32Float);
        if !r32features
            .flags
            .contains(wgpu::TextureFormatFeatureFlags::FILTERABLE)
        {
            panic!("This app depends on R32Float filtering which is not supported")
        }

        let mut features = wgpu::Features::empty();
        // Enable nonstandard features
        features |= wgpu::Features::TEXTURE_ADAPTER_SPECIFIC_FORMAT_FEATURES;
        features |= wgpu::Features::FLOAT32_FILTERABLE;

        // Create the logical device and command queue
        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor {
                label: None,
                required_features: features,
                // Make sure we use the texture resolution limits from the adapter, so we can support images the size of the swapchain.
                required_limits: wgpu::Limits::downlevel_webgl2_defaults()
                    .using_resolution(adapter.limits()),
                experimental_features: Default::default(),
                memory_hints: Performance,
                trace: Trace::Off,
            })
            .await
            .expect("Failed to create device");

        let size = window.inner_size();
        let width = size.width.max(1);
        let height = size.height.max(1);
        // let surface_config = surface.get_default_config(&adapter, width, height).unwrap();

        let surface_caps = surface.get_capabilities(&adapter);
        let surface_format = surface_caps
            .formats
            .iter()
            .copied()
            .find(|f| f.is_srgb())
            .unwrap_or(surface_caps.formats[0]);

        let surface_config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format: surface_format,
            width,
            height,
            #[cfg(target_arch = "wasm32")]
            present_mode: wgpu::PresentMode::Fifo,
            #[cfg(not(target_arch = "wasm32"))]
            present_mode: wgpu::PresentMode::Mailbox,
            alpha_mode: wgpu::CompositeAlphaMode::Opaque,
            view_formats: vec![],
            desired_maximum_frame_latency: 2,
        };

        surface.configure(&device, &surface_config);

        // We use the `egui_winit_platform` crate to handle integration with wgpu, and create the runtime context
        let egui_platform = egui_winit_platform::Platform::new(PlatformDescriptor {
            physical_width: size.width as u32,
            physical_height: size.height as u32,
            scale_factor: window.scale_factor(),
            font_definitions: FontDefinitions::default(),
            style: Style::default(),
        });

        // We use the egui_wgpu_backend crate as the render backend.
        let egui_rpass = RenderPass::new(&device, surface_format, 1);

        // Display the demo application that ships with egui.
        // let mut demo_app = egui_demo_lib::ColorTest::default();

        // let start_time = Instant::now();

        let ui_settings = ui::Settings::default();

        let mut db = wgpu_render::DatabaseStruct::default();
        db.set_config((), ui_settings.clone());
        db.set_window_size_with_durability((), size, salsa::Durability::MEDIUM);
        db.set_device_with_durability((), Rc::new(device), salsa::Durability::HIGH);
        db.set_queue_with_durability((), Rc::new(queue), salsa::Durability::HIGH);
        db.set_swapchain_format_with_durability(
            (),
            DebugIt(surface_format),
            salsa::Durability::HIGH,
        );

        WgpuCtx {
            surface,
            surface_config,
            // adapter,
            db,
            ui_settings,
            egui_platform,
            started: wasm_timer::Instant::now(),
            frame_count: Default::default(),
            recent_frame_rate: Default::default(),
            egui_rpass,
            window,
        }
    }

    pub fn new(window: Arc<Window>) -> WgpuCtx {
        pollster::block_on(WgpuCtx::new_async(window))
    }

    fn resize(&mut self, new_size: (u32, u32)) {
        let (width, height) = new_size;
        self.surface_config.width = width.max(1);
        self.surface_config.height = height.max(1);
        self.surface
            .configure(&self.db.device(()), &self.surface_config);
        self.db.set_window_size_with_durability(
            (),
            (self.surface_config.width, self.surface_config.height).into(),
            salsa::Durability::MEDIUM,
        );
    }

    fn draw(&mut self) {
        self.frame_count += 1;
        let elapsed = self.started.elapsed();
        if elapsed.as_secs_f64() > 0.25 {
            self.recent_frame_rate = elapsed.as_secs_f64() / self.frame_count as f64;
            self.started = wasm_timer::Instant::now();
            self.frame_count = 0;
        }

        self.db.set_config((), self.ui_settings.clone());

        let device = &mut self.db.device(());
        let queue = &mut self.db.queue(());
        let output_texture = self
            .surface
            .get_current_texture()
            .expect("Failed to acquire next swap chain texture");
        let mut encoder =
            device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        {
            // TODO: ENABLE
            render(&self.db, &output_texture, &mut encoder);

            let output_view = output_texture
                .texture
                .create_view(&wgpu::TextureViewDescriptor::default());

            // Draw UI
            self.egui_platform.begin_pass();

            ui::update(
                &self.egui_platform.context(),
                &mut self.ui_settings,
                self.recent_frame_rate,
            );

            // End the UI frame. We could now handle the output and draw the UI with the backend.
            let output = self.egui_platform.end_pass(Some(&self.window));
            let paint_jobs = self.egui_platform.context().tessellate(output.shapes, 1.0);

            // Upload all resources for the GPU.
            let screen_descriptor = ScreenDescriptor {
                physical_width: self.surface_config.width,
                physical_height: self.surface_config.height,
                scale_factor: self.window.scale_factor() as f32,
            };

            self.egui_rpass
                .add_textures(device, queue, &output.textures_delta)
                .unwrap();
            self.egui_rpass
                .remove_textures(output.textures_delta)
                .unwrap();

            self.egui_rpass
                .update_buffers(device, queue, &paint_jobs, &screen_descriptor);

            // Record all render passes.
            self.egui_rpass
                .execute(
                    &mut encoder,
                    &output_view,
                    &paint_jobs,
                    &screen_descriptor,
                    None,
                )
                .unwrap();
        }

        self.db.queue(()).submit(Some(encoder.finish()));

        output_texture.present()
    }

    pub fn window_event(&mut self, active_event_loop: &ActiveEventLoop, event: WindowEvent) {
        active_event_loop.set_control_flow(ControlFlow::Wait);

        let exclusive = self.egui_platform.captures_event(&event);
        self.egui_platform.handle_event(&event);

        // TODO: don't do this for everything.
        self.window.request_redraw();

        if !exclusive {
            match event {
                WindowEvent::Resized(new_size) => {
                    self.resize((new_size.width, new_size.height));
                    self.window.request_redraw();
                }
                WindowEvent::KeyboardInput {
                    event:
                        KeyEvent {
                            repeat: false,
                            state: ElementState::Pressed,
                            logical_key: Key::Named(NamedKey::Enter),
                            ..
                        },
                    ..
                } => {
                    let size = self.window.inner_size();
                    let w = size.width.max(1);
                    let h = size.height.max(1);
                    // if self.flag {
                    //     wgpu_ctx.resize((w, h));
                    // } else {
                    //     wgpu_ctx.resize((w / 2, h / 2));
                    // }
                    // self.flag = !self.flag;
                    self.window.request_redraw();
                }
                WindowEvent::RedrawRequested => {
                    self.draw();
                }
                _ => (),
            }
        }
    }
}
