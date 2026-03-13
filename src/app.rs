// https://github.com/w4ngzhen/wgpu_winit_example/blob/main/ch01_render_in_window/src/app.rs

use crate::wgpu_ctx::WgpuCtx;
use std::sync::Arc;
use winit::application::ApplicationHandler;
use winit::dpi::PhysicalSize;
use winit::event::WindowEvent;
use winit::event_loop::ActiveEventLoop;
use winit::window::{Window, WindowId};

pub struct App<'window> {
    // Use an `Option` to allow the window to not be available until the
    // application is properly running.
    window: Option<Arc<Window>>,
    wgpu_ctx: Option<WgpuCtx<'window>>,
}

impl<'t> Default for App<'t> {
    fn default() -> Self {
        Self {
            window: Default::default(),
            wgpu_ctx: Default::default(),
        }
    }
}

impl<'window> ApplicationHandler for App<'window> {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.window.is_none() {
            let win_attr = Window::default_attributes()
                .with_title("Rusty Flame")
                .with_inner_size(PhysicalSize::new(1200, 800));

            // On wasm, attach to the `main-canvas` element.
            #[cfg(target_arch = "wasm32")]
            let win_attr = {
                let doc = web_sys::window().and_then(|win| win.document()).unwrap();

                let canvas = doc.get_element_by_id("main-canvas").unwrap();
                let canvas =
                    wasm_bindgen::JsCast::dyn_into::<web_sys::HtmlCanvasElement>(canvas).unwrap();

                winit::platform::web::WindowAttributesExtWebSys::with_canvas(win_attr, Some(canvas))
            };

            let window = Arc::new(
                event_loop
                    .create_window(win_attr)
                    .expect("create window err."),
            );
            self.window = Some(window.clone());
            let wgpu_ctx = WgpuCtx::new(window.clone());
            self.wgpu_ctx = Some(wgpu_ctx);
        }
    }

    fn window_event(
        &mut self,
        event_loop: &ActiveEventLoop,
        _window_id: WindowId,
        event: WindowEvent,
    ) {
        match event {
            WindowEvent::CloseRequested => {
                event_loop.exit();
            }
            _ => (),
        }
        if let Some(wgpu_ctx) = self.wgpu_ctx.as_mut() {
            wgpu_ctx.window_event(event_loop, event);
        }
    }
}
