use crate::wgpu_ctx::WgpuCtx;
use std::ops::DerefMut;
use std::sync::{Arc, Mutex};
use winit::application::ApplicationHandler;
#[cfg(not(target_arch = "wasm32"))]
use winit::dpi::PhysicalSize;
use winit::event::WindowEvent;
use winit::event_loop::ActiveEventLoop;
use winit::window::{Window, WindowId};

#[derive(Default)]
pub struct App {
    // Use an `Option` to allow the window to not be available until the application is properly running.
    window: Option<Arc<Window>>,
    wgpu_ctx: Arc<Mutex<Option<WgpuCtx>>>,
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.window.is_none() {
            let win_attr = Window::default_attributes().with_title("Rusty Flame");
            #[cfg(not(target_arch = "wasm32"))]
            let win_attr = win_attr.with_inner_size(PhysicalSize::new(1200, 800));

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

            #[cfg(target_arch = "wasm32")]
            {
                let win2 = window.clone();
                let ctx2 = self.wgpu_ctx.clone();
                let fut = wasm_bindgen_futures::spawn_local((async move || {
                    let wgpu_ctx = WgpuCtx::new_async(win2).await;
                    let mut guard = ctx2.lock().unwrap();
                    *guard = Some(wgpu_ctx);
                })());
            }

            #[cfg(not(target_arch = "wasm32"))]
            {
                let wgpu_ctx = pollster::block_on(WgpuCtx::new_async(window));
                let mut guard = self.wgpu_ctx.lock().unwrap();
                *guard = Some(wgpu_ctx);
            }
        }
    }

    fn window_event(
        &mut self,
        event_loop: &ActiveEventLoop,
        _window_id: WindowId,
        event: WindowEvent,
    ) {
        if event == WindowEvent::CloseRequested {
            event_loop.exit();
        }
        let mut guard = self.wgpu_ctx.lock().unwrap();
        if let Some(wgpu_ctx) = guard.deref_mut() {
            wgpu_ctx.window_event(event_loop, event);
        }
    }
}
