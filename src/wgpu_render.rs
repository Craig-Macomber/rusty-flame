// https://github.com/gfx-rs/wgpu-rs/tree/master/examples/hello-triangle

use nalgebra::{Matrix3, Point2};
use winit::{
    dpi::Size,
    event::{Event, WindowEvent},
    event_loop::{ControlFlow, EventLoop},
    window::{Window, WindowBuilder},
};

use bytemuck::{Pod, Zeroable};
use std::borrow::Cow;
use wgpu::util::DeviceExt;

use crate::{
    flame::{BoundedState, State},
    geometry, get_state, split_levels,
};

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct Vertex {
    _pos: [f32; 2],
    _tex_coord: [f32; 2],
}

type TexCoord = [f32; 2];

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct Instance {
    row0: [f32; 4],
    row1: [f32; 4],
}

pub(crate) struct SceneState {
    cursor: Point2<f64>,
}

async fn run(event_loop: EventLoop<()>, window: Window) {
    let mut scene = SceneState {
        cursor: na::Point2::new(0.0, 0.0),
    };
    let mut started = std::time::Instant::now();
    let mut frame_count = 0u64;

    let size = window.inner_size();
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

    // Load the shaders from disk
    let shader = device.create_shader_module(&wgpu::ShaderModuleDescriptor {
        label: None,
        source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(include_str!("../shaders/wgpu.wgsl"))),
        flags: wgpu::ShaderFlags::all(), //wgpu::ShaderFlags::EXPERIMENTAL_TRANSLATION
    });

    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: None,
        bind_group_layouts: &[],
        push_constant_ranges: &[],
    });

    let swapchain_format = adapter.get_swap_chain_preferred_format(&surface);

    let render_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: None,
        layout: Some(&pipeline_layout),
        vertex: wgpu::VertexState {
            module: &shader,
            entry_point: "vs_main",
            buffers: &[
                wgpu::VertexBufferLayout {
                    array_stride: 2 * 4 * 4,
                    step_mode: wgpu::InputStepMode::Instance,
                    attributes: &wgpu::vertex_attr_array![0 => Float4, 1 => Float4], // Rows of matrix
                },
                wgpu::VertexBufferLayout {
                    array_stride: 2 * 2 * 4,
                    step_mode: wgpu::InputStepMode::Vertex,
                    attributes: &wgpu::vertex_attr_array![2 => Float2, 3 => Float2],
                },
            ],
        },
        fragment: Some(wgpu::FragmentState {
            module: &shader,
            entry_point: "fs_main",
            targets: &[swapchain_format.into()],
        }),
        primitive: wgpu::PrimitiveState::default(),
        depth_stencil: None,
        multisample: wgpu::MultisampleState::default(),
    });

    let mut sc_desc = wgpu::SwapChainDescriptor {
        usage: wgpu::TextureUsage::RENDER_ATTACHMENT,
        format: swapchain_format,
        width: size.width,
        height: size.height,
        present_mode: wgpu::PresentMode::Mailbox,
    };

    let mut swap_chain = device.create_swap_chain(&surface, &sc_desc);

    event_loop.run(move |event, _, control_flow| {
        // Have the closure take ownership of the resources.
        // `event_loop.run` never returns, therefore we must do this to ensure
        // the resources are properly cleaned up.
        let _ = (&instance, &adapter, &shader, &pipeline_layout);

        *control_flow = ControlFlow::Poll;
        match event {
            Event::WindowEvent {
                event: WindowEvent::Resized(size),
                ..
            } => {
                // Recreate the swap chain with the new size
                sc_desc.width = size.width;
                sc_desc.height = size.height;
                swap_chain = device.create_swap_chain(&surface, &sc_desc);
            }

            Event::WindowEvent {
                event: WindowEvent::CursorMoved { position, .. },
                ..
            } => {
                let size = window.inner_size();
                scene = SceneState {
                    cursor: Point2::new(
                        f64::from(position.x) / f64::from(size.width) * 2.0 - 1.0,
                        f64::from(position.y) / f64::from(size.height) * 2.0 - 1.0,
                    ),
                };
                window.request_redraw();
            }

            Event::RedrawRequested(_) => {
                let frame = swap_chain
                    .get_current_frame()
                    .expect("Failed to acquire next swap chain texture")
                    .output;

                let (vertex_data, instance_data) = build_mesh(&scene);

                let vertex_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("Vertex Buffer"),
                    contents: bytemuck::cast_slice(&vertex_data),
                    usage: wgpu::BufferUsage::VERTEX,
                });

                let instance_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("Instance Buffer"),
                    contents: bytemuck::cast_slice(&instance_data),
                    usage: wgpu::BufferUsage::VERTEX,
                });

                let mut encoder =
                    device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
                {
                    let mut rpass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                        label: None,
                        color_attachments: &[wgpu::RenderPassColorAttachmentDescriptor {
                            attachment: &frame.view,
                            resolve_target: None,
                            ops: wgpu::Operations {
                                load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                                store: true,
                            },
                        }],
                        depth_stencil_attachment: None,
                    });
                    rpass.set_pipeline(&render_pipeline);

                    rpass.set_vertex_buffer(0, instance_buf.slice(..));
                    rpass.set_vertex_buffer(1, vertex_buf.slice(..));

                    // TODO: Generate more triangles, and use more instances.
                    rpass.draw(
                        0..(vertex_data.len() as u32),
                        0..(instance_data.len() as u32),
                    );
                }

                queue.submit(Some(encoder.finish()));

                frame_count += 1;
                if frame_count % 100 == 0 {
                    let elapsed = started.elapsed();
                    log::info!(
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

fn build_mesh(scene: &SceneState) -> (Vec<Vertex>, Vec<Instance>) {
    // Treat window as 2*2
    // TODO: is this the right setup?
    let root = get_state([scene.cursor.x + 1.0, scene.cursor.y + 1.0], [2.0, 2.0]);
    let state = root.get_state();
    let bounds = state.get_bounds();
    let root_mat = geometry::letter_box(
        geometry::Rect {
            min: na::Point2::new(-1.0, -1.0),
            max: na::Point2::new(1.0, 1.0),
        },
        bounds,
    );

    let corners = bounds.corners();
    let mut verts: Vec<TexCoord> = vec![];
    let tri_verts = [
        corners[0], corners[1], corners[2], corners[0], corners[2], corners[3],
    ];

    let uv_corners = geometry::Rect {
        min: na::Point2::new(0.0, 0.0),
        max: na::Point2::new(1.0, 1.0),
    }
    .corners();

    let mut uv_verts: Vec<TexCoord> = vec![];
    let uv_tri_verts: Vec<TexCoord> = [
        uv_corners[0],
        uv_corners[1],
        uv_corners[2],
        uv_corners[0],
        uv_corners[2],
        uv_corners[3],
    ]
    .iter()
    .map(|c| [c.x as f32, c.y as f32].into())
    .collect();

    let split = split_levels();

    state.process_levels(split.mesh, &mut |state| {
        for t in &tri_verts {
            let t2 = state.mat * t;
            verts.push([t2.x as f32, t2.y as f32].into());
        }
        uv_verts.extend(uv_tri_verts.iter());
    });

    let verts = verts
        .into_iter()
        .zip(uv_verts.into_iter())
        .map(|(v, u)| Vertex {
            _pos: v,
            _tex_coord: u,
        })
        .collect();

    let mut instances: Vec<Instance> = vec![];
    state.process_levels(split.instance, &mut |state| {
        let m: Matrix3<f64> = (root_mat * state.mat).to_homogeneous();
        let s = m.as_slice();
        instances.push(Instance {
            row0: [s[0] as f32, s[3] as f32, s[6] as f32, 0f32],
            row1: [s[1] as f32, s[4] as f32, s[7] as f32, 0f32],
        });
    });

    return (verts, instances);
}
