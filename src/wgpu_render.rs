use nalgebra::{Matrix3, Point2};
use winit::{
    dpi::Size,
    event::{Event, WindowEvent},
    event_loop::{ControlFlow, EventLoop},
    window::{Window, WindowBuilder},
};

use bytemuck::{Pod, Zeroable};
use std::borrow::Cow;
use wgpu::{util::DeviceExt, BindGroup, Extent3d, TextureFormat};

use crate::{
    flame::{BoundedState, State},
    geometry, get_state, split_levels, SMALL_ACCUMULATION_BUFFER_SIZE,
};

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct Vertex {
    position: Position,
    texture_coordinate: TextureCoordinate,
}

type TextureCoordinate = [f32; 2];

type Position = [f32; 2];

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct Instance {
    row0: [f32; 4],
    row1: [f32; 4],
}

pub(crate) struct SceneState {
    cursor: Point2<f64>,
}

struct AccumulatePass {
    pipeline: wgpu::RenderPipeline,
    bind_group: Option<wgpu::BindGroup>,
    view: wgpu::TextureView,
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
        label: Some("wgpu.wgsl"),
        source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(include_str!("../shaders/wgpu.wgsl"))),
        flags: wgpu::ShaderFlags::all(),
    });

    let swapchain_format = adapter.get_swap_chain_preferred_format(&surface);

    let vertex_shader = wgpu::VertexState {
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
    };

    let blend_add = wgpu::BlendState {
        src_factor: wgpu::BlendFactor::One,
        dst_factor: wgpu::BlendFactor::One,
        operation: wgpu::BlendOperation::Add,
    };

    // TODO: mipmap filtering and generation
    let accumulation_sampler: wgpu::Sampler = device.create_sampler(&wgpu::SamplerDescriptor {
        label: Some("accumulation sampler"),
        mag_filter: wgpu::FilterMode::Linear,
        min_filter: wgpu::FilterMode::Linear,
        ..Default::default()
    });

    let accumulation_bind_group_layout =
        device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStage::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        multisampled: false,
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStage::FRAGMENT,
                    ty: wgpu::BindingType::Sampler {
                        comparison: false,
                        filtering: true,
                    },
                    count: None,
                },
            ],
            label: None,
        });

    let make_bind_group = |p: &AccumulatePass| -> BindGroup {
        device.create_bind_group(&wgpu::BindGroupDescriptor {
            layout: &accumulation_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&p.view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(&accumulation_sampler),
                },
            ],
            label: None,
        })
    };

    let make_accumulation_pass =
        |width, height, name, previous: Option<&AccumulatePass>| -> AccumulatePass {
            // Create bind group
            let bind_group = match previous {
                Some(p) => Some(make_bind_group(p)),
                None => None,
            };

            let groups = &[&accumulation_bind_group_layout];
            let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: None,
                bind_group_layouts: match bind_group {
                    Some(_) => groups,
                    None => &[],
                },
                push_constant_ranges: &[],
            });

            let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some(name),
                layout: Some(&pipeline_layout),
                vertex: vertex_shader.clone(),
                fragment: Some(wgpu::FragmentState {
                    module: &shader,
                    entry_point: match previous {
                        Some(_) => "fs_main_textured",
                        None => "fs_main",
                    },
                    targets: &[wgpu::ColorTargetState {
                        format: TextureFormat::R32Float,
                        color_blend: blend_add.clone(),
                        alpha_blend: blend_add.clone(),
                        write_mask: wgpu::ColorWrite::ALL,
                    }],
                }),
                primitive: wgpu::PrimitiveState::default(),
                depth_stencil: None,
                multisample: wgpu::MultisampleState::default(),
            });

            let texture: wgpu::Texture = device.create_texture(&wgpu::TextureDescriptor {
                size: Extent3d {
                    width,
                    height,
                    depth: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: TextureFormat::R32Float,
                usage: wgpu::TextureUsage::RENDER_ATTACHMENT | wgpu::TextureUsage::SAMPLED,
                label: Some(name),
            });

            let view: wgpu::TextureView =
                texture.create_view(&wgpu::TextureViewDescriptor::default());

            AccumulatePass {
                pipeline,
                bind_group,
                view,
            }
        };

    let small_pass = make_accumulation_pass(
        SMALL_ACCUMULATION_BUFFER_SIZE,
        SMALL_ACCUMULATION_BUFFER_SIZE,
        "small",
        None,
    );
    let large_pass = make_accumulation_pass(size.width, size.height, "large", Some(&small_pass));

    let large_bind_group = make_bind_group(&large_pass);

    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: None,
        bind_group_layouts: &[&accumulation_bind_group_layout],
        push_constant_ranges: &[],
    });

    let postprocess_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some("postprocess"),
        layout: Some(&pipeline_layout),
        vertex: vertex_shader.clone(),
        fragment: Some(wgpu::FragmentState {
            module: &shader,
            entry_point: "fs_main_textured",
            targets: &[wgpu::ColorTargetState {
                format: swapchain_format,
                color_blend: blend_add.clone(),
                alpha_blend: blend_add.clone(),
                write_mask: wgpu::ColorWrite::ALL,
            }],
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
        let _ = (&instance, &adapter, &shader, &small_pass, &large_pass);

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

                {
                    // Draw main pass
                    let (vertex_data, instance_data) = build_mesh(&scene);

                    let vertex_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                        label: Some("Vertex Buffer"),
                        contents: bytemuck::cast_slice(&vertex_data),
                        usage: wgpu::BufferUsage::VERTEX,
                    });

                    let instance_buf =
                        device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                            label: Some("Instance Buffer"),
                            contents: bytemuck::cast_slice(&instance_data),
                            usage: wgpu::BufferUsage::VERTEX,
                        });

                    let mut encoder: wgpu::CommandEncoder = device
                        .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
                    {
                        let mut small_render_pass = apply_pass(&mut encoder, &small_pass);
                        small_render_pass.set_vertex_buffer(0, instance_buf.slice(..));
                        small_render_pass.set_vertex_buffer(1, vertex_buf.slice(..));
                        small_render_pass.draw(
                            0..(vertex_data.len() as u32),
                            0..(instance_data.len() as u32),
                        );
                        drop(small_render_pass);

                        let mut large_render_pass = apply_pass(&mut encoder, &large_pass);
                        large_render_pass.set_vertex_buffer(0, instance_buf.slice(..));
                        large_render_pass.set_vertex_buffer(1, vertex_buf.slice(..));
                        large_render_pass.draw(
                            0..(vertex_data.len() as u32),
                            0..(instance_data.len() as u32),
                        );
                        drop(large_render_pass);

                        let mut postprocess_pass =
                            encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
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
                        postprocess_pass.set_pipeline(&postprocess_pipeline);
                        postprocess_pass.set_bind_group(0, &large_bind_group, &[]);
                        postprocess_pass.set_vertex_buffer(0, instance_buf.slice(..));
                        postprocess_pass.set_vertex_buffer(1, vertex_buf.slice(..));
                        postprocess_pass.draw(
                            0..(vertex_data.len() as u32),
                            0..(instance_data.len() as u32),
                        );
                    }

                    queue.submit(Some(encoder.finish()));
                }

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

fn apply_pass<'b>(
    encoder: &'b mut wgpu::CommandEncoder,
    pass: &'b AccumulatePass,
) -> wgpu::RenderPass<'b> {
    let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
        label: None,
        color_attachments: &[wgpu::RenderPassColorAttachmentDescriptor {
            attachment: &pass.view,
            resolve_target: None,
            ops: wgpu::Operations {
                load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                store: true,
            },
        }],
        depth_stencil_attachment: None,
    });
    render_pass.set_pipeline(&pass.pipeline);
    match pass.bind_group {
        Some(ref b) => render_pass.set_bind_group(0, &b, &[]),
        None => {}
    };
    render_pass
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
    let mut positions: Vec<Position> = vec![];
    let tri_verts = [
        corners[0], corners[1], corners[2], corners[0], corners[2], corners[3],
    ];

    let uv_corners = geometry::Rect {
        min: na::Point2::new(0.0, 0.0),
        max: na::Point2::new(1.0, 1.0),
    }
    .corners();

    let mut uv_verts: Vec<TextureCoordinate> = vec![];
    let uv_tri_verts: Vec<TextureCoordinate> = [
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
            positions.push([t2.x as f32, t2.y as f32].into());
        }
        uv_verts.extend(uv_tri_verts.iter());
    });

    let verts = positions
        .into_iter()
        .zip(uv_verts.into_iter())
        .map(|(v, u)| Vertex {
            position: v,
            texture_coordinate: u,
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
