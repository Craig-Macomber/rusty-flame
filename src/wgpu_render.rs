use nalgebra::Point2;
use winit::{
    event::{Event, WindowEvent},
    event_loop::{ControlFlow, EventLoop},
    window::Window,
};

use std::{borrow::Cow, collections::HashMap};
use wgpu::{util::DeviceExt, BindGroup, Buffer, Extent3d, TextureFormat};

use crate::{
    flame::Root,
    get_state,
    mesh::{build_instances, build_mesh, build_quad},
};

#[derive(PartialEq)]
pub struct SceneState {
    pub cursor: Point2<f64>,
}

#[derive(Clone)]
pub(crate) struct Accumulate {
    pub instance_levels: u32,
    pub mesh_levels: u32,
    size: winit::dpi::PhysicalSize<u32>,
    name: String,
}

/// Scene independent plan. Redone on window resize.
struct Plan {
    passes: Vec<Accumulate>,
}

const SMALL_ACCUMULATION_BUFFER_SIZE: u32 = 256;
const MID_ACCUMULATION_BUFFER_SIZE: u32 = 512;
const LARGE_ACCUMULATION_BUFFER_SIZE: u32 = 1024;
const LARGE_ACCUMULATION_BUFFER_SIZE2: u32 = 1800;

fn plan_render(size: winit::dpi::PhysicalSize<u32>) -> Plan {
    Plan {
        passes: vec![
            Accumulate {
                instance_levels: 6,
                mesh_levels: 6,
                size: winit::dpi::PhysicalSize {
                    width: SMALL_ACCUMULATION_BUFFER_SIZE,
                    height: SMALL_ACCUMULATION_BUFFER_SIZE,
                },
                name: "Small".to_owned(),
            },
            Accumulate {
                instance_levels: 4,
                mesh_levels: 6,
                size: winit::dpi::PhysicalSize {
                    width: MID_ACCUMULATION_BUFFER_SIZE,
                    height: MID_ACCUMULATION_BUFFER_SIZE,
                },
                name: "Mid".to_owned(),
            },
            Accumulate {
                instance_levels: 4,
                mesh_levels: 4,
                size: winit::dpi::PhysicalSize {
                    width: LARGE_ACCUMULATION_BUFFER_SIZE,
                    height: LARGE_ACCUMULATION_BUFFER_SIZE,
                },
                name: "Large".to_owned(),
            },
            Accumulate {
                instance_levels: 2,
                mesh_levels: 2,
                size: winit::dpi::PhysicalSize {
                    width: LARGE_ACCUMULATION_BUFFER_SIZE2,
                    height: LARGE_ACCUMULATION_BUFFER_SIZE2,
                },
                name: "Large2".to_owned(),
            },
            Accumulate {
                instance_levels: 2,
                mesh_levels: 2,
                size,
                name: "Main".to_owned(),
            },
        ],
    }
}

struct AccumulatePass {
    pipeline: wgpu::RenderPipeline,
    bind_group: Option<wgpu::BindGroup>,
    view: wgpu::TextureView,
    spec: Accumulate,
}

struct MeshData {
    count: usize,
    buffer: Buffer,
}

pub async fn run(event_loop: EventLoop<()>, window: Window) {
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

    let postprocess_shader = device.create_shader_module(&wgpu::ShaderModuleDescriptor {
        label: Some("postprocess.wgsl"),
        source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(include_str!(
            "../shaders/postprocess.wgsl"
        ))),
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

    let blend_replace = wgpu::BlendState {
        src_factor: wgpu::BlendFactor::One,
        dst_factor: wgpu::BlendFactor::Zero,
        operation: wgpu::BlendOperation::Add,
    };

    // TODO: mipmap filtering and generation
    let accumulation_sampler: wgpu::Sampler = device.create_sampler(&wgpu::SamplerDescriptor {
        label: Some("accumulation sampler"),
        mag_filter: wgpu::FilterMode::Linear,
        min_filter: wgpu::FilterMode::Linear,
        ..Default::default()
    });

    let nearest_sampler: wgpu::Sampler = device.create_sampler(&wgpu::SamplerDescriptor {
        label: Some("accumulation sampler"),
        mag_filter: wgpu::FilterMode::Nearest,
        min_filter: wgpu::FilterMode::Nearest,
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

    let make_bind_group = |p: &AccumulatePass, filter: bool| -> BindGroup {
        device.create_bind_group(&wgpu::BindGroupDescriptor {
            layout: &accumulation_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&p.view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(if filter {
                        &nearest_sampler
                    } else {
                        &accumulation_sampler
                    }),
                },
            ],
            label: None,
        })
    };

    let make_accumulation_pass = |accumulate: &Accumulate,
                                  previous: Option<&AccumulatePass>|
     -> AccumulatePass {
        // Create bind group
        let bind_group = match previous {
            Some(p) => Some(make_bind_group(p, true)),
            None => None,
        };

        let groups = &[&accumulation_bind_group_layout];
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("accumulation pipeline"),
            bind_group_layouts: match bind_group {
                Some(_) => groups,
                None => &[],
            },
            push_constant_ranges: &[],
        });

        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some(&accumulate.name),
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
                width: accumulate.size.width,
                height: accumulate.size.height,
                depth: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: TextureFormat::R32Float,
            usage: wgpu::TextureUsage::RENDER_ATTACHMENT | wgpu::TextureUsage::SAMPLED,
            label: Some(&accumulate.name),
        });

        let view: wgpu::TextureView = texture.create_view(&wgpu::TextureViewDescriptor::default());

        AccumulatePass {
            pipeline,
            bind_group,
            view,
            spec: accumulate.clone(),
        }
    };

    let plan = plan_render(size);

    let mut passes: Vec<AccumulatePass> = vec![];
    for p in &plan.passes {
        passes.push(make_accumulation_pass(p, passes.last()))
    }

    let last_pass = passes.last().unwrap();

    let large_bind_group = make_bind_group(&last_pass, false);

    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("postprocess pipeline"),
        bind_group_layouts: &[&accumulation_bind_group_layout],
        push_constant_ranges: &[],
    });

    let postprocess_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some("postprocess"),
        layout: Some(&pipeline_layout),
        vertex: wgpu::VertexState {
            module: &postprocess_shader,
            entry_point: "vs_main",
            buffers: &[wgpu::VertexBufferLayout {
                array_stride: 2 * 2 * 4,
                step_mode: wgpu::InputStepMode::Vertex,
                attributes: &wgpu::vertex_attr_array![0 => Float2, 1 => Float2],
            }],
        },
        fragment: Some(wgpu::FragmentState {
            module: &postprocess_shader,
            entry_point: "fs_main",
            targets: &[wgpu::ColorTargetState {
                format: swapchain_format,
                color_blend: blend_replace.clone(),
                alpha_blend: blend_replace.clone(),
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
        present_mode: wgpu::PresentMode::Fifo, // Mailbox
    };

    let mut swap_chain = device.create_swap_chain(&surface, &sc_desc);

    event_loop.run(move |event, _, control_flow| {
        // Have the closure take ownership of the resources.
        // `event_loop.run` never returns, therefore we must do this to ensure
        // the resources are properly cleaned up.
        let _ = (&instance, &adapter, &shader, &passes, &postprocess_shader);

        *control_flow = ControlFlow::Wait;
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
                let new_scene = SceneState {
                    cursor: Point2::new(
                        f64::from(position.x) / f64::from(size.width) * 2.0 - 1.0,
                        f64::from(position.y) / f64::from(size.height) * 2.0 - 1.0,
                    ),
                };
                if new_scene != scene {
                    scene = new_scene;
                    window.request_redraw();
                }
            }

            Event::RedrawRequested(_) => {
                let frame = swap_chain
                    .get_current_frame()
                    .expect("Failed to acquire next swap chain texture")
                    .output;

                {
                    let root: Root =
                        get_state([scene.cursor.x + 1.0, scene.cursor.y + 1.0], [2.0, 2.0]);

                    let mut mesh_cache: HashMap<u32, MeshData> = HashMap::new();

                    let build_mesh_data = |levels: &u32| {
                        let vertexes = build_mesh(&root, *levels);
                        let buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                            label: Some("Vertex Buffer"),
                            contents: bytemuck::cast_slice(&vertexes),
                            usage: wgpu::BufferUsage::VERTEX,
                        });
                        MeshData {
                            count: vertexes.len(),
                            buffer,
                        }
                    };

                    let mut instance_cache: HashMap<u32, MeshData> = HashMap::new();

                    let build_instance_data = |levels: &u32| {
                        let instances = build_instances(&root, *levels);
                        let buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                            label: Some("Instance Buffer"),
                            contents: bytemuck::cast_slice(&instances),
                            usage: wgpu::BufferUsage::VERTEX,
                        });
                        MeshData {
                            count: instances.len(),
                            buffer,
                        }
                    };

                    // Draw main pass

                    let mut encoder: wgpu::CommandEncoder = device
                        .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
                    {
                        for p in &passes {
                            let vertex_data = mesh_cache
                                .entry(p.spec.mesh_levels)
                                .or_insert_with_key(build_mesh_data);
                            let instance_data = instance_cache
                                .entry(p.spec.instance_levels)
                                .or_insert_with_key(build_instance_data);

                            let mut small_render_pass = apply_pass(&mut encoder, p);
                            small_render_pass.set_vertex_buffer(0, instance_data.buffer.slice(..));
                            small_render_pass.set_vertex_buffer(1, vertex_data.buffer.slice(..));
                            small_render_pass.draw(
                                0..(vertex_data.count as u32),
                                0..(instance_data.count as u32),
                            );
                        }

                        let quad_vertexes = build_quad();
                        let quad_buf =
                            device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                                label: Some("Quad Vertex Buffer"),
                                contents: bytemuck::cast_slice(&quad_vertexes),
                                usage: wgpu::BufferUsage::VERTEX,
                            });
                        {
                            let mut postprocess_pass =
                                encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                                    label: Some("Postprocess render pass"),
                                    color_attachments: &[
                                        wgpu::RenderPassColorAttachmentDescriptor {
                                            attachment: &frame.view,
                                            resolve_target: None,
                                            ops: wgpu::Operations {
                                                load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                                                store: true,
                                            },
                                        },
                                    ],
                                    depth_stencil_attachment: None,
                                });

                            postprocess_pass.set_pipeline(&postprocess_pipeline);
                            postprocess_pass.set_bind_group(0, &large_bind_group, &[]);
                            postprocess_pass.set_vertex_buffer(0, quad_buf.slice(..));
                            postprocess_pass.draw(0..(quad_vertexes.len() as u32), 0..1);
                        }
                    }

                    queue.submit(Some(encoder.finish()));
                }

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

fn apply_pass<'b>(
    encoder: &'b mut wgpu::CommandEncoder,
    pass: &'b AccumulatePass,
) -> wgpu::RenderPass<'b> {
    let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
        label: Some("Accumulate"),
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
