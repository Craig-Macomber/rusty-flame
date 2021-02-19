use bytemuck::Pod;
use std::{borrow::Cow, collections::HashMap, hash::Hash, rc::Rc};
use wgpu::{
    util::DeviceExt, BindGroup, BindGroupLayout, Buffer, Device, Extent3d, Queue, ShaderModule,
    TextureFormat,
};
use winit::dpi::PhysicalSize;

use crate::{
    flame::Root,
    get_state,
    mesh::{build_instances, build_mesh, build_quad},
    plan::{plan_render, Accumulate},
};

#[salsa::query_group(RendererStorage)]
pub trait Renderer: salsa::Database {
    #[salsa::input]
    fn cursor(&self, key: ()) -> [f64; 2];

    #[salsa::input]
    fn size(&self, key: ()) -> PhysicalSize<u32>;

    #[salsa::input]
    fn device(&self, key: ()) -> Rc<Device>;

    #[salsa::input]
    fn queue(&self, key: ()) -> Rc<Queue>;

    #[salsa::input]
    fn swapchain_format(&self, key: ()) -> DebugIt<TextureFormat>;

    fn data(&self, key: ()) -> PtrRc<PlaneRendererData>;

    fn sized_plan(&self, key: ()) -> PtrRc<SizePlanRenderer>;

    fn root(&self, key: ()) -> Root;
}

#[derive(Debug, Clone)]
pub struct DebugIt<T>(pub T);

#[derive(Debug)]
pub struct PtrRc<T>(Rc<T>);

// TODO: why does derive clone not work for this?
impl<T> Clone for PtrRc<T> {
    fn clone(&self) -> Self {
        PtrRc(self.0.clone())
    }
}

impl<T> PartialEq for PtrRc<T> {
    fn ne(&self, other: &Self) -> bool {
        !Rc::ptr_eq(&self.0, &other.0)
    }

    fn eq(&self, other: &Self) -> bool {
        Rc::ptr_eq(&self.0, &other.0)
    }
}

impl<T> Eq for PtrRc<T> {}

///////////////////////////////////////////////////////////////////////////
// Step 2. Define the queries.

fn root(db: &dyn Renderer, (): ()) -> Root {
    let cursor = db.cursor(());
    get_state(cursor)
}

///////////////////////////////////////////////////////////////////////////
// Step 3. Define the database struct
#[salsa::database(RendererStorage)]
#[derive(Default)]
pub struct DatabaseStruct {
    storage: salsa::Storage<Self>,
}

impl salsa::Database for DatabaseStruct {}

/// Non size dependent cached data
#[derive(Debug)]
pub struct PlaneRendererData {
    shader: ShaderModule,
    postprocess_shader: ShaderModule,
    gradient_bind_group: wgpu::BindGroup,
    gradient_bind_group_layout: BindGroupLayout,
    swapchain_format: TextureFormat,
}

/// Size dependant data
#[derive(Debug)]
pub struct SizePlanRenderer {
    passes: Vec<AccumulatePass>,
    postprocess_pipeline: wgpu::RenderPipeline,
    large_bind_group: wgpu::BindGroup,
    quad: MeshData,
}

fn data(db: &dyn Renderer, (): ()) -> PtrRc<PlaneRendererData> {
    PtrRc(Rc::new(PlaneRendererData::new(db)))
}

fn sized_plan(db: &dyn Renderer, (): ()) -> PtrRc<SizePlanRenderer> {
    let device = &db.device(());
    let data = &*db.data(()).0;
    let size = db.size(());

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

    let vertex_shader = wgpu::VertexState {
        module: &data.shader,
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
                module: &data.shader,
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

    let plan = plan_render(size.into());

    let mut passes: Vec<AccumulatePass> = vec![];
    for p in &plan.passes {
        passes.push(make_accumulation_pass(p, passes.last()))
    }

    let last_pass = passes.last().unwrap();

    let large_bind_group = make_bind_group(&last_pass, false);

    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("postprocess pipeline"),
        bind_group_layouts: &[
            &accumulation_bind_group_layout,
            &data.gradient_bind_group_layout,
        ],
        push_constant_ranges: &[],
    });

    let postprocess_pipeline: wgpu::RenderPipeline =
        device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("postprocess"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &data.postprocess_shader,
                entry_point: "vs_main",
                buffers: &[wgpu::VertexBufferLayout {
                    array_stride: 2 * 2 * 4,
                    step_mode: wgpu::InputStepMode::Vertex,
                    attributes: &wgpu::vertex_attr_array![0 => Float2, 1 => Float2],
                }],
            },
            fragment: Some(wgpu::FragmentState {
                module: &data.postprocess_shader,
                entry_point: "fs_main",
                targets: &[wgpu::ColorTargetState {
                    format: data.swapchain_format,
                    color_blend: blend_replace.clone(),
                    alpha_blend: blend_replace.clone(),
                    write_mask: wgpu::ColorWrite::ALL,
                }],
            }),
            primitive: wgpu::PrimitiveState::default(),
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
        });

    PtrRc(Rc::new(SizePlanRenderer {
        passes,
        large_bind_group,
        postprocess_pipeline,
        quad: MeshData::new(device, &build_quad(), "Quad Vertex Buffer"),
    }))
}

impl PlaneRendererData {
    fn new(db: &dyn Renderer) -> Self {
        let device = db.device(());
        let queue = db.queue(());
        let swapchain_format = db.swapchain_format(()).0;

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

        let diffuse_bytes = include_bytes!("../images/gradient.png");
        let diffuse_image = image::load_from_memory(diffuse_bytes).unwrap();
        let diffuse_rgba = diffuse_image.as_rgba8().unwrap();

        use image::GenericImageView;
        let dimensions = diffuse_image.dimensions();

        let texture_size = wgpu::Extent3d {
            width: dimensions.0,
            height: dimensions.1,
            depth: 1,
        };

        let diffuse_texture = device.create_texture(&wgpu::TextureDescriptor {
            // All textures are stored as 3D, we represent our 2D texture
            // by setting depth to 1.
            size: texture_size,
            mip_level_count: 1, // We'll talk about this a little later
            sample_count: 1,
            dimension: wgpu::TextureDimension::D1,
            format: wgpu::TextureFormat::Rgba8UnormSrgb,
            // SAMPLED tells wgpu that we want to use this texture in shaders
            // COPY_DST means that we want to copy data to this texture
            usage: wgpu::TextureUsage::SAMPLED | wgpu::TextureUsage::COPY_DST,
            label: Some("diffuse_texture"),
        });

        queue.write_texture(
            // Tells wgpu where to copy the pixel data
            wgpu::TextureCopyView {
                texture: &diffuse_texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
            },
            // The actual pixel data
            diffuse_rgba,
            // The layout of the texture
            wgpu::TextureDataLayout {
                offset: 0,
                bytes_per_row: 4 * dimensions.0,
                rows_per_image: dimensions.1,
            },
            texture_size,
        );

        let diffuse_texture_view =
            diffuse_texture.create_view(&wgpu::TextureViewDescriptor::default());
        let diffuse_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Nearest,
            mipmap_filter: wgpu::FilterMode::Nearest,
            ..Default::default()
        });

        let gradient_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                entries: &[
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStage::FRAGMENT,
                        ty: wgpu::BindingType::Texture {
                            multisampled: false,
                            sample_type: wgpu::TextureSampleType::Float { filterable: true },
                            view_dimension: wgpu::TextureViewDimension::D1,
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

        let gradient_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            layout: &gradient_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&diffuse_texture_view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(&diffuse_sampler),
                },
            ],
            label: None,
        });

        Self {
            shader,
            postprocess_shader,
            gradient_bind_group,
            gradient_bind_group_layout,
            swapchain_format,
        }
    }
}

#[derive(Debug)]
struct AccumulatePass {
    pipeline: wgpu::RenderPipeline,
    bind_group: Option<wgpu::BindGroup>,
    view: wgpu::TextureView,
    spec: Accumulate,
}

#[derive(Debug)]
struct MeshData {
    count: usize,
    buffer: Buffer,
}

impl MeshData {
    pub fn new<'a, T: Pod>(device: &'a Device, data: &Vec<T>, label: &'a str) -> MeshData {
        MeshData {
            count: data.len(),
            buffer: device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some(label),
                contents: bytemuck::cast_slice(&data),
                usage: wgpu::BufferUsage::VERTEX,
            }),
        }
    }
}

/// Caches the result of Factory(Input) -> Output for various Inputs.
struct Cache<Input, Output, Factory> {
    map: HashMap<Input, Output>,
    function: Factory,
}

impl<Input, Output, Factory> Cache<Input, Output, Factory>
where
    Factory: Fn(&Input) -> Output,
    Input: Eq + Hash,
{
    pub fn new(f: Factory) -> Cache<Input, Output, Factory> {
        Cache {
            map: HashMap::new(),
            function: f,
        }
    }

    pub fn get(&mut self, k: Input) -> &Output {
        self.map.entry(k).or_insert_with_key(&self.function)
    }
}

pub fn render(db: &DatabaseStruct, frame: &wgpu::SwapChainTexture) {
    let render_data = &db.data(()).0;
    let device = &db.device(());
    let root = db.root(());
    let plan = db.sized_plan(()).0;

    let mut mesh_cache = Cache::new(|levels: &u32| {
        MeshData::new(device, &build_mesh(&root, *levels), "Vertex Buffer")
    });

    let mut instance_cache = Cache::new(|levels: &u32| {
        MeshData::new(device, &build_instances(&root, *levels), "Instance Buffer")
    });

    // Draw main pass
    let mut encoder: wgpu::CommandEncoder =
        device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
    {
        for p in &plan.passes {
            let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Accumulate"),
                color_attachments: &[wgpu::RenderPassColorAttachmentDescriptor {
                    attachment: &p.view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                        store: true,
                    },
                }],
                depth_stencil_attachment: None,
            });
            render_pass.set_pipeline(&p.pipeline);
            if let Some(ref b) = p.bind_group {
                render_pass.set_bind_group(0, &b, &[])
            };

            let vertexes = mesh_cache.get(p.spec.mesh_levels);
            let instances = instance_cache.get(p.spec.instance_levels);

            render_pass.set_vertex_buffer(0, instances.buffer.slice(..));
            render_pass.set_vertex_buffer(1, vertexes.buffer.slice(..));
            render_pass.draw(0..(vertexes.count as u32), 0..(instances.count as u32));
        }

        {
            let mut postprocess_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Postprocess render pass"),
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

            postprocess_pass.set_pipeline(&plan.postprocess_pipeline);
            postprocess_pass.set_bind_group(0, &plan.large_bind_group, &[]);
            postprocess_pass.set_bind_group(1, &render_data.gradient_bind_group, &[]);
            postprocess_pass.set_vertex_buffer(0, plan.quad.buffer.slice(..));
            postprocess_pass.set_bind_group(0, &plan.large_bind_group, &[]);
            postprocess_pass.draw(0..(plan.quad.count as u32), 0..1);
        }
    }

    db.queue(()).submit(Some(encoder.finish()));
}
