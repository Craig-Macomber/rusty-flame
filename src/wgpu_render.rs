use geometry::Bounds;
use num::rational::Ratio;
use std::{borrow::Cow, rc::Rc};
use wgpu::{
    BindGroup, BindGroupDescriptor, BindGroupEntry, BindGroupLayout, BindGroupLayoutEntry,
    BindingResource, BindingType, CommandEncoderDescriptor, Device, Extent3d, FilterMode,
    PipelineLayoutDescriptor, Queue, SamplerDescriptor, ShaderModule, ShaderModuleDescriptor,
    ShaderSource, ShaderStage, TextureDescriptor, TextureFormat, TextureSampleType, TextureUsage,
    TextureViewDescriptor, TextureViewDimension,
};
use winit::dpi::PhysicalSize;

use crate::{
    flame::{BoundedState, Root},
    geometry::{self, box_to_box, Rect},
    get_state,
    mesh::{build_instances, build_mesh},
    plan::{plan_render, Accumulate, Plan},
    postprocess,
    render_common::MeshData,
    util_types::{DebugIt, PtrRc},
};

#[salsa::query_group(InputStorage)]
pub trait Inputs: salsa::Database {
    #[salsa::input]
    fn cursor(&self, key: ()) -> [f64; 2];

    #[salsa::input]
    fn window_size(&self, key: ()) -> PhysicalSize<u32>;

    #[salsa::input]
    fn device(&self, key: ()) -> Rc<Device>;

    #[salsa::input]
    fn queue(&self, key: ()) -> Rc<Queue>;

    #[salsa::input]
    fn swapchain_format(&self, key: ()) -> DebugIt<TextureFormat>;
}

#[salsa::query_group(RendererStorage)]
pub trait Renderer: Inputs {
    fn data(&self, key: ()) -> PtrRc<DeviceData>;
    fn sized_plan(&self, key: ()) -> PtrRc<SizePlanRenderer>;
    fn root(&self, key: ()) -> Root;
    fn mesh(&self, key: u32) -> PtrRc<MeshData>;
    fn instance(&self, key: InstanceKey) -> PtrRc<MeshData>;
    fn bounds(&self, key: ()) -> Rect;
    fn plan(&self, key: ()) -> PtrRc<Plan>;
    fn postprocess_data(&self, key: ()) -> PtrRc<postprocess::Data>;
}

fn root(db: &dyn Renderer, (): ()) -> Root {
    get_state(db.cursor(()))
}

fn plan(db: &dyn Renderer, (): ()) -> PtrRc<Plan> {
    plan_render(db.window_size(()).into()).into()
}

fn postprocess_data(db: &dyn Renderer, (): ()) -> PtrRc<postprocess::Data> {
    postprocess::data(db, ()).into()
}

fn bounds(db: &dyn Renderer, (): ()) -> Rect {
    let root = db.root(());
    let passes = &db.plan(()).passes;

    let levels = u32::min(
        5,
        passes
            .iter()
            .map(|pass| pass.mesh_levels + pass.instance_levels)
            .sum(),
    );

    // This can be expensive, so cache it.
    let bounds = root.get_state().get_bounds(levels);
    if bounds.is_infinite() {
        panic!("infinite bounds")
    }
    bounds
}

fn mesh(db: &dyn Renderer, levels: u32) -> PtrRc<MeshData> {
    let bounds = db.bounds(());
    MeshData::new(
        &*db.device(()),
        &build_mesh(&db.root(()), bounds, levels),
        "Vertex Buffer",
    )
    .into()
}

#[derive(Debug, Clone, Hash, PartialEq, Eq)]
pub struct InstanceKey {
    levels: u32,
    // width / height
    // TODO: use a uniform buffer for root transformation for better caching and to allow non-letterbox positioning of final pass.
    aspect_ratio: Ratio<u32>,
}

fn instance(db: &dyn Renderer, key: InstanceKey) -> PtrRc<MeshData> {
    let bounds = db.bounds(());

    let window_rect = geometry::Rect {
        min: na::Point2::new(0.0, 0.0),
        max: na::Point2::new(
            *key.aspect_ratio.numer() as f64,
            *key.aspect_ratio.denom() as f64,
        ),
    };

    let root_mat = geometry::letter_box(window_rect, bounds);

    let rebox = box_to_box(
        geometry::Rect {
            min: na::Point2::new(-1.0, -1.0),
            max: na::Point2::new(1.0, 1.0),
        },
        window_rect,
    );

    MeshData::new(
        &*db.device(()),
        &build_instances(&db.root(()), rebox * root_mat, key.levels),
        "Instance Buffer",
    )
    .into()
}

#[salsa::database(RendererStorage, InputStorage)]
#[derive(Default)]
pub struct DatabaseStruct {
    storage: salsa::Storage<Self>,
}

impl salsa::Database for DatabaseStruct {}

/// Device dependant, but otherwise constant data.
#[derive(Debug)]
pub struct DeviceData {
    shader: ShaderModule,
    pub accumulation_bind_group_layout: BindGroupLayout,
}

#[derive(Debug)]
struct AccumulatePass {
    pipeline: wgpu::RenderPipeline,
    bind_group: Option<wgpu::BindGroup>,
    view: wgpu::TextureView,
    spec: Accumulate,
}

/// Size dependant data
#[derive(Debug)]
pub struct SizePlanRenderer {
    passes: Vec<AccumulatePass>,
    pub large_bind_group: wgpu::BindGroup,
}

fn sized_plan(db: &dyn Renderer, (): ()) -> PtrRc<SizePlanRenderer> {
    let device = db.device(());
    let data = db.data(());

    let blend_add = wgpu::BlendState {
        src_factor: wgpu::BlendFactor::One,
        dst_factor: wgpu::BlendFactor::One,
        operation: wgpu::BlendOperation::Add,
    };

    // TODO: mipmap filtering and generation
    let accumulation_sampler: wgpu::Sampler = device.create_sampler(&SamplerDescriptor {
        label: Some("accumulation sampler"),
        mag_filter: FilterMode::Linear,
        min_filter: FilterMode::Linear,
        ..Default::default()
    });

    let nearest_sampler: wgpu::Sampler = device.create_sampler(&SamplerDescriptor {
        label: Some("nearest sampler"),
        mag_filter: FilterMode::Nearest,
        min_filter: FilterMode::Nearest,
        ..Default::default()
    });

    let make_bind_group = |p: &AccumulatePass, filter: bool| -> BindGroup {
        device.create_bind_group(&BindGroupDescriptor {
            layout: &data.accumulation_bind_group_layout,
            entries: &[
                BindGroupEntry {
                    binding: 0,
                    resource: BindingResource::TextureView(&p.view),
                },
                BindGroupEntry {
                    binding: 1,
                    resource: BindingResource::Sampler(if filter {
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

    let make_accumulation_pass =
        |accumulate: &Accumulate, previous: Option<&AccumulatePass>| -> AccumulatePass {
            // Create bind group
            let bind_group = match previous {
                Some(p) => Some(make_bind_group(p, true)),
                None => None,
            };

            let groups = &[&data.accumulation_bind_group_layout];
            let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
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

            let texture: wgpu::Texture = device.create_texture(&TextureDescriptor {
                size: Extent3d {
                    width: accumulate.size.width,
                    height: accumulate.size.height,
                    depth: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: TextureFormat::R32Float,
                usage: TextureUsage::RENDER_ATTACHMENT | TextureUsage::SAMPLED,
                label: Some(&accumulate.name),
            });

            let view: wgpu::TextureView = texture.create_view(&TextureViewDescriptor::default());

            AccumulatePass {
                pipeline,
                bind_group,
                view,
                spec: accumulate.clone(),
            }
        };

    let plan = db.plan(());

    let mut passes: Vec<AccumulatePass> = vec![];
    for p in &plan.passes {
        passes.push(make_accumulation_pass(p, passes.last()))
    }

    let last_pass = passes.last().unwrap();

    let large_bind_group = make_bind_group(&last_pass, false);

    SizePlanRenderer {
        passes,
        large_bind_group,
    }
    .into()
}

fn data(db: &dyn Renderer, (): ()) -> PtrRc<DeviceData> {
    let device = db.device(());

    // Load the shaders from disk
    let shader = device.create_shader_module(&ShaderModuleDescriptor {
        label: Some("wgpu.wgsl"),
        source: ShaderSource::Wgsl(Cow::Borrowed(include_str!("../shaders/wgpu.wgsl"))),
        flags: wgpu::ShaderFlags::all(),
    });

    let accumulation_bind_group_layout =
        device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            entries: &[
                BindGroupLayoutEntry {
                    binding: 0,
                    visibility: ShaderStage::FRAGMENT,
                    ty: BindingType::Texture {
                        multisampled: false,
                        sample_type: TextureSampleType::Float { filterable: true },
                        view_dimension: TextureViewDimension::D2,
                    },
                    count: None,
                },
                BindGroupLayoutEntry {
                    binding: 1,
                    visibility: ShaderStage::FRAGMENT,
                    ty: BindingType::Sampler {
                        comparison: false,
                        filtering: true,
                    },
                    count: None,
                },
            ],
            label: None,
        });

    DeviceData {
        shader,
        accumulation_bind_group_layout,
    }
    .into()
}

pub fn render(db: &DatabaseStruct, frame: &wgpu::SwapChainTexture) {
    let device = db.device(());
    let plan = db.sized_plan(());

    let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor { label: None });
    {
        for p in &plan.passes {
            let vertexes = db.mesh(p.spec.mesh_levels);
            let instances = db.instance(InstanceKey {
                levels: p.spec.instance_levels,
                aspect_ratio: Ratio::new(p.spec.size.width, p.spec.size.height),
            });

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

            render_pass.set_vertex_buffer(0, instances.buffer.slice(..));
            render_pass.set_vertex_buffer(1, vertexes.buffer.slice(..));
            render_pass.draw(0..(vertexes.count), 0..(instances.count));
        }

        postprocess::render(db, frame, &mut encoder);

        // TODO: debug option to draw intermediate texture to screen at actual resolution
    }

    db.queue(()).submit(Some(encoder.finish()));
}
