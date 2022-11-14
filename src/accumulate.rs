use num::rational::Ratio;
use std::borrow::Cow;
use wgpu::{
    BindGroup, BindGroupDescriptor, BindGroupEntry, BindGroupLayout, BindGroupLayoutEntry,
    BindingResource, BindingType, Extent3d, FilterMode, PipelineLayoutDescriptor,
    SamplerDescriptor, ShaderModule, ShaderModuleDescriptor, ShaderSource, ShaderStages,
    TextureDescriptor, TextureFormat, TextureSampleType, TextureUsages, TextureViewDescriptor,
    TextureViewDimension,
};
use winit::dpi::PhysicalSize;

use crate::{
    flame::{BoundedState, State},
    geometry::{self, box_to_box, letter_box_scale, Bounds, Rect},
    mesh::{build_instances, build_mesh},
    render_common::MeshData,
    util_types::PtrRc,
    wgpu_render::Renderer,
};

#[derive(Clone, Debug, Hash, PartialEq, Eq)]
struct Accumulate {
    pub levels: u32,
    pub size: PhysicalSize<u32>,
    pub name: String,
}

impl Accumulate {
    fn mesh_levels(&self) -> u32 {
        self.levels - self.instance_levels()
    }

    fn instance_levels(&self) -> u32 {
        self.levels / 2
    }
}

#[salsa::query_group(AccumulateStorage)]
pub trait Accumulator: Renderer {
    fn data(&self, key: ()) -> PtrRc<DeviceData>;
    fn pass(&self, key: PassKey) -> PtrRc<Pass>;
    fn mesh(&self, key: u32) -> PtrRc<MeshData>;
    fn instance(&self, key: InstanceKey) -> PtrRc<MeshData>;
    fn bounds(&self, key: ()) -> Rect;
}

fn bounds(db: &dyn Accumulator, (): ()) -> Rect {
    let root = db.root(());
    let levels = 5;

    // This can be expensive, so cache it.
    let bounds = root.get_state().get_bounds(levels);
    if bounds.is_infinite() {
        panic!("infinite bounds")
    }
    bounds
}

pub fn mesh(db: &dyn Accumulator, levels: u32) -> PtrRc<MeshData> {
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

pub fn instance(db: &dyn Accumulator, key: InstanceKey) -> PtrRc<MeshData> {
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

/// Device dependant, but otherwise constant data.
#[derive(Debug)]
pub struct DeviceData {
    shader: ShaderModule,
    pub accumulation_bind_group_layout: BindGroupLayout,
    accumulation_sampler: wgpu::Sampler,
    nearest_sampler: wgpu::Sampler,
}

#[derive(Debug)]
pub struct Pass {
    pipeline: wgpu::RenderPipeline,
    output_bind_group: wgpu::BindGroup,
    view: wgpu::TextureView,
    spec: Accumulate,
    smaller: Option<PassKey>,
}

#[derive(Debug, Clone, Hash, PartialEq, Eq)]
pub struct PassKey {
    pub resolution: PhysicalSize<u32>,
    pub filter: bool,
}

pub fn data(db: &dyn Accumulator, (): ()) -> PtrRc<DeviceData> {
    let device = db.device(());
    DeviceData {
        // Load the shaders from disk
        shader: device.create_shader_module(ShaderModuleDescriptor {
            label: Some("wgpu.wgsl"),
            source: ShaderSource::Wgsl(Cow::Borrowed(include_str!("../shaders/wgpu.wgsl"))),
        }),

        accumulation_bind_group_layout: device.create_bind_group_layout(
            &wgpu::BindGroupLayoutDescriptor {
                entries: &[
                    BindGroupLayoutEntry {
                        binding: 0,
                        visibility: ShaderStages::FRAGMENT,
                        ty: BindingType::Texture {
                            multisampled: false,
                            // R32Float textures to not support filtering be default: requires native feature opt-in.
                            sample_type: TextureSampleType::Float { filterable: true },
                            view_dimension: TextureViewDimension::D2,
                        },
                        count: None,
                    },
                    BindGroupLayoutEntry {
                        binding: 1,
                        visibility: ShaderStages::FRAGMENT,
                        ty: BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                        count: None,
                    },
                ],
                label: None,
            },
        ),

        // TODO: mipmap filtering and generation
        accumulation_sampler: device.create_sampler(&SamplerDescriptor {
            label: Some("accumulation sampler"),
            mag_filter: FilterMode::Linear,
            min_filter: FilterMode::Linear,
            ..Default::default()
        }),

        nearest_sampler: device.create_sampler(&SamplerDescriptor {
            label: Some("nearest sampler"),
            mag_filter: FilterMode::Nearest,
            min_filter: FilterMode::Nearest,
            ..Default::default()
        }),
    }
    .into()
}

impl Pass {
    pub fn render(&self, db: &dyn Accumulator, encoder: &mut wgpu::CommandEncoder) -> &BindGroup {
        let vertexes = db.mesh(self.spec.mesh_levels());
        let instances = db.instance(InstanceKey {
            levels: self.spec.instance_levels(),
            aspect_ratio: Ratio::new(self.spec.size.width, self.spec.size.height),
        });

        // TODO: avoid having 3 "if let"s for this.
        let smaller_pass = if let Some(b) = &self.smaller {
            let inner = db.pass(b.clone());
            Some(inner)
        } else {
            None
        };

        let smaller = smaller_pass.as_ref().map(|b| b.render(db, encoder));

        let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("Accumulate"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: &self.view,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                    store: true,
                },
            })],
            depth_stencil_attachment: None,
        });
        render_pass.set_pipeline(&self.pipeline);
        if let Some(b) = &smaller {
            render_pass.set_bind_group(0, b, &[])
        };

        render_pass.set_vertex_buffer(0, instances.buffer.slice(..));
        render_pass.set_vertex_buffer(1, vertexes.buffer.slice(..));
        render_pass.draw(0..(vertexes.count), 0..(instances.count));
        &self.output_bind_group
    }
}

fn area_sf(t: &na::Affine2<f64>) -> f64 {
    let mat = t.matrix();
    // get the upper 2x2 (as that is what effects scaling). TODO: better way to get scale factor.
    let m2 = nalgebra::Matrix2::from_fn(|a, b| mat.row(a)[b]);
    m2.determinant()
}

fn texture_size(s: f64) -> u32 {
    u32::max(1, (s / 8.0) as u32 * 8)
}

/// Returns a BindGroup for reading from the the output from the pass
pub fn pass(db: &dyn Accumulator, key: PassKey) -> PtrRc<Pass> {
    let b = db.bounds(());
    let root = db.root(());
    let mut sf_min = f64::INFINITY;
    let mut sf_max = f64::NEG_INFINITY;
    let mut fill_ratio = 0.0;
    let mut count = 0;
    // TODO: should render variable number of iterations of different functions to get more uniform scale instead of fixed level (recurse if it helps)
    // TODO: avoid redoing this analysis for every pass
    root.get_state().process_levels(1, &mut |x| {
        let sf = area_sf(&x.mat);
        sf_min = f64::min(sf_min, sf);
        sf_max = f64::max(sf_max, sf);
        fill_ratio += sf;
        count += 1;
    });
    let sf_min = f64::sqrt(sf_min);
    let _sf_max = f64::sqrt(sf_max);

    let lb_scale = letter_box_scale(
        Rect {
            min: na::Point2::origin(),
            max: na::Point2::new(key.resolution.width as f64, key.resolution.height as f64),
        },
        b,
    );

    let width_to_fill = lb_scale * b.width();
    let height_to_fill = lb_scale * b.height();

    let fill_area = fill_ratio * width_to_fill * height_to_fill;

    // TODO: reasonable cost function
    let mut passes: u32 = if fill_area > 1024.0 * 1024.0 {
        2
    } else if fill_area > 256.0 * 256.0 {
        6
    } else {
        8
    };

    // Avoid buffers being too large
    const BUFFER_LIMIT: usize = 512;
    while passes > 2 && db.config(()).n.pow(passes / 2) > BUFFER_LIMIT {
        passes -= 1;
    }

    let sf = sf_min.powi(passes as i32);

    let width = texture_size(width_to_fill * sf);
    let height = texture_size(height_to_fill * sf);

    let smaller = if width > 16 || height > 16 {
        Some(PassKey {
            filter: true,
            resolution: [width, height].into(),
        })
    } else {
        None
    };
    make_pass(
        db,
        Accumulate {
            levels: passes,
            size: key.resolution,
            name: "AutoSized".to_owned(),
        },
        smaller,
        key.filter,
    )
    .into()
}

fn make_pass(
    db: &dyn Accumulator,
    accumulate: Accumulate,
    smaller: Option<PassKey>,
    filter: bool,
) -> Pass {
    let device = db.device(());
    let data = db.data(());

    let blend_add = wgpu::BlendComponent {
        src_factor: wgpu::BlendFactor::One,
        dst_factor: wgpu::BlendFactor::One,
        operation: wgpu::BlendOperation::Add,
    };

    let blend_state_add = wgpu::BlendState {
        color: blend_add,
        alpha: blend_add,
    };

    let groups = &[&data.accumulation_bind_group_layout];
    let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
        label: Some("accumulation pipeline"),
        bind_group_layouts: if smaller.is_some() { groups } else { &[] },
        push_constant_ranges: &[],
    });

    let vertex_shader = wgpu::VertexState {
        module: &data.shader,
        entry_point: "vs_main",
        buffers: &[
            wgpu::VertexBufferLayout {
                array_stride: 2 * 4 * 4,
                step_mode: wgpu::VertexStepMode::Instance,
                attributes: &wgpu::vertex_attr_array![0 => Float32x4, 1 => Float32x4], // Rows of matrix
            },
            wgpu::VertexBufferLayout {
                array_stride: 2 * 2 * 4,
                step_mode: wgpu::VertexStepMode::Vertex,
                attributes: &wgpu::vertex_attr_array![2 => Float32x2, 3 => Float32x2],
            },
        ],
    };

    let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some(&accumulate.name),
        layout: Some(&pipeline_layout),
        vertex: vertex_shader,
        fragment: Some(wgpu::FragmentState {
            module: &data.shader,
            entry_point: if smaller.is_some() {
                "fs_main_textured"
            } else {
                "fs_main"
            },
            targets: &[Some(wgpu::ColorTargetState {
                format: TextureFormat::R32Float,
                blend: Some(blend_state_add),
                write_mask: wgpu::ColorWrites::ALL,
            })],
        }),
        primitive: wgpu::PrimitiveState::default(),
        depth_stencil: None,
        multisample: wgpu::MultisampleState::default(),
        multiview: None,
    });

    let texture: wgpu::Texture = device.create_texture(&TextureDescriptor {
        size: Extent3d {
            width: accumulate.size.width,
            height: accumulate.size.height,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: TextureFormat::R32Float,
        usage: TextureUsages::TEXTURE_BINDING | TextureUsages::RENDER_ATTACHMENT,
        label: Some(&accumulate.name),
    });

    let view: wgpu::TextureView = texture.create_view(&TextureViewDescriptor::default());

    let output_bind_group = device.create_bind_group(&BindGroupDescriptor {
        layout: &data.accumulation_bind_group_layout,
        entries: &[
            BindGroupEntry {
                binding: 0,
                resource: BindingResource::TextureView(&view),
            },
            BindGroupEntry {
                binding: 1,
                resource: BindingResource::Sampler(if filter {
                    &data.nearest_sampler
                } else {
                    &data.accumulation_sampler
                }),
            },
        ],
        label: None,
    });

    Pass {
        pipeline,
        view,
        output_bind_group,
        smaller,
        spec: accumulate,
    }
}
