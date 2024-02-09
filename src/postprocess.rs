use std::borrow::Cow;
use wgpu::{
    AddressMode, BindGroupDescriptor, BindGroupEntry, BindGroupLayoutEntry, BindingResource,
    BindingType, FilterMode, PipelineLayoutDescriptor, SamplerDescriptor, ShaderModuleDescriptor,
    ShaderSource, ShaderStages, TextureAspect, TextureDescriptor, TextureFormat, TextureSampleType,
    TextureUsages, TextureViewDescriptor, TextureViewDimension,
};

use crate::{
    mesh::build_quad, render_common::MeshData, util_types::PtrRc, wgpu_render::Postprocesser,
};

/// Device dependant, but otherwise constant data.
#[derive(Debug)]
pub struct Data {
    gradient_bind_group: wgpu::BindGroup,
    quad: MeshData,
    pipeline: wgpu::RenderPipeline,
}

pub fn data(db: &dyn Postprocesser, (): ()) -> PtrRc<Data> {
    let device = db.device(());
    let queue = db.queue(());
    let data = db.data(());

    let shader = device.create_shader_module(ShaderModuleDescriptor {
        label: Some("postprocess.wgsl"),
        source: ShaderSource::Wgsl(Cow::Borrowed(include_str!("../shaders/postprocess.wgsl"))),
    });

    let gradient_bytes = include_bytes!("../images/gradient.png");
    let gradient_image = image::load_from_memory(gradient_bytes).unwrap();
    let gradient_rgba = gradient_image.as_rgba8().unwrap();

    use image::GenericImageView;
    let dimensions = gradient_image.dimensions();

    let texture_size = wgpu::Extent3d {
        width: dimensions.0,
        height: dimensions.1,
        depth_or_array_layers: 1,
    };

    let gradient_texture = device.create_texture(&TextureDescriptor {
        size: texture_size,
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D1,
        format: TextureFormat::Rgba8UnormSrgb,
        usage: TextureUsages::TEXTURE_BINDING | TextureUsages::COPY_DST,
        label: Some("gradient_texture"),
        view_formats: &vec![],
    });

    queue.write_texture(
        wgpu::ImageCopyTexture {
            texture: &gradient_texture,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: TextureAspect::All,
        },
        gradient_rgba,
        wgpu::ImageDataLayout {
            offset: 0,
            bytes_per_row: Some(4 * dimensions.0),
            rows_per_image: Some(dimensions.1),
        },
        texture_size,
    );

    let gradient_texture_view = gradient_texture.create_view(&TextureViewDescriptor::default());
    let gradient_sampler = device.create_sampler(&SamplerDescriptor {
        address_mode_u: AddressMode::ClampToEdge,
        address_mode_v: AddressMode::ClampToEdge,
        address_mode_w: AddressMode::ClampToEdge,
        mag_filter: FilterMode::Linear,
        min_filter: FilterMode::Linear,
        mipmap_filter: FilterMode::Linear, // TODO: mip map gradient
        ..Default::default()
    });

    let gradient_bind_group_layout =
        device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            entries: &[
                BindGroupLayoutEntry {
                    binding: 0,
                    visibility: ShaderStages::FRAGMENT,
                    ty: BindingType::Texture {
                        multisampled: false,
                        sample_type: TextureSampleType::Float { filterable: true },
                        view_dimension: TextureViewDimension::D1,
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
        });

    let gradient_bind_group = device.create_bind_group(&BindGroupDescriptor {
        layout: &gradient_bind_group_layout,
        entries: &[
            BindGroupEntry {
                binding: 0,
                resource: BindingResource::TextureView(&gradient_texture_view),
            },
            BindGroupEntry {
                binding: 1,
                resource: BindingResource::Sampler(&gradient_sampler),
            },
        ],
        label: None,
    });

    let blend_replace = wgpu::BlendComponent {
        src_factor: wgpu::BlendFactor::One,
        dst_factor: wgpu::BlendFactor::Zero,
        operation: wgpu::BlendOperation::Add,
    };

    let blend_state_replace = wgpu::BlendState {
        color: blend_replace,
        alpha: blend_replace,
    };

    let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
        label: Some("postprocess pipeline"),
        bind_group_layouts: &[
            &data.accumulation_bind_group_layout,
            &gradient_bind_group_layout,
        ],
        push_constant_ranges: &[],
    });

    let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some("postprocess"),
        layout: Some(&pipeline_layout),
        vertex: wgpu::VertexState {
            module: &shader,
            entry_point: "vs_main",
            buffers: &[wgpu::VertexBufferLayout {
                array_stride: 2 * 2 * 4,
                step_mode: wgpu::VertexStepMode::Vertex,
                attributes: &wgpu::vertex_attr_array![0 => Float32x2, 1 => Float32x2],
            }],
        },
        fragment: Some(wgpu::FragmentState {
            module: &shader,
            entry_point: "fs_main",
            targets: &[Some(wgpu::ColorTargetState {
                format: *db.swapchain_format(()),
                blend: Some(blend_state_replace),
                write_mask: wgpu::ColorWrites::ALL,
            })],
        }),
        primitive: wgpu::PrimitiveState::default(),
        depth_stencil: None,
        multisample: wgpu::MultisampleState::default(),
        multiview: None,
    });

    Data {
        gradient_bind_group,
        quad: MeshData::new(&device, &build_quad(), "Quad Vertex Buffer"),
        pipeline,
    }
    .into()
}

/// Draws a source accumulation texture into dst with log density coloring
pub fn render(
    db: &dyn Postprocesser,
    encoder: &mut wgpu::CommandEncoder,
    src: &wgpu::BindGroup,
    dst: &wgpu::TextureView,
) {
    let data = db.postprocess_data(());

    let mut postprocess_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
        label: Some("Postprocess render pass"),
        color_attachments: &[Some(wgpu::RenderPassColorAttachment {
            view: dst,
            resolve_target: None,
            ops: wgpu::Operations {
                load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                store: wgpu::StoreOp::Store,
            },
        })],
        depth_stencil_attachment: None,
        occlusion_query_set: None,
        timestamp_writes: None,
    });

    postprocess_pass.set_pipeline(&data.pipeline);
    postprocess_pass.set_bind_group(0, src, &[]);
    postprocess_pass.set_bind_group(1, &data.gradient_bind_group, &[]);
    postprocess_pass.set_vertex_buffer(0, data.quad.buffer.slice(..));
    postprocess_pass.draw(0..(data.quad.count), 0..1);
}
