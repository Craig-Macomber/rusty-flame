use std::borrow::Cow;
use wgpu::{
    AddressMode, BindGroupDescriptor, BindGroupEntry, BindGroupLayout, BindGroupLayoutEntry,
    BindingResource, BindingType, FilterMode, PipelineLayoutDescriptor, SamplerDescriptor,
    ShaderModule, ShaderModuleDescriptor, ShaderSource, ShaderStage, TextureDescriptor,
    TextureFormat, TextureSampleType, TextureUsage, TextureViewDescriptor, TextureViewDimension,
};

use crate::{mesh::build_quad, render_common::MeshData, util_types::PtrRc, wgpu_render::Renderer};

/// Device dependant, but otherwise constant data.
#[derive(Debug)]
pub struct PostProcessData {
    shader: ShaderModule,
    gradient_bind_group: wgpu::BindGroup,
    gradient_bind_group_layout: BindGroupLayout,
    quad: MeshData,
    pipeline: wgpu::RenderPipeline,
}

pub fn data(db: &dyn Renderer, (): ()) -> PtrRc<PostProcessData> {
    let device = db.device(());
    let queue = db.queue(());
    let data = db.data(());

    let shader = device.create_shader_module(&ShaderModuleDescriptor {
        label: Some("postprocess.wgsl"),
        source: ShaderSource::Wgsl(Cow::Borrowed(include_str!("../shaders/postprocess.wgsl"))),
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

    let diffuse_texture = device.create_texture(&TextureDescriptor {
        size: texture_size,
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D1,
        format: TextureFormat::Rgba8UnormSrgb,
        usage: TextureUsage::SAMPLED | TextureUsage::COPY_DST,
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

    let diffuse_texture_view = diffuse_texture.create_view(&TextureViewDescriptor::default());
    let diffuse_sampler = device.create_sampler(&SamplerDescriptor {
        address_mode_u: AddressMode::ClampToEdge,
        address_mode_v: AddressMode::ClampToEdge,
        address_mode_w: AddressMode::ClampToEdge,
        mag_filter: FilterMode::Linear,
        min_filter: FilterMode::Nearest,
        mipmap_filter: FilterMode::Nearest,
        ..Default::default()
    });

    let gradient_bind_group_layout =
        device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            entries: &[
                BindGroupLayoutEntry {
                    binding: 0,
                    visibility: ShaderStage::FRAGMENT,
                    ty: BindingType::Texture {
                        multisampled: false,
                        sample_type: TextureSampleType::Float { filterable: true },
                        view_dimension: TextureViewDimension::D1,
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

    let gradient_bind_group = device.create_bind_group(&BindGroupDescriptor {
        layout: &gradient_bind_group_layout,
        entries: &[
            BindGroupEntry {
                binding: 0,
                resource: BindingResource::TextureView(&diffuse_texture_view),
            },
            BindGroupEntry {
                binding: 1,
                resource: BindingResource::Sampler(&diffuse_sampler),
            },
        ],
        label: None,
    });

    let blend_replace = wgpu::BlendState {
        src_factor: wgpu::BlendFactor::One,
        dst_factor: wgpu::BlendFactor::Zero,
        operation: wgpu::BlendOperation::Add,
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
                step_mode: wgpu::InputStepMode::Vertex,
                attributes: &wgpu::vertex_attr_array![0 => Float2, 1 => Float2],
            }],
        },
        fragment: Some(wgpu::FragmentState {
            module: &shader,
            entry_point: "fs_main",
            targets: &[wgpu::ColorTargetState {
                format: *db.swapchain_format(()),
                color_blend: blend_replace.clone(),
                alpha_blend: blend_replace.clone(),
                write_mask: wgpu::ColorWrite::ALL,
            }],
        }),
        primitive: wgpu::PrimitiveState::default(),
        depth_stencil: None,
        multisample: wgpu::MultisampleState::default(),
    });

    PostProcessData {
        shader,
        gradient_bind_group,
        gradient_bind_group_layout,
        quad: MeshData::new(&device, &build_quad(), "Quad Vertex Buffer"),
        pipeline,
    }
    .into()
}

pub fn render(
    db: &dyn Renderer,
    frame: &wgpu::SwapChainTexture,
    encoder: &mut wgpu::CommandEncoder,
) {
    let plan = db.sized_plan(());
    let data = db.postprocess_data(());

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

    postprocess_pass.set_pipeline(&data.pipeline);
    postprocess_pass.set_bind_group(0, &plan.large_bind_group, &[]);
    postprocess_pass.set_bind_group(1, &data.gradient_bind_group, &[]);
    postprocess_pass.set_vertex_buffer(0, data.quad.buffer.slice(..));
    postprocess_pass.draw(0..(data.quad.count), 0..1);
}
