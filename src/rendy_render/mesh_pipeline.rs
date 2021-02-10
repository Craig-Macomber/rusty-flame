use na::{Matrix3, Point2};

use crate::{
    flame::{BoundedState, State},
    geometry, get_state,
    rendy_render::ACCUMULATION_FORMAT,
    split_levels,
};
use rendy::{
    command::{QueueId, RenderPassEncoder},
    core::types::Layout,
    factory::Factory,
    graph::{
        render::{PrepareResult, SimpleGraphicsPipeline, SimpleGraphicsPipelineDesc},
        GraphContext, ImageAccess, NodeBuffer, NodeImage,
    },
    hal::{self},
    memory::Dynamic,
    mesh::{AsVertex, TexCoord},
    resource::{Buffer, BufferInfo, DescriptorSet, DescriptorSetLayout, Escape, Handle},
    shader::{PathBufShaderInfo, ShaderKind, SourceLanguage, SpirvReflection, SpirvShader},
};

use super::SceneState;
use rendy::{
    hal::device::Device,
    resource::{Filter, ImageViewInfo, SamplerDesc, ViewKind, WrapMode},
};

lazy_static::lazy_static! {
    static ref VERTEX: SpirvShader = PathBufShaderInfo::new(
        std::path::PathBuf::from(concat!(env!("CARGO_MANIFEST_DIR"), "/shaders/sprite.vert")),
        ShaderKind::Vertex,
        SourceLanguage::GLSL,
        "main",
    ).precompile().unwrap();

    static ref FRAGMENT: SpirvShader = PathBufShaderInfo::new(
        std::path::PathBuf::from(concat!(env!("CARGO_MANIFEST_DIR"), "/shaders/solid.frag")),
        ShaderKind::Fragment,
        SourceLanguage::GLSL,
        "main",
    ).precompile().unwrap();

    static ref SHADERS: rendy::shader::ShaderSetBuilder = rendy::shader::ShaderSetBuilder::default()
        .with_vertex(&*VERTEX).unwrap()
        .with_fragment(&*FRAGMENT).unwrap();

    static ref VERTEX_TEXTURED: SpirvShader = PathBufShaderInfo::new(
        std::path::PathBuf::from(concat!(env!("CARGO_MANIFEST_DIR"), "/shaders/sprite_textured.vert")),
        ShaderKind::Vertex,
        SourceLanguage::GLSL,
        "main",
    ).precompile().unwrap();

    static ref FRAGMENT_TEXTURED: SpirvShader = PathBufShaderInfo::new(
        std::path::PathBuf::from(concat!(env!("CARGO_MANIFEST_DIR"), "/shaders/textured.frag")),
        ShaderKind::Fragment,
        SourceLanguage::GLSL,
        "main",
    ).precompile().unwrap();

    static ref SHADERS_TEXTURED: rendy::shader::ShaderSetBuilder = rendy::shader::ShaderSetBuilder::default()
        .with_vertex(&*VERTEX_TEXTURED).unwrap()
        .with_fragment(&*FRAGMENT_TEXTURED).unwrap();

    static ref SHADER_REFLECTION: SpirvReflection = SHADERS.reflect().unwrap();

    static ref SHADER_REFLECTION_TEXTURED: SpirvReflection = SHADERS_TEXTURED.reflect().unwrap();
}

#[derive(Debug, Default)]
pub struct PipelineDesc;

#[derive(Debug, Default)]
pub struct PipelineDescTextured;

#[derive(Debug)]
pub struct Pipeline<B: gfx_hal::Backend> {
    data: Option<PipelineData<B>>,
}

#[derive(Debug)]
pub struct PipelineTextured<B: gfx_hal::Backend> {
    data: Option<PipelineData<B>>,
    descriptors: Escape<DescriptorSet<B>>,
}

#[derive(Debug)]
pub struct PipelineData<B: gfx_hal::Backend> {
    vertex_buffer: Escape<Buffer<B>>,
    vertex_count: u32,
    instance_buffer: Escape<Buffer<B>>,
    instance_count: u32,
    uv_buffer: Option<Escape<Buffer<B>>>,
    aux: Point2<f64>,
}

impl<B> SimpleGraphicsPipelineDesc<B, SceneState> for PipelineDesc
where
    B: gfx_hal::Backend,
{
    type Pipeline = Pipeline<B>;

    fn vertices(
        &self,
    ) -> Vec<(
        Vec<gfx_hal::pso::Element<gfx_hal::format::Format>>,
        gfx_hal::pso::ElemStride,
        gfx_hal::pso::VertexInputRate,
    )> {
        vec![
            SHADER_REFLECTION
                .attributes(&["vertex_position"])
                .unwrap()
                .gfx_vertex_input_desc(hal::pso::VertexInputRate::Vertex),
            SHADER_REFLECTION
                .attributes(&["sprite_position_1", "sprite_position_2"])
                .unwrap()
                .gfx_vertex_input_desc(hal::pso::VertexInputRate::Instance(1)),
        ]
    }

    fn layout(&self) -> Layout {
        SHADER_REFLECTION.layout().unwrap()
    }

    fn colors(&self) -> Vec<hal::pso::ColorBlendDesc> {
        vec![
            hal::pso::ColorBlendDesc {
                mask: hal::pso::ColorMask::ALL,
                blend: Some(hal::pso::BlendState::ADD)
            };
            1
        ]
    }

    fn depth_stencil(&self) -> Option<gfx_hal::pso::DepthStencilDesc> {
        None
    }

    fn load_shader_set(
        &self,
        factory: &mut Factory<B>,
        _aux: &SceneState,
    ) -> rendy::shader::ShaderSet<B> {
        SHADERS.build(factory, Default::default()).unwrap()
    }

    fn build(
        self,
        _ctx: &GraphContext<B>,
        _factory: &mut Factory<B>,
        _queue: QueueId,
        _aux: &SceneState,
        buffers: Vec<NodeBuffer>,
        images: Vec<NodeImage>,
        set_layouts: &[Handle<DescriptorSetLayout<B>>],
    ) -> Result<Pipeline<B>, gfx_hal::pso::CreationError> {
        assert!(buffers.is_empty());
        assert!(images.is_empty());
        assert!(set_layouts.is_empty());
        Ok(Pipeline { data: None })
    }
}

impl<B> SimpleGraphicsPipelineDesc<B, SceneState> for PipelineDescTextured
where
    B: gfx_hal::Backend,
{
    type Pipeline = PipelineTextured<B>;

    fn images(&self) -> Vec<ImageAccess> {
        vec![ImageAccess {
            access: hal::image::Access::SHADER_READ,
            usage: hal::image::Usage::SAMPLED,
            layout: hal::image::Layout::ShaderReadOnlyOptimal,
            stages: hal::pso::PipelineStage::FRAGMENT_SHADER,
        }]
    }

    fn vertices(
        &self,
    ) -> Vec<(
        Vec<gfx_hal::pso::Element<gfx_hal::format::Format>>,
        gfx_hal::pso::ElemStride,
        gfx_hal::pso::VertexInputRate,
    )> {
        vec![
            SHADER_REFLECTION_TEXTURED
                .attributes(&["vertex_position"])
                .unwrap()
                .gfx_vertex_input_desc(hal::pso::VertexInputRate::Vertex),
            SHADER_REFLECTION_TEXTURED
                .attributes(&["vertex_uv"])
                .unwrap()
                .gfx_vertex_input_desc(hal::pso::VertexInputRate::Vertex),
            SHADER_REFLECTION_TEXTURED
                .attributes(&["sprite_position_1", "sprite_position_2"])
                .unwrap()
                .gfx_vertex_input_desc(hal::pso::VertexInputRate::Instance(1)),
        ]
    }

    fn layout(&self) -> Layout {
        SHADER_REFLECTION_TEXTURED.layout().unwrap()
    }

    fn colors(&self) -> Vec<hal::pso::ColorBlendDesc> {
        vec![
            hal::pso::ColorBlendDesc {
                mask: hal::pso::ColorMask::ALL,
                blend: Some(hal::pso::BlendState::ADD)
            };
            1
        ]
    }

    fn depth_stencil(&self) -> Option<gfx_hal::pso::DepthStencilDesc> {
        None
    }

    fn load_shader_set(
        &self,
        factory: &mut Factory<B>,
        _aux: &SceneState,
    ) -> rendy::shader::ShaderSet<B> {
        SHADERS_TEXTURED.build(factory, Default::default()).unwrap()
    }

    fn build(
        self,
        ctx: &GraphContext<B>,
        factory: &mut Factory<B>,
        _queue: QueueId,
        _aux: &SceneState,
        buffers: Vec<NodeBuffer>,
        images: Vec<NodeImage>,
        set_layouts: &[Handle<DescriptorSetLayout<B>>],
    ) -> Result<PipelineTextured<B>, gfx_hal::pso::CreationError> {
        assert_eq!(buffers.len(), 0);
        assert_eq!(images.len(), 1);
        assert_eq!(set_layouts.len(), 1);

        let accumulation_image = ctx.get_image(images[0].id).unwrap();

        let view = factory
            .create_image_view(
                accumulation_image.clone(),
                ImageViewInfo {
                    view_kind: ViewKind::D2,
                    format: ACCUMULATION_FORMAT,
                    swizzle: hal::format::Swizzle::NO,
                    range: images[0].range.clone(),
                },
            )
            .expect("Could not create accumulation_image view");

        let image_sampler = factory
            .create_sampler(SamplerDesc::new(Filter::Linear, WrapMode::Clamp))
            .unwrap();

        let descriptor_set = factory
            .create_descriptor_set(set_layouts[0].clone())
            .unwrap();

        unsafe {
            factory.device().write_descriptor_sets(vec![
                hal::pso::DescriptorSetWrite {
                    set: descriptor_set.raw(),
                    binding: 1,
                    array_offset: 0,
                    descriptors: vec![hal::pso::Descriptor::Image(
                        view.raw(),
                        hal::image::Layout::ShaderReadOnlyOptimal,
                    )],
                },
                hal::pso::DescriptorSetWrite {
                    set: descriptor_set.raw(),
                    binding: 0,
                    array_offset: 0,
                    descriptors: vec![hal::pso::Descriptor::Sampler(image_sampler.raw())],
                },
            ]);
        }

        Ok(PipelineTextured {
            data: None,
            descriptors: descriptor_set,
        })
    }
}

fn build_mesh<B: gfx_hal::Backend>(
    factory: &Factory<B>,
    aux: &SceneState,
    textured: bool,
) -> PipelineData<B> {
    let root = get_state([aux.cursor.x + 1.0, aux.cursor.y + 1.0], [2.0, 2.0]);
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
        if textured {
            uv_verts.extend(uv_tri_verts.iter());
        }
    });
    let vertex_count: u32 = verts.len() as u32;

    let mut vertex_buffer = factory
        .create_buffer(
            BufferInfo {
                size: u64::from(TexCoord::vertex().stride) * u64::from(vertex_count),
                usage: gfx_hal::buffer::Usage::VERTEX,
            },
            Dynamic,
        )
        .unwrap();

    let mut instances: Vec<f32> = vec![];
    state.process_levels(split.instance, &mut |state| {
        let m: Matrix3<f64> = (root_mat * state.mat).to_homogeneous();
        let s = m.as_slice();
        instances.extend(
            [
                s[0] as f32,
                s[3] as f32,
                s[6] as f32,
                0f32,
                s[1] as f32,
                s[4] as f32,
                s[7] as f32,
                0f32,
            ]
            .iter(),
        );
    });
    let instance_count: u32 = instances.len() as u32 / 8;
    let mut instance_buffer = factory
        .create_buffer(
            BufferInfo {
                size: (instances.len() * std::mem::size_of::<f32>()) as u64,
                usage: gfx_hal::buffer::Usage::VERTEX,
            },
            Dynamic,
        )
        .unwrap();

    unsafe {
        factory
            .upload_visible_buffer(&mut vertex_buffer, 0, &verts)
            .unwrap();
        factory
            .upload_visible_buffer(&mut instance_buffer, 0, &instances)
            .unwrap();
    }

    let uv_buffer = if textured {
        let mut uv_buffer = factory
            .create_buffer(
                BufferInfo {
                    size: u64::from(TexCoord::vertex().stride) * u64::from(vertex_count),
                    usage: gfx_hal::buffer::Usage::VERTEX,
                },
                Dynamic,
            )
            .unwrap();

        unsafe {
            factory
                .upload_visible_buffer(&mut uv_buffer, 0, &uv_verts)
                .unwrap();
        }
        Some(uv_buffer)
    } else {
        None
    };

    PipelineData {
        vertex_buffer,
        vertex_count,
        instance_buffer,
        instance_count,
        uv_buffer,
        aux: aux.cursor.clone(),
    }
}

impl<B> SimpleGraphicsPipeline<B, SceneState> for Pipeline<B>
where
    B: gfx_hal::Backend,
{
    type Desc = PipelineDesc;

    fn prepare(
        &mut self,
        factory: &Factory<B>,
        _queue: QueueId,
        _set_layouts: &[Handle<DescriptorSetLayout<B>>],
        _index: usize,
        aux: &SceneState,
    ) -> PrepareResult {
        match self.data.as_ref() {
            None => {}
            Some(data) => {
                if data.aux == aux.cursor {
                    // TODO: why doesn't DrawReuse work here? Maybe related to index?
                    return PrepareResult::DrawRecord;
                }
            }
        }
        self.data = Some(build_mesh(factory, aux, false));
        PrepareResult::DrawRecord
    }

    fn draw(
        &mut self,
        _layout: &B::PipelineLayout,
        mut encoder: RenderPassEncoder<'_, B>,
        _index: usize,
        _aux: &SceneState,
    ) {
        let data = self.data.as_mut().unwrap();
        unsafe {
            encoder.bind_vertex_buffers(
                0,
                vec![
                    (data.vertex_buffer.raw(), 0),
                    (data.instance_buffer.raw(), 0),
                ],
            );
            encoder.draw(0..data.vertex_count, 0..data.instance_count);
        }
    }

    fn dispose(self, _factory: &mut Factory<B>, _aux: &SceneState) {}
}

impl<B> SimpleGraphicsPipeline<B, SceneState> for PipelineTextured<B>
where
    B: gfx_hal::Backend,
{
    type Desc = PipelineDescTextured;

    fn prepare(
        &mut self,
        factory: &Factory<B>,
        _queue: QueueId,
        _set_layouts: &[Handle<DescriptorSetLayout<B>>],
        _index: usize,
        aux: &SceneState,
    ) -> PrepareResult {
        match self.data.as_ref() {
            None => {}
            Some(data) => {
                if data.aux == aux.cursor {
                    // TODO: why doesn't DrawReuse work here? Maybe related to index?
                    return PrepareResult::DrawRecord;
                }
            }
        }
        self.data = Some(build_mesh(factory, aux, true));
        PrepareResult::DrawRecord
    }

    fn draw(
        &mut self,
        layout: &B::PipelineLayout,
        mut encoder: RenderPassEncoder<'_, B>,
        _index: usize,
        _aux: &SceneState,
    ) {
        let data = self.data.as_mut().unwrap();
        unsafe {
            encoder.bind_graphics_descriptor_sets(
                layout,
                0,
                std::iter::once(self.descriptors.raw()),
                std::iter::empty(),
            );
            encoder.bind_vertex_buffers(
                0,
                vec![
                    (data.vertex_buffer.raw(), 0),
                    (data.uv_buffer.as_ref().unwrap().raw(), 0),
                    (data.instance_buffer.raw(), 0),
                ],
            );
            encoder.draw(0..data.vertex_count, 0..data.instance_count);
        }
    }

    fn dispose(self, _factory: &mut Factory<B>, _aux: &SceneState) {}
}
