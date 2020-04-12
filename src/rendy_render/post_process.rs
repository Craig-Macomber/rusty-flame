use na::Point2;

use rendy::{
    command::{QueueId, RenderPassEncoder},
    core::types::Layout,
    factory::Factory,
    graph::{
        render::{PrepareResult, SimpleGraphicsPipeline, SimpleGraphicsPipelineDesc},
        GraphContext, ImageAccess, NodeBuffer, NodeImage,
    },
    hal::{self, device::Device},
    memory::Dynamic,
    mesh::{AsVertex, TexCoord},
    resource::{
        Buffer, BufferInfo, DescriptorSet, DescriptorSetLayout, Escape, Filter, Handle, ImageView,
        ImageViewInfo, Sampler, SamplerDesc, ViewKind, WrapMode,
    },
    shader::{PathBufShaderInfo, ShaderKind, SourceLanguage, SpirvReflection, SpirvShader},
};

use crate::geometry;
use crate::rendy_render::ACCUMULATION_FORMAT;

lazy_static::lazy_static! {
    static ref VERTEX: SpirvShader = PathBufShaderInfo::new(
        std::path::PathBuf::from(concat!(env!("CARGO_MANIFEST_DIR"), "/shaders/basic.vert")),
        ShaderKind::Vertex,
        SourceLanguage::GLSL,
        "main",
    ).precompile().unwrap();

    static ref FRAGMENT: SpirvShader = PathBufShaderInfo::new(
        std::path::PathBuf::from(concat!(env!("CARGO_MANIFEST_DIR"), "/shaders/post_process.frag")),
        ShaderKind::Fragment,
        SourceLanguage::GLSL,
        "main",
    ).precompile().unwrap();

    static ref SHADERS: rendy::shader::ShaderSetBuilder = rendy::shader::ShaderSetBuilder::default()
        .with_vertex(&*VERTEX).unwrap()
        .with_fragment(&*FRAGMENT).unwrap();

    static ref SHADER_REFLECTION: SpirvReflection = SHADERS.reflect().unwrap();
}

#[derive(Debug, Default)]
pub struct PipelineDesc;

#[derive(Debug)]
pub struct Pipeline<B: hal::Backend> {
    buffer: Escape<Buffer<B>>,
    descriptors: Escape<DescriptorSet<B>>,
    image_sampler: Escape<Sampler<B>>,
    image_view: Escape<ImageView<B>>,
}

impl<B> SimpleGraphicsPipelineDesc<B, Point2<f64>> for PipelineDesc
where
    B: gfx_hal::Backend,
{
    type Pipeline = Pipeline<B>;

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
        vec![SHADER_REFLECTION
            .attributes(&["vertex_position"])
            .unwrap()
            .gfx_vertex_input_desc(hal::pso::VertexInputRate::Vertex)]
    }

    fn layout(&self) -> Layout {
        SHADER_REFLECTION.layout().unwrap()
    }

    fn colors(&self) -> Vec<hal::pso::ColorBlendDesc> {
        vec![
            hal::pso::ColorBlendDesc {
                mask: hal::pso::ColorMask::ALL,
                blend: Some(hal::pso::BlendState::REPLACE)
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
        _aux: &Point2<f64>,
    ) -> rendy::shader::ShaderSet<B> {
        SHADERS.build(factory, Default::default()).unwrap()
    }

    fn build(
        self,
        ctx: &GraphContext<B>,
        factory: &mut Factory<B>,
        _queue: QueueId,
        _aux: &Point2<f64>,
        buffers: Vec<NodeBuffer>,
        images: Vec<NodeImage>,
        set_layouts: &[Handle<DescriptorSetLayout<B>>],
    ) -> Result<Pipeline<B>, gfx_hal::pso::CreationError> {
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
            .create_sampler(SamplerDesc::new(Filter::Nearest, WrapMode::Clamp))
            .unwrap();

        let corners = geometry::Rect {
            min: na::Point2::new(-1.0, -1.0),
            max: na::Point2::new(1.0, 1.0),
        }
        .corners();
        let verts: Vec<TexCoord> = [
            corners[0], corners[1], corners[2], corners[0], corners[2], corners[3],
        ]
        .iter()
        .map(|c| [c.x as f32, c.y as f32].into())
        .collect();

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

        unsafe {
            factory
                .upload_visible_buffer(&mut vertex_buffer, 0, &verts)
                .unwrap();
        }

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

        Ok(Pipeline {
            buffer: vertex_buffer,
            descriptors: descriptor_set,
            image_view: view,
            image_sampler,
        })
    }
}

impl<B> SimpleGraphicsPipeline<B, Point2<f64>> for Pipeline<B>
where
    B: gfx_hal::Backend,
{
    type Desc = PipelineDesc;

    fn prepare(
        &mut self,
        _factory: &Factory<B>,
        _queue: QueueId,
        _set_layouts: &[Handle<DescriptorSetLayout<B>>],
        _index: usize,
        _aux: &Point2<f64>,
    ) -> PrepareResult {
        PrepareResult::DrawReuse
    }

    fn draw(
        &mut self,
        layout: &B::PipelineLayout,
        mut encoder: RenderPassEncoder<'_, B>,
        _index: usize,
        _aux: &Point2<f64>,
    ) {
        unsafe {
            encoder.bind_graphics_descriptor_sets(
                layout,
                0,
                std::iter::once(self.descriptors.raw()),
                std::iter::empty(),
            );

            encoder.bind_vertex_buffers(0, std::iter::once((self.buffer.raw(), 0)));
            encoder.draw(0..6, 0..1);
        }
    }

    fn dispose(self, _factory: &mut Factory<B>, _aux: &Point2<f64>) {}
}
