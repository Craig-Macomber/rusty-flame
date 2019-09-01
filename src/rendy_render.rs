use na::{Point2, Vector2};
use rendy::{
    command::{Families, QueueId, RenderPassEncoder},
    factory::{Config, Factory},
    graph::{
        present::PresentNode, render::*, Graph, GraphBuilder, GraphContext, NodeBuffer, NodeImage,
    },
    hal,
    memory::Dynamic,
    mesh::{AsVertex, TexCoord},
    resource::Buffer,
    resource::{BufferInfo, DescriptorSetLayout, Escape, Handle},
    shader::{PathBufShaderInfo, ShaderKind, SourceLanguage},
    wsi::winit::{Event, EventsLoop, Window, WindowBuilder, WindowEvent},
};

use crate::flame::BoundedState;
use crate::{get_state, process_scene};

type Backend = rendy::vulkan::Backend;

lazy_static::lazy_static! {
    static ref VERTEX: PathBufShaderInfo = PathBufShaderInfo::new(
        std::path::PathBuf::from(concat!(env!("CARGO_MANIFEST_DIR"), "/shaders/triangle/tri.vert")),
        ShaderKind::Vertex,
        SourceLanguage::GLSL,
        "main",
    );

    static ref FRAGMENT: PathBufShaderInfo = PathBufShaderInfo::new(
        std::path::PathBuf::from(concat!(env!("CARGO_MANIFEST_DIR"), "/shaders/triangle/tri.frag")),
        ShaderKind::Fragment,
        SourceLanguage::GLSL,
        "main",
    );

    static ref SHADERS: rendy::shader::ShaderSetBuilder = rendy::shader::ShaderSetBuilder::default()
        .with_vertex(&*VERTEX).unwrap()
        .with_fragment(&*FRAGMENT).unwrap();
}

pub fn main() {
    let config: Config = Default::default();

    let (mut factory, mut families): (Factory<Backend>, _) = rendy::factory::init(config).unwrap();

    let mut event_loop = EventsLoop::new();

    let window = WindowBuilder::new()
        .with_title("Rusty Flame")
        .build(&event_loop)
        .unwrap();

    event_loop.poll_events(|_| ());

    let mut cursor = Point2::new(0.0, 0.0);

    loop {
        factory.maintain(&mut families);

        let mut should_close = false;
        event_loop.poll_events(|event| match event {
            Event::WindowEvent {
                event: WindowEvent::CloseRequested,
                ..
            } => {
                should_close = true;
            }
            Event::WindowEvent {
                event: WindowEvent::CursorMoved { position, .. },
                ..
            } => {
                let size = window.get_inner_size().unwrap();
                cursor = Point2::new(
                    position.x / size.width * 2.0 - 1.0,
                    position.y / size.height * 2.0 - 1.0,
                );
            }
            _ => {}
        });
        if should_close {
            break;
        }
        let mut graph = make_graph(&mut factory, &mut families, &window, cursor);
        graph.run(&mut factory, &mut families, &cursor);
        graph.dispose(&mut factory, &cursor);
    }
}

fn make_graph(
    factory: &mut Factory<Backend>,
    families: &mut Families<Backend>,
    window: &Window,
    cursor: Point2<f64>,
) -> Graph<Backend, Point2<f64>> {
    let surface = factory.create_surface(window);

    let mut graph_builder = GraphBuilder::<Backend, Point2<f64>>::new();

    let size = window
        .get_inner_size()
        .unwrap()
        .to_physical(window.get_hidpi_factor());

    let color = graph_builder.create_image(
        gfx_hal::image::Kind::D2(size.width as u32, size.height as u32, 1, 1),
        1,
        factory.get_surface_format(&surface),
        Some(gfx_hal::command::ClearValue::Color(
            [0.0, 0.0, 0.0, 0.0].into(),
        )),
    );

    let pass = graph_builder.add_node(
        TriangleRenderPipeline::builder()
            .into_subpass()
            .with_color(color)
            .into_pass(),
    );

    graph_builder.add_node(PresentNode::builder(&factory, surface, color).with_dependency(pass));

    graph_builder.build(factory, families, &cursor).unwrap()
}

#[derive(Debug, Default)]
struct TriangleRenderPipelineDesc;

#[derive(Debug)]
struct TriangleRenderPipeline<B: gfx_hal::Backend> {
    vertex: Escape<Buffer<B>>,
    vertex_count: u32,
}

impl<B> SimpleGraphicsPipelineDesc<B, Point2<f64>> for TriangleRenderPipelineDesc
where
    B: gfx_hal::Backend,
{
    type Pipeline = TriangleRenderPipeline<B>;

    fn vertices(
        &self,
    ) -> Vec<(
        Vec<gfx_hal::pso::Element<gfx_hal::format::Format>>,
        gfx_hal::pso::ElemStride,
        gfx_hal::pso::VertexInputRate,
    )> {
        vec![TexCoord::vertex().gfx_vertex_input_desc(gfx_hal::pso::VertexInputRate::Vertex)]
    }

    fn colors(&self) -> Vec<hal::pso::ColorBlendDesc> {
        vec![hal::pso::ColorBlendDesc(hal::pso::ColorMask::ALL, hal::pso::BlendState::ADD,); 1]
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
        _ctx: &GraphContext<B>,
        factory: &mut Factory<B>,
        _queue: QueueId,
        aux: &Point2<f64>,
        buffers: Vec<NodeBuffer>,
        images: Vec<NodeImage>,
        set_layouts: &[Handle<DescriptorSetLayout<B>>],
    ) -> Result<TriangleRenderPipeline<B>, failure::Error> {
        assert!(buffers.is_empty());
        assert!(images.is_empty());
        assert!(set_layouts.is_empty());

        let mut verts: Vec<TexCoord> = vec![];
        let tri_verts = vec![
            Point2::new(0.0, -1.0),
            Point2::new(1.0, 1.0),
            Point2::new(-1.0, 1.0),
        ];

        let root = get_state([aux.x, aux.y], [2.0, 2.0]);
        let state = root.get_state();
        let bounds = state.get_bounds();
        let scale = 1.0 / f64::max(bounds.width(), bounds.height());

        process_scene(state, &mut |state| {
            for t in &tri_verts {
                let t2 = (state.mat * t - bounds.min) * scale * 2.0 - Vector2::new(1.0, 1.0);
                verts.push([t2.x as f32, t2.y as f32].into());
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

        unsafe {
            factory
                .upload_visible_buffer(&mut vertex_buffer, 0, &verts)
                .unwrap();
        }

        Ok(TriangleRenderPipeline {
            vertex: vertex_buffer,
            vertex_count,
        })
    }
}

impl<B> SimpleGraphicsPipeline<B, Point2<f64>> for TriangleRenderPipeline<B>
where
    B: gfx_hal::Backend,
{
    type Desc = TriangleRenderPipelineDesc;

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
        _layout: &B::PipelineLayout,
        mut encoder: RenderPassEncoder<'_, B>,
        _index: usize,
        _aux: &Point2<f64>,
    ) {
        unsafe {
            encoder.bind_vertex_buffers(0, Some((self.vertex.raw(), 0)));
            encoder.draw(0..self.vertex_count, 0..1);
        }
    }

    fn dispose(self, _factory: &mut Factory<B>, _aux: &Point2<f64>) {}
}
