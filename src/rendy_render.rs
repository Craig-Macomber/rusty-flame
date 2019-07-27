use na::Point2;
use rendy::{
    command::{Families, QueueId, RenderPassEncoder},
    factory::{Config, Factory},
    graph::{
        present::PresentNode, render::*, Graph, GraphBuilder, GraphContext, NodeBuffer, NodeImage,
    },
    memory::Dynamic,
    mesh::{AsVertex, PosColor},
    resource::Buffer,
    resource::{BufferInfo, DescriptorSetLayout, Escape, Handle},
    shader::{PathBufShaderInfo, ShaderKind, SourceLanguage},
    wsi::winit::{Event, EventsLoop, WindowBuilder, WindowEvent},
};

use crate::process_scene;

type Backend = rendy::vulkan::Backend;

lazy_static::lazy_static! {
    static ref VERTEX: PathBufShaderInfo = PathBufShaderInfo::new(
        std::path::PathBuf::from(concat!(env!("CARGO_MANIFEST_DIR"), "/examples/shaders/triangle/tri.vert")),
        ShaderKind::Vertex,
        SourceLanguage::GLSL,
        "main",
    );

    static ref FRAGMENT: PathBufShaderInfo = PathBufShaderInfo::new(
        std::path::PathBuf::from(concat!(env!("CARGO_MANIFEST_DIR"), "/examples/shaders/triangle/tri.frag")),
        ShaderKind::Fragment,
        SourceLanguage::GLSL,
        "main",
    );

    static ref SHADERS: rendy::shader::ShaderSetBuilder = rendy::shader::ShaderSetBuilder::default()
        .with_vertex(&*VERTEX).unwrap()
        .with_fragment(&*FRAGMENT).unwrap();
}

pub fn main() {
    env_logger::Builder::from_default_env()
        .filter_level(log::LevelFilter::Warn)
        .filter_module("triangle", log::LevelFilter::Trace)
        .init();

    let config: Config = Default::default();

    let (mut factory, mut families): (Factory<Backend>, _) = rendy::factory::init(config).unwrap();

    let mut event_loop = EventsLoop::new();

    let window = WindowBuilder::new()
        .with_title("Rusty Flame")
        .build(&event_loop)
        .unwrap();

    event_loop.poll_events(|_| ());

    let surface = factory.create_surface(&window);

    let mut graph_builder = GraphBuilder::<Backend, ()>::new();

    let size = window
        .get_inner_size()
        .unwrap()
        .to_physical(window.get_hidpi_factor());

    let color = graph_builder.create_image(
        gfx_hal::image::Kind::D2(size.width as u32, size.height as u32, 1, 1),
        1,
        factory.get_surface_format(&surface),
        Some(gfx_hal::command::ClearValue::Color(
            [1.0, 1.0, 1.0, 1.0].into(),
        )),
    );

    let pass = graph_builder.add_node(
        TriangleRenderPipeline::builder()
            .into_subpass()
            .with_color(color)
            .into_pass(),
    );

    graph_builder.add_node(PresentNode::builder(&factory, surface, color).with_dependency(pass));

    let graph = graph_builder
        .build(&mut factory, &mut families, &())
        .unwrap();

    run(&mut event_loop, &mut factory, &mut families, graph).unwrap();
}

fn run(
    event_loop: &mut EventsLoop,
    factory: &mut Factory<Backend>,
    families: &mut Families<Backend>,
    mut graph: Graph<Backend, ()>,
) -> Result<(), failure::Error> {
    loop {
        factory.maintain(families);

        let mut should_close = false;
        event_loop.poll_events(|event| {
            if let Event::WindowEvent {
                event: WindowEvent::CloseRequested,
                ..
            } = event
            {
                should_close = true
            }
        });
        if should_close {
            break;
        }
        graph.run(factory, families, &());
    }

    graph.dispose(factory, &());
    Ok(())
}

#[derive(Debug, Default)]
struct TriangleRenderPipelineDesc;

#[derive(Debug)]
struct TriangleRenderPipeline<B: gfx_hal::Backend> {
    vertex: Escape<Buffer<B>>,
    vertex_count: u32,
}

impl<B, T> SimpleGraphicsPipelineDesc<B, T> for TriangleRenderPipelineDesc
where
    B: gfx_hal::Backend,
    T: ?Sized,
{
    type Pipeline = TriangleRenderPipeline<B>;

    fn vertices(
        &self,
    ) -> Vec<(
        Vec<gfx_hal::pso::Element<gfx_hal::format::Format>>,
        gfx_hal::pso::ElemStride,
        gfx_hal::pso::VertexInputRate,
    )> {
        vec![PosColor::vertex().gfx_vertex_input_desc(gfx_hal::pso::VertexInputRate::Vertex)]
    }

    fn depth_stencil(&self) -> Option<gfx_hal::pso::DepthStencilDesc> {
        None
    }

    fn load_shader_set(&self, factory: &mut Factory<B>, _aux: &T) -> rendy::shader::ShaderSet<B> {
        SHADERS.build(factory, Default::default()).unwrap()
    }

    fn build(
        self,
        _ctx: &GraphContext<B>,
        factory: &mut Factory<B>,
        _queue: QueueId,
        _aux: &T,
        buffers: Vec<NodeBuffer>,
        images: Vec<NodeImage>,
        set_layouts: &[Handle<DescriptorSetLayout<B>>],
    ) -> Result<TriangleRenderPipeline<B>, failure::Error> {
        assert!(buffers.is_empty());
        assert!(images.is_empty());
        assert!(set_layouts.is_empty());

        let mut verts = vec![];
        let tri_verts = vec![
            Point2::new(0.0, -1.0),
            Point2::new(1.0, 1.0),
            Point2::new(-1.0, 1.0)
        ];

        process_scene([0.0, 0.0], [2.0, 2.0], &mut |state| {
            for t in &tri_verts {
                let t2 = state.mat * t;
                verts.push(PosColor {
                    position: [t2.x as f32, t2.y as f32, 0.0].into(),
                    color: [1.0, 0.0, 0.0, 1.0].into(),
                });
            }
        });
        let vertex_count: u32 = verts.len() as u32;

        let mut vertex_buffer = factory
            .create_buffer(
                BufferInfo {
                    size: u64::from(PosColor::vertex().stride) * u64::from(vertex_count),
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
            vertex_count: vertex_count,
        })
    }
}

impl<B, T> SimpleGraphicsPipeline<B, T> for TriangleRenderPipeline<B>
where
    B: gfx_hal::Backend,
    T: ?Sized,
{
    type Desc = TriangleRenderPipelineDesc;

    fn prepare(
        &mut self,
        _factory: &Factory<B>,
        _queue: QueueId,
        _set_layouts: &[Handle<DescriptorSetLayout<B>>],
        _index: usize,
        _aux: &T,
    ) -> PrepareResult {
        PrepareResult::DrawReuse
    }

    fn draw(
        &mut self,
        _layout: &B::PipelineLayout,
        mut encoder: RenderPassEncoder<'_, B>,
        _index: usize,
        _aux: &T,
    ) {
        unsafe {
            encoder.bind_vertex_buffers(0, Some((self.vertex.raw(), 0)));
            encoder.draw(0..self.vertex_count, 0..1);
        }
    }

    fn dispose(self, _factory: &mut Factory<B>, _aux: &T) {}
}
