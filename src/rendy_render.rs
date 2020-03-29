use crate::geometry;
use na::{Affine2, Matrix3, Point2, Vector2};

use rendy::{
    command::{Families, QueueId, RenderPassEncoder},
    factory::{Config, Factory},
    graph::{
        present::PresentNode,
        render::{
            PrepareResult, RenderGroupBuilder, SimpleGraphicsPipeline, SimpleGraphicsPipelineDesc,
        },
        Graph, GraphBuilder, GraphContext, NodeBuffer, NodeImage,
    },
    hal::{self},
    init::winit::{
        dpi::Size,
        event::{Event, WindowEvent},
        event_loop::{ControlFlow, EventLoop},
        window::{Window, WindowBuilder},
    },
    init::AnyWindowedRendy,
    memory::Dynamic,
    mesh::{AsVertex, TexCoord},
    resource::{Buffer, BufferInfo, DescriptorSet, DescriptorSetLayout, Escape, Handle},
    shader::{PathBufShaderInfo, ShaderKind, SourceLanguage, SpirvShader},
};

use crate::flame::State;

use crate::flame::BoundedState;
use crate::{get_state, process_scene, BASE_LEVELS, INSTANCE_LEVELS};

use geometry::Rect;
use rendy::shader::SpirvReflection;
use rendy_core::types::{vertex::Tangent, Layout};

type Backend = rendy::vulkan::Backend;

lazy_static::lazy_static! {
    static ref VERTEX_OLD: SpirvShader = PathBufShaderInfo::new(
        std::path::PathBuf::from(concat!(env!("CARGO_MANIFEST_DIR"), "/shaders/triangle/tri.vert")),
        ShaderKind::Vertex,
        SourceLanguage::GLSL,
        "main",
    ).precompile().unwrap();

    static ref VERTEX: SpirvShader = PathBufShaderInfo::new(
        std::path::PathBuf::from(concat!(env!("CARGO_MANIFEST_DIR"), "/shaders/sprite.vert")),
        ShaderKind::Vertex,
        SourceLanguage::GLSL,
        "main",
    ).precompile().unwrap();

    static ref FRAGMENT: SpirvShader = PathBufShaderInfo::new(
        std::path::PathBuf::from(concat!(env!("CARGO_MANIFEST_DIR"), "/shaders/triangle/tri.frag")),
        ShaderKind::Fragment,
        SourceLanguage::GLSL,
        "main",
    ).precompile().unwrap();

    static ref SHADERS: rendy::shader::ShaderSetBuilder = rendy::shader::ShaderSetBuilder::default()
        .with_vertex(&*VERTEX).unwrap()
        .with_fragment(&*FRAGMENT).unwrap();

    static ref SHADER_REFLECTION: SpirvReflection = SHADERS.reflect().unwrap();
}

pub fn main() {
    let config: Config = Default::default();
    let event_loop = EventLoop::new();
    let window = WindowBuilder::new()
        .with_inner_size(Size::Physical((3000, 2000).into()))
        .with_title("Rusty Flame");

    let rendy = AnyWindowedRendy::init_auto(&config, window, &event_loop).unwrap();
    rendy::with_any_windowed_rendy!((rendy)
        (mut factory, mut families, surface, window) => {
            let mut cursor = na::Point2::new(0.0, 0.0);
            let mut graph = Some(build_graph(&mut factory, &mut families, surface, &window, cursor));

            let started = std::time::Instant::now();

            let mut frame = 0u64;
            let mut elapsed = started.elapsed();

            event_loop.run(move |event, _, control_flow| {
                *control_flow = ControlFlow::Poll;
                match event {
                    Event::WindowEvent { event, .. } => match event {
                        WindowEvent::CloseRequested => *control_flow = ControlFlow::Exit,
                        WindowEvent::Resized(_dims) => {
                            let started = std::time::Instant::now();
                            //graph.take().unwrap().dispose(&mut factory, &cursor);
                            log::trace!("Graph disposed in: {:?}", started.elapsed());
                            return;
                        }
                        WindowEvent::CursorMoved { position, .. } => {
                            let size = window.inner_size();
                            cursor = Point2::new(
                                (position.x as f64)/ (size.width as f64) * 2.0 - 1.0,
                                (position.y as f64)/ (size.height as f64) * 2.0 - 1.0,
                            );
                        }
                        _ => {}
                    },
                    Event::MainEventsCleared => {
                        factory.maintain(&mut families);
                        if let Some(ref mut graph) = graph {
                            graph.run(&mut factory, &mut families, &cursor);
                            frame += 1;
                        }

                        elapsed = started.elapsed();
                        if frame % 100 == 0 {
                            let elapsed_ns = elapsed.as_secs() * 1_000_000_000 + elapsed.subsec_nanos() as u64;

                            log::info!(
                                "Frame Time: {:?}. FPS: {}",
                                elapsed / frame as u32,
                                frame * 1_000_000_000 / elapsed_ns
                            );
                        }
                    }
                    _ => {}
                }

                if *control_flow == ControlFlow::Exit && graph.is_some() {
                    let elapsed_ns = elapsed.as_secs() * 1_000_000_000 + elapsed.subsec_nanos() as u64;

                    log::info!(
                        "Elapsed: {:?}. Frame Time: {:?}. Frames: {}. FPS: {}",
                        elapsed,
                        elapsed / frame as u32,
                        frame,
                        frame * 1_000_000_000 / elapsed_ns
                    );

                    graph.take().unwrap().dispose(&mut factory, &cursor);
                }
            });
        }
    );
}

fn build_graph(
    factory: &mut Factory<Backend>,
    families: &mut Families<Backend>,
    surface: rendy::wsi::Surface<Backend>,
    window: &Window,
    cursor: Point2<f64>,
) -> Graph<Backend, Point2<f64>> {
    let mut graph_builder = GraphBuilder::<Backend, Point2<f64>>::new();

    let size = window.inner_size();

    let color = graph_builder.create_image(
        gfx_hal::image::Kind::D2(size.width as u32, size.height as u32, 1, 1),
        1,
        factory.get_surface_format(&surface),
        Some(hal::command::ClearValue {
            color: hal::command::ClearColor {
                float32: [0.0, 0.0, 0.0, 0.0],
            },
        }),
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
    vertex_buffer: Escape<Buffer<B>>,
    vertex_count: u32,
    instance_buffer: Escape<Buffer<B>>,
    instance_count: u32,
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
        return SHADER_REFLECTION.layout().unwrap();
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
    ) -> Result<TriangleRenderPipeline<B>, gfx_hal::pso::CreationError> {
        assert!(buffers.is_empty());
        assert!(images.is_empty());
        assert!(set_layouts.is_empty());
        Ok(build_mesh(factory, aux))
    }
}

fn build_mesh<B: gfx_hal::Backend>(
    factory: &Factory<B>,
    aux: &Point2<f64>,
) -> TriangleRenderPipeline<B> {
    let root = get_state([aux.x, aux.y], [2.0, 2.0]);
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

    state.process_levels(BASE_LEVELS, &mut |state| {
        for t in tri_verts.iter() {
            let t2 = state.mat * t;
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

    let mut instances: Vec<Tangent> = vec![];
    state.process_levels(INSTANCE_LEVELS, &mut |state| {
        let m: Matrix3<f64> = (root_mat * state.mat).to_homogeneous();
        instances.push(Tangent {
            0: [
                *m.get((0, 0)).unwrap() as f32,
                *m.get((0, 1)).unwrap() as f32,
                *m.get((0, 2)).unwrap() as f32,
                0f32,
            ],
        });
        instances.push(Tangent {
            0: [
                *m.get((1, 0)).unwrap() as f32,
                *m.get((1, 1)).unwrap() as f32,
                *m.get((1, 2)).unwrap() as f32,
                0f32,
            ],
        });
    });
    let instance_count: u32 = instances.len() as u32 / 2;
    let mut instance_buffer = factory
        .create_buffer(
            BufferInfo {
                size: u64::from(Tangent::vertex().stride) * u64::from(instance_count * 2),
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

    TriangleRenderPipeline {
        vertex_buffer,
        vertex_count,
        instance_buffer,
        instance_count,
    }
}

impl<B> SimpleGraphicsPipeline<B, Point2<f64>> for TriangleRenderPipeline<B>
where
    B: gfx_hal::Backend,
{
    type Desc = TriangleRenderPipelineDesc;

    fn prepare(
        &mut self,
        factory: &Factory<B>,
        _queue: QueueId,
        _set_layouts: &[Handle<DescriptorSetLayout<B>>],
        _index: usize,
        aux: &Point2<f64>,
    ) -> PrepareResult {
        *self = build_mesh(factory, aux);
        PrepareResult::DrawRecord
    }

    fn draw(
        &mut self,
        _layout: &B::PipelineLayout,
        mut encoder: RenderPassEncoder<'_, B>,
        _index: usize,
        _aux: &Point2<f64>,
    ) {
        unsafe {
            encoder.bind_vertex_buffers(0, std::iter::once((self.vertex_buffer.raw(), 0)));
            encoder.bind_vertex_buffers(1, std::iter::once((self.instance_buffer.raw(), 0)));
            encoder.draw(0..self.vertex_count, 0..self.instance_count);
        }
    }

    fn dispose(self, _factory: &mut Factory<B>, _aux: &Point2<f64>) {}
}
