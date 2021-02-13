mod mesh_pipeline;
mod post_process;

use na::Point2;

use rendy::{
    command::Families,
    factory::{Config, Factory},
    graph::{
        present::PresentNode,
        render::{RenderGroupBuilder, SimpleGraphicsPipeline},
        Graph, GraphBuilder,
    },
    hal,
    init::winit::{
        dpi::Size,
        event::{Event, WindowEvent},
        event_loop::{ControlFlow, EventLoop},
        window::{Window, WindowBuilder},
    },
    init::AnyWindowedRendy,
};

use crate::SMALL_ACCUMULATION_BUFFER_SIZE;

type Backend = rendy::vulkan::Backend;

pub fn main() {
    simple_logger::SimpleLogger::new().init().unwrap();
    let config: Config = Default::default();
    let event_loop = EventLoop::new();
    let window = WindowBuilder::new()
        .with_inner_size(Size::Physical((3000, 2000).into()))
        .with_title("Rusty Flame");

    let rendy = AnyWindowedRendy::init_auto(&config, window, &event_loop).unwrap();
    rendy::with_any_windowed_rendy!((rendy)
        (mut factory, mut families, surface, window) => {
            let mut scene = SceneState{cursor: na::Point2::new(0.0, 0.0)};
            let mut graph = Some(build_graph(&mut factory, &mut families, surface, &window, &scene));
            let mut started = std::time::Instant::now();
            let mut frame = 0u64;

            event_loop.run(move |event, _, control_flow| {
                if *control_flow == ControlFlow::Exit {
                    return; // TODO: why is this needed?
                }
                *control_flow = ControlFlow::Poll;
                match event {
                    Event::WindowEvent { event, .. } => match event {
                        WindowEvent::CloseRequested => *control_flow = ControlFlow::Exit,
                        WindowEvent::Resized(_dims) => {
                            // TODO: support resize: https://github.com/amethyst/rendy/issues/54
                        }
                        WindowEvent::CursorMoved { position, .. } => {
                            let size = window.inner_size();
                            scene = SceneState{cursor: Point2::new(
                                f64::from(position.x) / f64::from(size.width) * 2.0 - 1.0,
                                f64::from(position.y) / f64::from(size.height) * 2.0 - 1.0,
                            )};
                        }
                        _ => {}
                    },
                    Event::MainEventsCleared => {
                        factory.maintain(&mut families);
                        graph.as_mut().unwrap().run(&mut factory, &mut families, &scene);
                        frame += 1;
                        if frame % 100 == 0 {
                            let elapsed = started.elapsed();
                            log::info!(
                                "Frame Time: {:?}. FPS: {:.3}",
                                elapsed / frame as u32,
                                frame as f64 / elapsed.as_secs_f64()
                            );
                            started = std::time::Instant::now();
                            frame = 0;
                        }
                    }
                    _ => {}
                }

                if *control_flow == ControlFlow::Exit {
                    graph.take().unwrap().dispose(&mut factory, &scene);
                }
            });
        }
    );
}

pub(crate) const ACCUMULATION_FORMAT: hal::format::Format = hal::format::Format::R32Sfloat;

pub(crate) struct SceneState {
    cursor: Point2<f64>,
}

const USE_TEXTURED_MESH: bool = true;

fn build_graph(
    factory: &mut Factory<Backend>,
    families: &mut Families<Backend>,
    surface: rendy::wsi::Surface<Backend>,
    window: &Window,
    cursor: &SceneState,
) -> Graph<Backend, SceneState> {
    let mut graph_builder = GraphBuilder::<Backend, SceneState>::new();

    let size = window.inner_size();
    let window_size = gfx_hal::image::Kind::D2(size.width as u32, size.height as u32, 1, 1);

    let accumulation_size = if USE_TEXTURED_MESH {
        gfx_hal::image::Kind::D2(
            SMALL_ACCUMULATION_BUFFER_SIZE,
            SMALL_ACCUMULATION_BUFFER_SIZE,
            1,
            1,
        )
    } else {
        window_size
    };

    let accumulation_image = graph_builder.create_image(
        accumulation_size,
        1,
        ACCUMULATION_FORMAT,
        Some(hal::command::ClearValue {
            color: hal::command::ClearColor {
                float32: [0.0, 0.0, 0.0, 0.0],
            },
        }),
    );

    let accumulation_node = graph_builder.add_node(
        mesh_pipeline::Pipeline::builder()
            .into_subpass()
            .with_color(accumulation_image)
            .into_pass(),
    );

    let (accumulation_node, accumulation_image) = if USE_TEXTURED_MESH {
        let accumulation_image2 = graph_builder.create_image(
            window_size,
            1,
            ACCUMULATION_FORMAT,
            Some(hal::command::ClearValue {
                color: hal::command::ClearColor {
                    float32: [0.0, 0.0, 0.0, 0.0],
                },
            }),
        );

        let accumulation_node2 = graph_builder.add_node(
            mesh_pipeline::PipelineTextured::builder()
                .with_image(accumulation_image)
                .into_subpass()
                .with_dependency(accumulation_node)
                .with_color(accumulation_image2)
                .into_pass(),
        );

        (accumulation_node2, accumulation_image2)
    } else {
        (accumulation_node, accumulation_image)
    };

    let output_image = graph_builder.create_image(
        window_size,
        1,
        factory.get_surface_format(&surface),
        Some(hal::command::ClearValue {
            color: hal::command::ClearColor {
                float32: [0.2, 0.0, 0.0, 0.0],
            },
        }),
    );

    let post_process_node = graph_builder.add_node(
        post_process::Pipeline::builder()
            .with_image(accumulation_image)
            .into_subpass()
            .with_dependency(accumulation_node)
            .with_color(output_image)
            .into_pass(),
    );

    graph_builder.add_node(
        PresentNode::builder(&factory, surface, output_image).with_dependency(post_process_node),
    );

    graph_builder.build(factory, families, &cursor).unwrap()
}
