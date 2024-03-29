use std::rc::Rc;
use wgpu::{Device, Queue, TextureFormat, TextureViewDescriptor};
use winit::dpi::PhysicalSize;

use crate::{
    accumulate::{self, AccumulateStorage, Accumulator},
    flame::Root,
    postprocess, ui,
    util_types::{DebugIt, PtrRc},
};

#[salsa::query_group(InputStorage2)]
pub trait Inputs2: salsa::Database {
    #[salsa::input]
    fn window_size(&self, key: ()) -> PhysicalSize<u32>;

    #[salsa::input]
    fn swapchain_format(&self, key: ()) -> DebugIt<TextureFormat>;

    #[salsa::input]
    fn queue(&self, key: ()) -> Rc<Queue>;
}

#[salsa::query_group(InputStorage)]
pub trait Inputs: salsa::Database {
    #[salsa::input]
    fn device(&self, key: ()) -> Rc<Device>;

    #[salsa::input]
    fn config(&self, key: ()) -> ui::Settings;
}

#[salsa::query_group(RendererStorage)]
pub trait Renderer: Inputs {
    fn root(&self, key: ()) -> Root;
}

#[salsa::query_group(PostprocesserStorage)]
pub trait Postprocesser: Accumulator + Inputs2 {
    fn postprocess_data(&self, key: ()) -> PtrRc<postprocess::Data>;
}

fn postprocess_data(db: &dyn Postprocesser, (): ()) -> PtrRc<postprocess::Data> {
    postprocess::data(db, ())
}

#[salsa::database(
    RendererStorage,
    InputStorage,
    InputStorage2,
    AccumulateStorage,
    PostprocesserStorage
)]
#[derive(Default)]
pub struct DatabaseStruct {
    storage: salsa::Storage<Self>,
}

impl salsa::Database for DatabaseStruct {}

fn root(db: &dyn Renderer, (): ()) -> Root {
    db.config(()).get_state()
}

pub fn render(
    db: &DatabaseStruct,
    frame: &wgpu::SurfaceTexture,
    mut encoder: &mut wgpu::CommandEncoder,
) {
    let accumulate = db.pass(accumulate::PassKey {
        resolution: db.window_size(()),
        filter: false,
    });
    let bind_group = accumulate.render(db, &mut encoder);
    postprocess::render(
        db,
        &mut encoder,
        bind_group,
        &frame.texture.create_view(&TextureViewDescriptor::default()),
    );
    // TODO: debug option to draw intermediate texture to screen at actual resolution
}
