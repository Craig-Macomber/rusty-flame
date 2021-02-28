use std::rc::Rc;
use wgpu::{CommandEncoderDescriptor, Device, Queue, TextureFormat};
use winit::dpi::PhysicalSize;

use crate::{
    accumulate::{self, AccumulateStorage, Accumulator},
    flame::Root,
    get_state, postprocess,
    util_types::{DebugIt, PtrRc},
};

#[salsa::query_group(InputStorage)]
pub trait Inputs: salsa::Database {
    #[salsa::input]
    fn cursor(&self, key: ()) -> [f64; 2];

    #[salsa::input]
    fn window_size(&self, key: ()) -> PhysicalSize<u32>;

    #[salsa::input]
    fn device(&self, key: ()) -> Rc<Device>;

    #[salsa::input]
    fn queue(&self, key: ()) -> Rc<Queue>;

    #[salsa::input]
    fn swapchain_format(&self, key: ()) -> DebugIt<TextureFormat>;
}

#[salsa::query_group(RendererStorage)]
pub trait Renderer: Inputs {
    fn root(&self, key: ()) -> Root;
}

#[salsa::query_group(PostprocesserStorage)]
pub trait Postprocesser: Accumulator {
    fn postprocess_data(&self, key: ()) -> PtrRc<postprocess::Data>;
}

fn postprocess_data(db: &dyn Postprocesser, (): ()) -> PtrRc<postprocess::Data> {
    postprocess::data(db, ()).into()
}

#[salsa::database(RendererStorage, InputStorage, AccumulateStorage, PostprocesserStorage)]
#[derive(Default)]
pub struct DatabaseStruct {
    storage: salsa::Storage<Self>,
}

impl salsa::Database for DatabaseStruct {}

fn root(db: &dyn Renderer, (): ()) -> Root {
    get_state(db.cursor(()))
}

pub fn render(db: &DatabaseStruct, frame: &wgpu::SwapChainTexture) {
    let device = db.device(());

    let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor { label: None });
    {
        let plan = db.plan(());
        let accumulate = db.pass(accumulate::PassKey {
            plans: plan.passes.clone(),
            filter: false,
        });
        let bind_group = accumulate.render(db, &mut encoder);
        postprocess::render(db, &mut encoder, bind_group, &frame.view);
        // TODO: debug option to draw intermediate texture to screen at actual resolution
    }

    db.queue(()).submit(Some(encoder.finish()));
}
