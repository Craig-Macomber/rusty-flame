use geometry::Bounds;
use std::rc::Rc;
use wgpu::{CommandEncoderDescriptor, Device, Queue, TextureFormat};
use winit::dpi::PhysicalSize;

use crate::{
    accumulate::{self},
    flame::{BoundedState, Root},
    geometry::{self, Rect},
    get_state,
    plan::{plan_render, Plan},
    postprocess,
    render_common::MeshData,
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
    fn data(&self, key: ()) -> PtrRc<accumulate::DeviceData>;
    fn accumulate_pass(
        &self,
        key: accumulate::AccumulatePassKey,
    ) -> PtrRc<accumulate::AccumulatePass>;
    fn root(&self, key: ()) -> Root;
    fn mesh(&self, key: u32) -> PtrRc<MeshData>;
    fn instance(&self, key: accumulate::InstanceKey) -> PtrRc<MeshData>;
    fn bounds(&self, key: ()) -> Rect;
    fn plan(&self, key: ()) -> PtrRc<Plan>;
    fn postprocess_data(&self, key: ()) -> PtrRc<postprocess::Data>;
}

fn root(db: &dyn Renderer, (): ()) -> Root {
    get_state(db.cursor(()))
}

fn plan(db: &dyn Renderer, (): ()) -> PtrRc<Plan> {
    plan_render(db.window_size(()).into()).into()
}

fn postprocess_data(db: &dyn Renderer, (): ()) -> PtrRc<postprocess::Data> {
    postprocess::data(db, ()).into()
}

fn bounds(db: &dyn Renderer, (): ()) -> Rect {
    let root = db.root(());
    let passes = &db.plan(()).passes;

    let levels = u32::min(
        5,
        passes
            .iter()
            .map(|pass| pass.mesh_levels + pass.instance_levels)
            .sum(),
    );

    // This can be expensive, so cache it.
    let bounds = root.get_state().get_bounds(levels);
    if bounds.is_infinite() {
        panic!("infinite bounds")
    }
    bounds
}

fn mesh(db: &dyn Renderer, levels: u32) -> PtrRc<MeshData> {
    accumulate::mesh(db, levels)
}

fn instance(db: &dyn Renderer, key: accumulate::InstanceKey) -> PtrRc<MeshData> {
    accumulate::instance(db, key)
}
#[salsa::database(RendererStorage, InputStorage)]
#[derive(Default)]
pub struct DatabaseStruct {
    storage: salsa::Storage<Self>,
}

impl salsa::Database for DatabaseStruct {}

fn data(db: &dyn Renderer, (): ()) -> PtrRc<accumulate::DeviceData> {
    accumulate::data(db, ())
}

/// Returns a BindGroup for reading from the the output from the pass
fn accumulate_pass(
    db: &dyn Renderer,
    key: accumulate::AccumulatePassKey,
) -> PtrRc<accumulate::AccumulatePass> {
    accumulate::accumulate_pass(db, key)
}

pub fn render(db: &DatabaseStruct, frame: &wgpu::SwapChainTexture) {
    let device = db.device(());

    let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor { label: None });
    {
        let plan = db.plan(());
        let accumulate = db.accumulate_pass(accumulate::AccumulatePassKey {
            plans: plan.passes.clone(),
            filter: false,
        });
        let bind_group = accumulate.render(db, &mut encoder);
        postprocess::render(db, &mut encoder, bind_group, &frame.view);
        // TODO: debug option to draw intermediate texture to screen at actual resolution
    }

    db.queue(()).submit(Some(encoder.finish()));
}
