use std::sync::Arc;
use wgpu::{Device, Queue, TextureFormat, TextureViewDescriptor};
use winit::dpi::PhysicalSize;

use crate::{
    accumulate::{self, bounds, pass},
    flame::Root,
    postprocess, ui,
    util_types::DebugIt,
};

#[salsa::input]
pub struct SalsaInputs {
    pub device: Arc<Device>,
    pub queue: Arc<Queue>,
    pub size: PhysicalSize<u32>,
    pub format: DebugIt<TextureFormat>,
    pub settings: ui::Settings,
}

#[salsa::tracked]
pub struct ComputedRoot<'db> {
    pub root: Root,
}

#[salsa::tracked]
pub fn compute_root(db: &dyn salsa::Database, inputs: SalsaInputs) -> ComputedRoot<'_> {
    ComputedRoot::new(db, inputs.settings(db).get_state())
}

#[salsa::tracked]
pub struct ComputedPostProcess<'db> {
    pub data: postprocess::Data,
}

#[salsa::tracked]
pub fn compute_postprocess(
    db: &dyn salsa::Database,
    inputs: SalsaInputs,
) -> ComputedPostProcess<'_> {
    ComputedPostProcess::new(db, postprocess::data(db, inputs))
}

// #[salsa::db]
// #[derive(Clone)]
// #[cfg_attr(not(test), derive(Default))]
// pub struct DatabaseImpl {
//     storage: salsa::Storage<Self>,
// }

// #[salsa::db]
// impl salsa::Database for DatabaseImpl {}

pub fn render(
    db: &dyn salsa::Database,
    inputs: SalsaInputs,
    frame: &wgpu::SurfaceTexture,
    encoder: &mut wgpu::CommandEncoder,
) {
    let root = compute_root(db, inputs);
    let bounds = bounds(db, root);
    let accumulate = pass(
        db,
        inputs,
        bounds,
        root,
        accumulate::PassKey {
            resolution: inputs.size(db),
            filter: false,
        },
    );
    let post = compute_postprocess(db, inputs);
    let bind_group = accumulate.render(db, inputs, root, post, encoder);
    postprocess::render(
        db,
        inputs,
        encoder,
        bind_group,
        &frame.texture.create_view(&TextureViewDescriptor::default()),
    );
    // TODO: debug option to draw intermediate texture to screen at actual resolution
}
