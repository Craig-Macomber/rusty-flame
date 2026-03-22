use std::sync::Arc;
use wgpu::{Device, Queue, TextureFormat, TextureViewDescriptor};
use winit::dpi::PhysicalSize;

use crate::common::{
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
pub struct CachedRoot<'db> {
    pub root: Root,
}

#[salsa::tracked(lru = 1)]
fn compute_root(db: &dyn salsa::Database, inputs: SalsaInputs) -> CachedRoot<'_> {
    CachedRoot::new(db, inputs.settings(db).get_state())
}

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
    let bind_group = accumulate.render(db, inputs, root, encoder);
    postprocess::render(
        db,
        inputs,
        encoder,
        bind_group,
        &frame.texture.create_view(&TextureViewDescriptor::default()),
    );
    // TODO: debug option to draw intermediate texture to screen at actual resolution
}
