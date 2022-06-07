//! Rendering helpers used by multiple rendering stages

use bytemuck::Pod;
use wgpu::{util::DeviceExt, Buffer, Device};

#[derive(Debug)]
pub struct MeshData {
    pub count: u32,
    pub buffer: Buffer,
}

impl MeshData {
    pub fn new<'a>(device: &'a Device, data: &[impl Pod], label: &'a str) -> MeshData {
        MeshData {
            count: data.len() as u32,
            buffer: device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some(label),
                contents: bytemuck::cast_slice(data),
                usage: wgpu::BufferUsages::VERTEX,
            }),
        }
    }
}
