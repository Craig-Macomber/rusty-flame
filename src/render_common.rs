//! Rendering helpers used by multiple rendering stages

use bytemuck::Pod;
use wgpu::{util::DeviceExt, Buffer, Device};

#[derive(Debug)]
pub struct MeshData {
    pub count: u32,
    pub buffer: Buffer,
}

impl MeshData {
    pub fn new<'a, T: Pod>(device: &'a Device, data: &[T], label: &'a str) -> MeshData {
        MeshData {
            count: data.len() as u32,
            buffer: device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some(label),
                contents: bytemuck::cast_slice(&data),
                usage: wgpu::BufferUsage::VERTEX,
            }),
        }
    }
}
