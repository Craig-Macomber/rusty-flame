use bytemuck::{Pod, Zeroable};
use nalgebra::{Affine2, Matrix3};

use crate::{
    flame::{Root, State},
    geometry::{self, Rect},
};

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct Vertex {
    position: Position,
    texture_coordinate: TextureCoordinate,
}

pub type TextureCoordinate = [f32; 2];

pub type Position = [f32; 2];

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub struct Instance {
    row0: [f32; 4],
    row1: [f32; 4],
}

fn convert_point(p: &na::Point2<f64>) -> [f32; 2] {
    [p.x as f32, p.y as f32]
}

const TRIANGLE_INDEXES_FOR_QUAD: [usize; 6] = [0, 1, 2, 0, 2, 3];
const UV_QUAD: [TextureCoordinate; 4] = [[0.0, 0.0], [0.0, 1.0], [1.0, 1.0], [1.0, 0.0]];

pub(crate) fn build_mesh(root: &Root, quad: Rect, levels: u32) -> Vec<Vertex> {
    let corners = quad.corners();

    let mut vertexes = vec![];
    root.get_state().process_levels(levels, &mut |state| {
        for i in &TRIANGLE_INDEXES_FOR_QUAD {
            let t2 = state.mat * corners[*i];
            vertexes.push(Vertex {
                position: convert_point(&t2),
                texture_coordinate: UV_QUAD[*i],
            })
        }
    });
    vertexes
}

pub(crate) fn build_instances(root: &Root, root_mat: Affine2<f64>, levels: u32) -> Vec<Instance> {
    let mut instances: Vec<Instance> = vec![];
    root.get_state().process_levels(levels, &mut |state| {
        let m: Matrix3<f64> = (root_mat * state.mat).to_homogeneous();
        let s = m.as_slice();
        instances.push(Instance {
            row0: [s[0] as f32, s[3] as f32, s[6] as f32, 0f32],
            row1: [s[1] as f32, s[4] as f32, s[7] as f32, 0f32],
        });
    });

    instances
}

pub(crate) fn build_quad() -> Vec<Vertex> {
    let corners: Vec<Position> = geometry::Rect {
        min: na::Point2::new(-1.0, -1.0),
        max: na::Point2::new(1.0, 1.0),
    }
    .corners()
    .iter()
    .map(convert_point)
    .collect();

    TRIANGLE_INDEXES_FOR_QUAD
        .iter()
        .map(|index| -> Vertex {
            Vertex {
                position: corners[*index],
                texture_coordinate: UV_QUAD[*index],
            }
        })
        .collect()
}
