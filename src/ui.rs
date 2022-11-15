use std::convert::TryFrom;

use crate::flame::Root;
use egui::{InnerResponse, Response, Ui};
use na::{Affine2, Point2, Rotation2, SMatrix, Similarity2, Translation2, Vector2};

#[derive(Clone, Debug, PartialEq)]
pub struct Settings {
    pub busy_loop: bool,
    pub auto_passes: bool,
    pub passes: u32, // TODO: make this work.
    pub n: usize,
    polygon: bool,
    scale: f64,
    rotation: f32,
    points: Vec<Point>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct Point {
    data: Affine2<f64>,
}

pub fn update(ctx: &egui::Context, setting: &mut Settings, frame_time: f64) {
    egui::SidePanel::right("Settings").show(ctx, |ui| {
        ui.checkbox(&mut setting.busy_loop, "Busy Loop");
        if setting.busy_loop {
            ui.label(format!("FPS: {:.0}", 1.0 / frame_time));
            ui.label(format!("Frame Time: {:.3}ms", frame_time * 1000.0));
        }
        ui.label("Points:");
        ui.add(egui::Slider::new(&mut setting.n, 2..=12));
        ui.checkbox(&mut setting.polygon, "Polygon");
        if setting.polygon {
            ui.label("Rotation:");
            ui.drag_angle(&mut setting.rotation);
            ui.label("Scale:");
            ui.add(
                egui::DragValue::new(&mut setting.scale)
                    .clamp_range(-0.8..=0.8)
                    .speed(0.0005),
            );
        } else {
            while setting.points.len() < setting.n {
                setting.points.push(Point {
                    data: get_polygon_point(setting, setting.points.len()),
                })
            }
            for p in &mut setting.points[0..setting.n] {
                affine_editor(ui, p);
            }
        }
    });
}

fn affine_editor(ui: &mut Ui, p: &mut Point) -> egui::InnerResponse<()> {
    let mut translation = p.data.transform_point(&Point2::new(0.0, 0.0)) - Point2::new(0.0, 0.0);
    let mut x = p.data.transform_vector(&Vector2::new(1.0, 0.0));
    let mut y = p.data.transform_vector(&Vector2::new(0.0, 1.0));

    let response = ui.group(|ui: &mut Ui| {
        vec_editor(ui, &mut translation);
        vec_editor(ui, &mut x);
        vec_editor(ui, &mut y);
    });
    // TODO: better way to construct this.
    let m: SMatrix<f64, 3, 3> = SMatrix::from_columns(&[
        x.to_homogeneous(),
        y.to_homogeneous(),
        (Point2::new(0.0, 0.0) + translation).to_homogeneous(),
    ]);
    p.data = Affine2::from_matrix_unchecked(m);
    response
}

fn vec_editor(ui: &mut Ui, p: &mut Vector2<f64>) -> egui::InnerResponse<()> {
    ui.horizontal(|ui: &mut Ui| {
        ui.label("X:");
        ui.add(
            egui::DragValue::new(&mut p.x)
                .clamp_range(-2.0..=2.0)
                .speed(0.001),
        );
        ui.label("Y:");
        ui.add(
            egui::DragValue::new(&mut p.y)
                .clamp_range(-2.0..=2.0)
                .speed(0.001),
        );
    })
}

fn get_polygon_point(setting: &Settings, i: usize) -> Affine2<f64> {
    let sm = Similarity2::from_scaling(setting.scale);

    let offset = Rotation2::new(std::f64::consts::PI * 2.0 * i as f64 / setting.n as f64)
        * Point2::new(1.0, 0.0);
    na::convert::<_, Affine2<f64>>(sm * Translation2::new(offset.x, offset.y))
        * Rotation2::new(setting.rotation as f64)
}

impl Settings {
    pub fn default() -> Self {
        Self {
            n: 5,
            scale: 0.5,
            rotation: 0.1,
            busy_loop: false,
            polygon: true,
            auto_passes: true,
            passes: 10,
            points: vec![],
        }
    }
    pub fn get_state(&self) -> Root {
        let va = (0..self.n)
            .map(|i| {
                if self.polygon {
                    get_polygon_point(self, i)
                } else {
                    self.points.get(i).unwrap().data
                }
            })
            .collect::<Vec<Affine2<f64>>>();

        Root::new(va)
    }
}
