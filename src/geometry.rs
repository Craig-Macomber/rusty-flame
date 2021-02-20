use nalgebra::Point2;
use winit::dpi::PhysicalSize;

pub trait Bounds: PartialEq + Sized {
    fn union(a: &Self, b: &Self) -> Self;
    fn origin() -> Self;
    fn is_infinite(&self) -> bool;

    fn contains(&self, other: &Self) -> bool {
        &Self::union(self, other) == self
    }

    fn grow(&self, portion: f64) -> Self;
}

#[derive(Copy, Clone, PartialEq, Debug)]
pub struct Rect {
    pub min: Point2<f64>,
    pub max: Point2<f64>,
}

impl Eq for Rect {}

impl Rect {
    pub fn corners(&self) -> [Point2<f64>; 4] {
        [
            self.min,
            Point2::new(self.min.x, self.max.y),
            self.max,
            Point2::new(self.max.x, self.min.y),
        ]
    }

    pub fn point(p: Point2<f64>) -> Self {
        Self { min: p, max: p }
    }

    pub fn contains_point(&self, p: Point2<f64>) -> bool {
        self.contains(&Rect::point(p))
    }

    pub fn width(&self) -> f64 {
        (self.max - self.min).x
    }

    pub fn height(&self) -> f64 {
        (self.max - self.min).y
    }
}

impl Bounds for Rect {
    fn origin() -> Self {
        Rect::point(Point2::new(0.0, 0.0))
    }

    fn union(a: &Self, b: &Self) -> Self {
        Self {
            min: Point2::new(f64::min(a.min.x, b.min.x), f64::min(a.min.y, b.min.y)),
            max: Point2::new(f64::max(a.max.x, b.max.x), f64::max(a.max.y, b.max.y)),
        }
    }

    fn grow(&self, portion: f64) -> Self {
        let v = (self.max - self.min) * (portion / 2.0);
        Rect {
            min: self.min - v,
            max: self.max + v,
        }
    }

    fn is_infinite(&self) -> bool {
        self.min.x == f64::NEG_INFINITY
            || self.min.y == f64::NEG_INFINITY
            || self.max.x == f64::INFINITY
            || self.max.y == f64::INFINITY
    }
}

pub fn letter_box(container: Rect, content: Rect) -> na::Affine2<f64> {
    let scale = f64::min(
        container.width() / content.width(),
        container.height() / content.height(),
    );

    na::convert(
        na::Similarity2::from_scaling(scale)
            * na::Translation2::new(
                -content.min.x
                    + ((container.width() / scale) - content.width()) / 2.0
                    + container.min.x / scale,
                -content.min.y
                    + ((container.height() / scale) - content.height()) / 2.0
                    + container.min.y / scale,
            ),
    )
}

pub fn box_to_box(container: Rect, content: Rect) -> na::Affine2<f64> {
    let scale_x = container.width() / content.width();
    let scale_y = container.height() / content.height();

    let m = na::Matrix3::from_rows(&[
        na::RowVector3::new(scale_x, 0.0, container.min.x - content.min.x * scale_x),
        na::RowVector3::new(0.0, scale_y, container.min.y - content.min.y * scale_y),
        na::RowVector3::new(0.0, 0.0, 1.0),
    ]);

    na::Affine2::from_matrix_unchecked(m)
}
