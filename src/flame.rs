use nalgebra::{Affine2, Point2, Similarity2};
use reduce::Reduce;
use std::fmt::Debug;
use std::mem;

pub trait State<'a> {
    fn visit_level<F: FnMut(&Self)>(&self, callback: &mut F);

    fn process_levels<F: FnMut(&Self)>(&self, level: u32, callback: &mut F) {
        if level == 0 {
            callback(self);
        } else {
            self.visit_level(&mut |s| {
                s.process_levels(level - 1, callback);
            });
        }
    }
}

pub trait Bounds: PartialEq + Sized {
    fn union(a: &Self, b: &Self) -> Self;
    fn origin() -> Self;

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

impl Rect {
    pub fn corners(&self) -> [Point2<f64>; 4] {
        [
            self.min,
            self.max,
            Point2::new(self.min.x, self.max.y),
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
}

pub fn fixed_point_iterate<V: PartialEq, F: Fn(&V) -> V>(initial: V, f: F) -> V {
    let mut v: V = initial;
    loop {
        let v_new = f(&v);
        let v_old = mem::replace(&mut v, v_new);
        if v == v_old {
            return v;
        }
    }
}

pub trait BoundedState<'a>: State<'a> {
    type B: Bounds + Debug;

    fn get_bounds(&self) -> Self::B {
        let mut b = Self::B::origin();
        for level in 1..4 {
            let b_new = fixed_point_iterate(b, |input_bounds: &Self::B| {
                let mut b2: Option<Self::B> = None;
                self.process_levels(level, &mut |s| {
                    let b3 = s.transform_bounds(input_bounds);
                    b2 = Some(match &b2 {
                        None => b3,
                        Some(b4) => Self::B::union(&b4, &b3),
                    })
                });

                b2.unwrap()
            });
            b = b_new;
        }
        b
    }

    fn transform_bounds(&self, b: &Self::B) -> Self::B;
}

#[derive(Copy, Clone, Debug)]
pub struct AffineState<'a> {
    pub mat: Affine2<f64>,
    mats: &'a [Affine2<f64>],
}

impl<'a> AffineState<'a> {
    pub fn new(mat_root: Affine2<f64>, transforms: &'a [Affine2<f64>]) -> AffineState<'a> {
        AffineState {
            mat: mat_root,
            mats: transforms,
        }
    }
}

impl<'a> BoundedState<'a> for AffineState<'a> {
    type B = Rect;
    fn transform_bounds(&self, b: &Self::B) -> Self::B {
        let corners = b.corners();
        let points = corners
            .iter()
            .map(|p| Rect::point(self.mat.transform_point(p)));
        points.reduce(|a, b| Rect::union(&a, &b)).unwrap()
    }
}

impl<'a> State<'a> for AffineState<'a> {
    fn visit_level<F: FnMut(&Self)>(&self, callback: &mut F) {
        for t in self.mats.iter().map(|m| m * self.mat) {
            let s = Self {
                mat: t,
                mats: self.mats,
            };
            callback(&s);
        }
    }
}

#[derive(Debug)]
pub struct Root {
    pub storage: Vec<Affine2<f64>>,
}

impl Root {
    pub fn get_state(&self) -> AffineState {
        AffineState::new(na::convert(Similarity2::from_scaling(1.0)), &self.storage)
    }
}

#[cfg(test)]
mod tests {
    use crate::flame::{fixed_point_iterate, AffineState, BoundedState, Bounds, Rect, Root, State};
    use na::{Affine2, Point2, Rotation2, Similarity2, Translation2};

    #[test]
    fn fixed_point() {
        assert_eq!(fixed_point_iterate(10, |i| i / 2), 0);
    }

    fn checked_bounds(s: &AffineState) -> Rect {
        let b = s.get_bounds();
        let corners = b.corners();
        let mut out = vec![];
        s.process_levels(5, &mut |s| {
            out.extend(corners.iter().map(|p| s.mat.transform_point(p)))
        });

        assert!(b.corners().iter().all(|p| b.contains_point(*p)));
        b
    }
    #[test]
    fn empty_bounds() {
        let v = [na::convert(Similarity2::from_scaling(0.5))];
        let state = AffineState::new(na::convert(Similarity2::from_scaling(1.0)), &v);

        assert_eq!(checked_bounds(&state), Rect::origin());
    }

    #[test]
    fn shifted_bounds() {
        let v = Root {
            storage: vec![na::convert(
                Similarity2::from_scaling(0.5) * Translation2::new(5.0, 6.0),
            )],
        };

        assert_eq!(
            fixed_point_iterate(Point2::new(0.0, 0.0), |p| v.storage[0].transform_point(p)),
            Point2::new(5.0, 6.0)
        );

        assert_eq!(
            checked_bounds(&v.get_state()),
            Rect::point(Point2::new(5.0, 6.0))
        );
    }

    #[test]
    fn line_bounds() {
        let v = [
            na::convert(Similarity2::from_scaling(0.5)),
            na::convert(Similarity2::from_scaling(0.5) * Translation2::new(0.0, 1.0)),
        ];
        let state = AffineState::new(na::convert(Similarity2::from_scaling(1.0)), &v);

        assert_eq!(
            fixed_point_iterate(Point2::new(5.0, 5.0), |p| v[0].transform_point(p)),
            Point2::new(0.0, 0.0)
        );
        assert_eq!(
            fixed_point_iterate(Point2::new(5.0, 5.0), |p| v[1].transform_point(p)),
            Point2::new(0.0, 1.0)
        );

        assert_eq!(
            v[1].transform_point(&Point2::new(0.0, 0.0)),
            Point2::new(0.0, 0.5)
        );

        assert_eq!(
            checked_bounds(&state),
            Rect {
                min: Point2::new(0.0, 0.0),
                max: Point2::new(0.0, 1.0)
            }
        );
    }

    #[test]
    fn poly_bounds() {
        for n in 3..10 {
            let shift = 0.5;
            let scale = 0.5;
            let sm = Similarity2::from_scaling(scale);

            let va = (0..n)
                .map(|i| {
                    let offset =
                        Rotation2::new(std::f64::consts::PI * 2.0 * f64::from(i) / f64::from(n))
                            * Point2::new(shift, 0.0);
                    na::convert::<_, Affine2<f64>>(sm * Translation2::new(offset.x, offset.y))
                        * Rotation2::new(0.3)
                })
                .collect::<Vec<Affine2<f64>>>();

            let root = Root { storage: va };

            let bounds = checked_bounds(&root.get_state());
            assert!(bounds.contains(&Rect {
                min: Point2::new(-0.3, -0.3),
                max: Point2::new(0.3, 0.3)
            }));
            assert!(Rect {
                min: Point2::new(-0.7, -0.7),
                max: Point2::new(0.7, 0.7)
            }
            .contains(&bounds));
        }
    }
}
