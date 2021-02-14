use crate::fixed_point;
use crate::geometry::{Bounds, Rect};
use nalgebra::{Affine2, Similarity2};
use reduce::Reduce;
use std::fmt::Debug;

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

pub trait BoundedState<'a>: State<'a> {
    type B: Bounds + Debug;

    fn get_bounds(&self) -> Self::B {
        let mut b = Self::B::origin();
        // Starting with too few levels can diverge to infinity for large scale factors
        for level in 2..6 {
            let b_new = fixed_point::iterate(b, |input_bounds: &Self::B| {
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
pub struct Root<T> {
    pub storage: Vec<T>,
}

impl Root<Affine2<f64>> {
    pub fn get_state(&self) -> AffineState {
        AffineState::new(na::convert(Similarity2::from_scaling(1.0)), &self.storage)
    }
}

#[cfg(test)]
mod tests {
    use crate::flame::{fixed_point, AffineState, BoundedState, Bounds, Rect, Root, State};
    use na::{Affine2, Point2, Rotation2, Similarity2, Translation2};

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
        let v = Root::<Affine2<f64>> {
            storage: vec![na::convert(
                Similarity2::from_scaling(0.5) * Translation2::new(5.0, 6.0),
            )],
        };

        assert_eq!(
            fixed_point::iterate(Point2::new(0.0, 0.0), |p| v.storage[0].transform_point(p)),
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
            fixed_point::iterate(Point2::new(5.0, 5.0), |p| v[0].transform_point(p)),
            Point2::new(0.0, 0.0)
        );
        assert_eq!(
            fixed_point::iterate(Point2::new(5.0, 5.0), |p| v[1].transform_point(p)),
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
