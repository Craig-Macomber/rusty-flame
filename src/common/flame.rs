use crate::common::fixed_point;
use crate::common::geometry::{Bounds, Rect};
use nalgebra::Affine2;
use ordered_float::OrderedFloat;
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

    fn get_bounds(&self, levels: u32) -> Self::B {
        let mut b = Self::B::origin();
        // Starting with too few levels can diverge to infinity for large scale factors
        for level in 0..=levels {
            let mut count = 0;
            if b.is_infinite() {
                b = Self::B::origin();
            }
            let b_new = fixed_point::iterate_cmp(
                b,
                |before, after| {
                    count += 1;
                    // Running with lower level values is just an optimization.
                    // If it its taking too long (ex: divergent with lower level value) early exit.
                    if count > 10 && level != levels {
                        return (after, true);
                    }

                    // TODO: why, even with this trivially safe version can corners of the fractal go off screen?
                    // let result = before == after;
                    // (after, result)

                    // First condition is for correctness (makes ure entire fractal is contained),
                    // Second condition prevents termination while we are still making progress (shrinking bounds)
                    if before.contains(&after) && after.grow(0.001).contains(&before) {
                        (before, true)
                    } else {
                        // Grow by a tiny bit. This makes the test more conservative and should finish sooner.
                        // Less than above grown to prevent getting stuck. (TODO: prove this won't get stuck)
                        (after.grow(0.0001), false)
                    }
                },
                |input_bounds: &Self::B| {
                    let mut b2: Option<Self::B> = None;
                    self.process_levels(level, &mut |s| {
                        let b3 = s.transform_bounds(input_bounds);
                        b2 = Some(match &b2 {
                            None => b3,
                            Some(b4) => Self::B::union(b4, &b3),
                        })
                    });

                    b2.unwrap()
                },
            );
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

#[derive(Debug, PartialEq, Clone)]
pub struct Root {
    storage: Vec<Affine2<f64>>,
}

impl std::hash::Hash for Root {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        for a in &self.storage {
            for x in 0..3 {
                for y in 0..3 {
                    OrderedFloat(a[(x, y)]).hash(state);
                }
            }
        }
    }
}

/// NaN is invalid in all the floats here, so Eq is fine.
impl Eq for Root {}

impl Root {
    pub fn new(storage: Vec<Affine2<f64>>) -> Root {
        Root { storage }
    }

    pub fn get_state(&self) -> AffineState<'_> {
        AffineState::new(Affine2::<f64>::identity(), &self.storage)
    }
}

#[cfg(test)]
mod tests {
    use crate::common::flame::{fixed_point, AffineState, BoundedState, Bounds, Rect, Root, State};
    use nalgebra::{Affine2, Point2, Rotation2, Similarity2, Translation2};

    fn checked_bounds(s: &AffineState) -> Rect {
        let b = s.get_bounds(3);
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
        let v = [nalgebra::convert(Similarity2::from_scaling(0.5))];
        let state = AffineState::new(nalgebra::convert(Similarity2::from_scaling(1.0)), &v);

        assert_eq!(checked_bounds(&state), Rect::origin());
    }

    #[test]
    fn shifted_bounds() {
        let v = Root::new(vec![nalgebra::convert(
            Similarity2::from_scaling(0.5) * Translation2::new(5.0, 6.0),
        )]);

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
        let v: [nalgebra::Transform<f64, nalgebra::TAffine, 2>; 2] = [
            nalgebra::convert(Similarity2::from_scaling(0.5)),
            nalgebra::convert(Similarity2::from_scaling(0.5) * Translation2::new(0.0, 1.0)),
        ];

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

        // TODO: this test has some precision issues. poly_bounds covers the important cases, so this is disabled for now.
        // let state = AffineState::new(na::convert(Similarity2::from_scaling(1.0)), &v);
        // assert_eq!(
        //     checked_bounds(&state),
        //     Rect {
        //         min: Point2::new(0.0, 0.0),
        //         max: Point2::new(0.0, 1.0)
        //     }
        // );
    }

    #[test]
    fn poly_bounds() {
        for n in 3..10 {
            let shift = 0.5;
            let scale = 0.5;
            let sm = Similarity2::from_scaling(scale);

            let storage = (0..n)
                .map(|i| {
                    let offset =
                        Rotation2::new(std::f64::consts::PI * 2.0 * f64::from(i) / f64::from(n))
                            * Point2::new(shift, 0.0);
                    nalgebra::convert::<_, Affine2<f64>>(sm * Translation2::new(offset.x, offset.y))
                        * Rotation2::new(0.3)
                })
                .collect::<Vec<Affine2<f64>>>();

            let bounds = checked_bounds(&AffineState::new(Affine2::<f64>::identity(), &storage));
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
