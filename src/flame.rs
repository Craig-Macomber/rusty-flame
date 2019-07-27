use nalgebra::Affine2;

pub trait State<'a> {
    fn visit_level<F: FnMut(&Self)>(&self, callback: &mut F);
}

#[derive(Copy, Clone)]
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

impl<'a> State<'a> for AffineState<'a> {
    fn visit_level<F: FnMut(&Self)>(&self, callback: &mut F) {
        for t in self.mats.iter().map(|m| self.mat * m) {
            let s = Self {
                mat: t,
                mats: self.mats,
            };
            callback(&s);
        }
    }
}

pub fn process_levels<'a, S: State<'a>, F: FnMut(&S)>(level: u32, state: &S, callback: &mut F) {
    if level == 0 {
        callback(state);
    } else {
        state.visit_level(&mut |s| {
            process_levels(level - 1, s, callback);
        });
    }}
