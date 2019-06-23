extern crate alga;
extern crate glutin_window;
extern crate graphics;
extern crate image as im;
extern crate nalgebra as na;
extern crate opengl_graphics;
extern crate piston;

use graphics::{Context, Graphics};
use na::{Affine2, Point2, Rotation2, Similarity2, Translation2};
use opengl_graphics::{GlGraphics, OpenGL, Texture};
use piston::event_loop::{EventLoop, EventSettings, Events};
use piston::input::{MouseCursorEvent, RenderEvent};
use piston::window::WindowSettings;
use piston_window::{image, PistonWindow, Size, TextureSettings, Transformed};

fn main() {
    let window_size = Size {
        width: 800.0,
        height: 800.0,
    };

    let opengl = OpenGL::V3_2;
    let mut window: PistonWindow = WindowSettings::new("Rusty Flame", window_size)
        .exit_on_esc(true)
        .graphics_api(opengl)
        .build()
        .unwrap();

    // window.set_bench_mode(true);

    let mut gl = GlGraphics::new(opengl);

    let mut cursor = [0.0, 0.0];
    let size = 32.0;

    let mut img = im::ImageBuffer::new(2, 2);
    for x in 0..2 {
        for y in 0..2 {
            img.put_pixel(
                x,
                y,
                im::Rgba([rand::random(), rand::random(), rand::random(), 255]),
            );
        }
    }
    let texture = Texture::from_image(&img, &TextureSettings::new());

    let cursor_color = [0.0, 0.0, 0.0, 1.0];

    let mut events = Events::new(EventSettings::new().lazy(true));
    while let Some(e) = events.next(&mut window) {
        e.mouse_cursor(|pos| {
            cursor = pos;
        });
        if let Some(args) = e.render_args() {
            gl.draw(args.viewport(), |c, g| {
                graphics::clear([1.0; 4], g);
                image(&texture, c.transform.trans(10.0, 10.0).zoom(size), g);

                // Cursor
                graphics::ellipse(
                    cursor_color,
                    graphics::ellipse::circle(cursor[0], cursor[1], 4.0),
                    c.transform,
                    g,
                );

                draw_content(cursor, window_size, &c, g);
            });
        }
    }
}

trait State<'a> {
    fn visit_level<F: FnMut(&Self)>(&self, callback: &mut F);
}

#[derive(Copy, Clone)]
struct AffineState<'a> {
    mat: Affine2<f64>,
    mats: &'a Vec<Affine2<f64>>,
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

fn draw_content<G: Graphics>(cursor: [f64; 2], draw_size: Size, c: &Context, g: &mut G) {
    let sm = Similarity2::from_scaling(0.5);

    let a: Affine2<f64> = na::convert(sm * Translation2::new(0.25, 0.25));
    let a2: Affine2<f64> = na::convert(sm * Translation2::new(-0.25, 0.25));

    let transforms = vec![
        na::convert(sm * Translation2::new(0.0, -0.25)),
        a2 * Rotation2::new(0.1 * (cursor[1] as f64) / 5.0),
        a * Rotation2::new(0.1 * (cursor[0] as f64) / 5.0),
    ];

    let mat_root = na::convert(
        Translation2::new(cursor[0], cursor[1]) * Similarity2::from_scaling(draw_size.width as f64),
    );

    let state = AffineState {
        mat: mat_root,
        mats: &transforms,
    };

    let start = Point2::new(0.0, 0.0);
    let end = Point2::new(1.0, 0.0);
    let line = graphics::line::Line::new([0.0, 1.0, 0.0, 1.0], 0.25);
    process_levels(10, &state, &mut |state| {
        let s2 = state.mat * start;
        let e2 = state.mat * end;
        line.draw([s2[0], s2[1], e2[0], e2[1]], &c.draw_state, c.transform, g);
    })
}

fn process_levels<'a, S: State<'a>, F: FnMut(&S)>(level: u32, state: &S, callback: &mut F) {
    if level == 0 {
        callback(state);
    } else {
        state.visit_level(&mut |s| {
            process_levels(level - 1, s, callback);
        });
    }
}