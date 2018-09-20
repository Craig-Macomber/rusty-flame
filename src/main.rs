extern crate alga;
extern crate glutin_window;
extern crate graphics;
extern crate nalgebra as na;
extern crate opengl_graphics;
extern crate piston;

use glutin_window::GlutinWindow as AppWindow;
use graphics::{Context, Graphics};
use na::{Affine2, Point2, Rotation2, Similarity2, Translation2};
use opengl_graphics::{GlGraphics, OpenGL};
use piston::event_loop::*;
use piston::input::*;
use piston::window::{Window, WindowSettings};

fn main() {
    let opengl = OpenGL::V3_2;
    let mut window: AppWindow = WindowSettings::new("Rusty Flame", [800, 800])
        .exit_on_esc(true)
        .opengl(opengl)
        .build()
        .unwrap();

    let ref mut gl = GlGraphics::new(opengl);
    let mut cursor = [0.0, 0.0];

    let mut events = Events::new(EventSettings::new().lazy(true));
    while let Some(e) = events.next(&mut window) {
        e.mouse_cursor(|x, y| {
            cursor = [x, y];
        });
        if let Some(args) = e.render_args() {
            gl.draw(args.viewport(), |c, g| {
                graphics::clear([1.0; 4], g);
                draw_rectangles(cursor, &window, &c, g);
            });
        }
    }
}

struct State<'a> {
    mat: Affine2<f64>,
    mats: &'a Vec<Affine2<f64>>,
}

fn draw_rectangles<G: Graphics>(cursor: [f64; 2], window: &Window, c: &Context, g: &mut G) {
    let draw_size = window.draw_size();

    // Cursor.
    let cursor_color = [0.0, 0.0, 0.0, 1.0];
    graphics::ellipse(
        cursor_color,
        graphics::ellipse::circle(cursor[0], cursor[1], 4.0),
        c.transform,
        g,
    );

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

    let state = State {
        mat: mat_root,
        mats: &transforms,
    };

    fn draw_level<G: Graphics>(level: i32, state: State, c: &Context, g: &mut G) {
        if level == 9 {
            let s = Point2::new(0.0, 0.0);
            let e = Point2::new(1.0, 0.0);
            let s2 = state.mat * s;
            let e2 = state.mat * e;
            graphics::line::Line::new([0.0, 1.0, 0.0, 1.0], 0.25).draw(
                [s2[0], s2[1], e2[0], e2[1]],
                &c.draw_state,
                c.transform,
                g,
            );
        } else {
            for t in state.mats.iter().map(|m| state.mat * m) {
                let s = State {
                    mat: t,
                    mats: state.mats,
                };
                draw_level(level + 1, s, c, g);
            }
        }
    }

    draw_level(0, state, c, g);
}
