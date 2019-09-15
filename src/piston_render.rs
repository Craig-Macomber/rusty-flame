use crate::flame::BoundedState;
use crate::{get_state, process_scene};
use na;
use na::Affine2;
use opengl_graphics::{GlGraphics, OpenGL, Texture};
use piston::event_loop::{EventLoop, EventSettings, Events};
use piston::input::{MouseCursorEvent, RenderEvent};
use piston::window::WindowSettings;
use piston_window::{image, math, PistonWindow, Size, TextureSettings, Transformed};
use vecmath;

pub fn main() {
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
    let line = graphics::line::Line::new([0.8, 0.0, 0.0, 0.3], 0.0125);

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

                let root = get_state(cursor, [window_size.width, window_size.height]);
                let state = root.get_state();
                let bounds = state.get_bounds();
                let scale = f64::min(window_size.width, window_size.height)
                    / f64::max(bounds.width(), bounds.height());

                let trans = math::multiply(
                    math::multiply(c.transform, math::scale(scale, scale)),
                    math::translate([-bounds.min.x, -bounds.min.y]),
                );

                let bounding_rect = graphics::rectangle::rectangle_by_corners(
                    bounds.min.x,
                    bounds.min.y,
                    bounds.max.x,
                    bounds.max.y,
                );

                graphics::rectangle(cursor_color, bounding_rect, trans, g);
                process_scene(state, &mut |state| {
                    let trans2 = math::multiply(trans, array_from_affine(&state.mat));
                    let s2 = bounds.min;
                    let e2 = bounds.max;
                    line.draw([s2[0], s2[1], e2[0], e2[1]], &c.draw_state, trans2, g);
                    graphics::rectangle([0.0, 1.0, 0.2, 0.1], bounding_rect, trans2, g);
                })
            });
        }
    }
}

fn array_from_affine<T: na::RealField>(a: &Affine2<T>) -> vecmath::Matrix2x3<T> {
    let m = a.matrix();
    [
        [*m.index((0, 0)), *m.index((0, 1)), *m.index((0, 2))],
        [*m.index((1, 0)), *m.index((1, 1)), *m.index((1, 2))],
    ]
}
