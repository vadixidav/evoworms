extern crate arrayfire as af;
extern crate cgmath;
extern crate glium;
extern crate glowygraph as gg;
extern crate itertools;

mod gru;
mod worm;
mod geom;

use glium::glutin;
use gg::render2::Node;
use itertools::Itertools;

fn main() {
    let mut events_loop = glutin::EventsLoop::new();
    let window = glutin::WindowBuilder::new();
    let context = glutin::ContextBuilder::new().with_depth_buffer(24);
    let display = glium::Display::new(window, context, &events_loop).unwrap();
    let grender = gg::render2::Renderer::new(&display);

    let mut worms = (0..)
        .map(|_| worm::Worm::new_rand())
        .take(20)
        .collect::<Vec<_>>();

    for worm in &mut worms {
        worm.add_growth(200);
        worm.advance();
    }

    // the main loop
    loop {
        use glium::Surface;

        // Get dimensions
        let dims = display.get_framebuffer_dimensions();
        let hscale = dims.1 as f32 / dims.0 as f32;

        // drawing a frame
        let mut target = display.draw();
        target.clear_color_and_depth((0.0, 0.0, 0.0, 0.0), 1.0);
        grender.render_nodes(
            &mut target,
            [[2.0, 0.0, 0.0], [0.0, 2.0, 0.0], [-1.0, -1.0, 2.0]],
            [[hscale, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            &worms
                .iter()
                .map(worm::Worm::get_points)
                .flatten()
                .into_iter()
                .map(From::from)
                .collect::<Vec<Node>>(),
        );
        target.finish().unwrap();

        for worm in &mut worms {
            worm.advance();
        }

        events_loop.poll_events(|event| match event {
            glutin::Event::WindowEvent { event, .. } => match event {
                glutin::WindowEvent::Closed => ::std::process::exit(0),
                _ => (),
            },
            _ => (),
        });
    }
}
