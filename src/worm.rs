use gru::GRU;
use af::{self, Array, Dim4, Window, Convertable};
use std::f32::consts::PI;
use geom;

fn get_bool(a: &Array) -> bool {
    let mut host = [false; 1];
    a.host(&mut host);
    host[0]
}

pub struct Worm {
    brain: GRU,
    /// How many times the tail needs to grow before it is caught up.
    grow: Array,
    /// Dimension 1 has X values in position 0 and Y values in position 1.
    body: Array,
    /// An array of just the angle the worm is turned to.
    angle: Array,
}

impl Worm {
    pub fn new_rand() -> Worm {
        Worm {
            brain: GRU::new_rand(128, 128),
            grow: Array::new(&[0; 1], Dim4::new(&[1, 1, 1, 1])),
            body: af::randu::<f32>(Dim4::new(&[3, 2, 1, 1])),
            angle: af::randu::<f32>(Dim4::new(&[1, 1, 1, 1])) * 2 * PI,
        }
    }

    pub fn add_growth<C: Convertable>(&mut self, growth: C) {
        self.grow = &self.grow + &growth.convert();
    }

    pub fn advance(&mut self) {
        let did_grow = !get_bool(&af::iszero(&self.grow));
        let delta = &geom::angle_norm(&self.angle) * 0.1;
        let next = &af::row(&self.body, 0) + &delta;
        self.body = if did_grow {
            println!("next dims: {:?}, body dims: {:?}", next.dims(), self.body.dims());
            self.body.eval();
            next.eval();
            self.grow = &self.grow - 1;
            af::join(0, &next, &self.body)
        } else {
            af::set_row(&af::shift(&self.body, &[1, 0, 0, 0]), &next, 0)
        };
    }

    pub fn draw(&self, window: &Window) {
        println!("dims: {:?}", self.body.dims());
        window.draw_plot2(&af::col(&self.body, 0), &af::col(&self.body, 1), None);
    }
}
