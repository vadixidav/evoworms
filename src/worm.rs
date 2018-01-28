use gru::GRU;
use af::{self, Array, Convertable, Dim4, Window};
use std::f32::consts::PI;
use geom;
use cgmath;
use itertools::Itertools;

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
            body: af::randu::<f32>(Dim4::new(&[1, 2, 1, 1])),
            angle: af::randu::<f32>(Dim4::new(&[1, 1, 1, 1])) * 2 * PI,
        }
    }

    pub fn add_growth<C: Convertable>(&mut self, growth: C) {
        self.grow = &self.grow + &growth.convert();
    }

    pub fn advance(&mut self) {
        let did_grow = !get_bool(&af::iszero(&self.grow));
        self.brain.mutate(0.01);
        let choices = self.brain
            .apply(&Array::new(&[0f32; 128], Dim4::new(&[128, 1, 1, 1])));
        self.angle = af::row(&choices, 0) * 0.01f32 + &self.angle;
        let delta = &geom::angle_norm(&self.angle) * 0.003f32;
        let next = &af::row(&self.body, 0) + &delta;
        self.body = if did_grow {
            self.grow = &self.grow - 1;
            af::join(0, &next, &self.body)
        } else {
            af::set_row(&af::shift(&self.body, &[1, 0, 0, 0]), &next, 0)
        };
        self.body = geom::wrap_pos(&self.body);
    }

    pub fn get_points(&self) -> Vec<cgmath::Point2<f32>> {
        use std::iter::repeat;
        let mut host = Vec::new();
        host.extend(repeat(0.0).take(self.body.dims().elements() as usize));
        self.body.host(&mut host);
        host[0..host.len() / 2]
            .iter()
            .cloned()
            .zip(host[host.len() / 2..].iter().cloned())
            .map(From::from)
            .collect()
    }
}
