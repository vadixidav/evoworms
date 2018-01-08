use gru::GRU;
use af::{self, Array, Dim4};
use std::f32::consts::PI;

pub struct Worm {
    brain: GRU,
    /// Dimension 1 has X values in position 0 and Y values in position 1.
    body: Array,
    /// An array of just the angle the worm is turned to.
    angle: Array,
}

impl Worm {
    pub fn new_rand() -> Worm {
        Worm {
            brain: GRU::new_rand(128, 128),
            body: af::randu::<f32>(Dim4::new(&[1, 2, 1, 1])),
            angle: af::randu::<f32>(Dim4::new(&[1, 1, 1, 1])) * 2 * PI,
        }
    }
}
