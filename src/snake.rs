use gru::GRU;
use af::{self, Array, Dim4};

pub struct Snake {
    brain: GRU,
    /// Dimension 1 has X values in position 0 and Y values in position 1.
    body: Array,
}

impl Snake {
    pub fn new_rand() -> Snake {
        Snake {
            brain: GRU::new_rand(128, 128),
            body: af::randu::<f32>(Dim4::new(&[1, 2, 1, 1])),
        }
    }
}
