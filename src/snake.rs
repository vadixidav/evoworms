use gru;
use af::{self, Array};

pub struct Snake {
    brain: gru::GRU,
    /// Dimension 1 has X values in position 0 and Y values in position 1.
    body: Array,
}
