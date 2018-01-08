use af::{self, Array};

/// Outputs distances in a column vector. Every index of dimension 1 must be a new coordinate dimension.
pub fn distance_squared(a: &Array, b: &Array) -> Array {
    af::accum(&af::pow(&(a - b), &2, false), 1)
}
