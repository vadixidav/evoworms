use af::{self, Array};

/// Outputs distances in an array. Every index of dimension 1 must be a new coordinate dimension.
pub fn distance_squared(a: &Array, b: &Array) -> Array {
    af::accum(&af::pow(&(a - b), &2, false), 1)
}

/// Wraps position vectors assuming all dimensions are toroidal from [0, 1).
pub fn wrap_pos(positions: &Array) -> Array {
    let rems = af::rem(positions, &1.0f32, false);
    af::select(&(&rems + 1.0), &af::lt(&rems, &0.0f32, false), &rems)
}

/// Wraps delta vectors assuming all dimensions are toroidal from [0, 1).
pub fn wrap_delta(deltas: &Array) -> Array {
    let pos = af::rem(&(deltas + 0.5), &1.0f32, false) - 0.5;
    let neg = af::rem(&(deltas - 0.5), &1.0f32, false) + 0.5;
    af::select(&neg, &af::lt(deltas, &0.0f32, false), &pos)
}

#[cfg(test)]
mod test {
    use super::*;
    use af::Dim4;

    #[test]
    fn test_wrap_pos() {
        let a = wrap_pos(&(af::randu::<f32>(Dim4::new(&[100, 2, 1, 1])) * 5.0 - 2.5));
        let mut host = [false; 2];
        af::all_true(
            &af::and(&af::ge(&a, &0, false), &af::lt(&a, &1, false), false),
            0,
        ).host(&mut host);
        assert!(host[0] && host[1]);
    }
}
