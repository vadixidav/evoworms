use af::{self, Array};

/// Outputs distances in an array. Every index of dimension 1 must be a new coordinate dimension.
pub fn distance_squared(a: &Array<f32>, b: &Array<f32>) -> Array<f32> {
    af::accum(&af::pow(&(a - b), &2, false), 1)
}

/// Wraps position vectors assuming all dimensions are toroidal from [0, 1).
pub fn wrap_pos(positions: &Array<f32>) -> Array<f32> {
    let rems = af::rem(positions, &1.0f32, false);
    let choices: Array<bool> = unsafe { ::std::mem::transmute(af::lt(&rems, &0.0f32, false)) };
    af::select(&(&rems + 1.0f32), &choices, &rems)
}

/// Wraps delta vectors assuming all dimensions are toroidal from [0, 1).
pub fn wrap_delta(deltas: &Array<f32>) -> Array<f32> {
    let pos = af::rem(&(deltas + 0.5f32), &1.0f32, false) - 0.5f32;
    let neg = af::rem(&(deltas - 0.5f32), &1.0f32, false) + 0.5f32;
    let choices: Array<bool> = unsafe { ::std::mem::transmute(af::lt(deltas, &0.0f32, false)) };
    af::select(&neg, &choices, &pos)
}

/// Generates a 2d unit vector pointing in the angle specified.
pub fn angle_norm(angle: &Array<f32>) -> Array<f32> {
    af::join(1, &af::cos(angle), &af::sin(angle))
}

#[cfg(test)]
mod test {
    use super::*;
    use af::Dim4;

    #[test]
    fn test_wrap_pos() {
        let a = wrap_pos(&(af::randu::<f32>(Dim4::new(&[100, 2, 1, 1])) * 5.0f32 - 2.5f32));
        let mut host = [false; 2];
        af::all_true(
            &af::and(&af::ge(&a, &0, false), &af::lt(&a, &1, false), false),
            0,
        )
        .host(&mut host);
        assert!(host[0] && host[1]);
    }
}
