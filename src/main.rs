extern crate arrayfire as af;

mod gru;

use af::*;

fn main() {
    let a = randu::<f64>(Dim4::new(&[100, 1, 1, 1]));
    let mut gru = gru::GRU::new_rand(100, 20);
    let initial = gru.apply(&a);
    let out = (0..16)
        .map(|_| gru.apply(&a))
        .fold(initial, |prev, curr| af::join(1, &prev, &curr));
    print(&out);
}
