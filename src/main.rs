extern crate arrayfire as af;

mod gru;
mod snake;
mod geom;

use af::*;

fn main() {
    let a = randu::<f32>(Dim4::new(&[100, 1, 1, 1]));
    let mut gru = gru::GRU::new_rand(100, 20);
    let initial = gru.apply(&a);
    let out = (0..30)
        .map(|_| gru.apply(&a))
        .fold(initial, |prev, curr| join(1, &prev, &curr));

    let wnd = Window::new(1280, 720, String::from("GRU Outputs"));

    loop {
        wnd.draw_image(&out, None);

        if wnd.is_closed() == true {
            break;
        }
    }
}
