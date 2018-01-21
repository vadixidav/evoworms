extern crate arrayfire as af;

mod gru;
mod worm;
mod geom;

use af::*;

fn main() {
    let wnd = Window::new(1280, 720, String::from("GRU Outputs"));
    let mut worm = worm::Worm::new_rand();
    worm.add_growth(5);

    while !wnd.is_closed() {
        worm.advance();
        worm.draw(&wnd);
        ::std::thread::sleep_ms(50);
    }
}
