#[cfg(target_arch = "wasm32")]
use wasm_bindgen::prelude::*;

mod common;

#[cfg(target_arch = "wasm32")]
#[wasm_bindgen]
pub fn wasm_run() {
    std::panic::set_hook(Box::new(console_error_panic_hook::hook));
    console_log::init().expect("could not initialize logger");
    common::main().unwrap();
}
