use std::env;

use contours2heightmap::run;
use log::info;

fn main() {
    if env::var("RUST_LOG").is_err() {
        unsafe {
            env::set_var("RUST_LOG", "trace");
        }
        pretty_env_logger::init();
        info!("RUST_LOG not set, defaulting to 'trace'"); // must be called after init
    } else {
        pretty_env_logger::init();
    }

    run();
    info!("Program finished.");
}
