use contours2heightmap::run;
use log::info;

fn main() {
    pretty_env_logger::init();
    
    run();
    info!("Program finished.");
}
