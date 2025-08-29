use contours2heightmap::run;

fn main() {
    pretty_env_logger::init();
    
    run();
    log::info!("Program finished.");
}
