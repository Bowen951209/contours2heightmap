use std::{env, path::PathBuf, process};

use contours2heightmap::{ColorMode, Config, FillMode, process_contours};

fn main() {
    let config = get_config();
    process_contours(config);
}

fn get_config() -> Config {
    let args: Vec<String> = env::args().collect();
    if args.len() < 2 {
        eprintln!("Please provide a file path as an argument.");
        process::exit(1);
    }

    let fill_mode = match env::var("FILL_MODE") {
        Err(_) => {
            println!("Fill mode not set. Using flat fill by default.");
            FillMode::Flat
        }

        Ok(v) => match v.parse::<i32>().ok() {
            Some(0) => FillMode::Flat,
            Some(1) => FillMode::Linear,
            _ => {
                println!("Unsupported fill mode. Using flat fill by default.");
                FillMode::Flat
            }
        },
    };

    let color_mode = match env::var("COLOR_MODE") {
        Err(_) => {
            println!("Color mode not set. Using gray by default.");
            ColorMode::Gray
        }

        Ok(v) => match v.parse::<i32>().ok() {
            Some(0) => ColorMode::Gray,
            Some(1) => ColorMode::RGB,
            _ => {
                println!("Unsupported color mode. Using gray by default.");
                ColorMode::Gray
            }
        },
    };

    let output_file_path: Option<PathBuf> = env::var("OUTPUT_PATH").ok().map(PathBuf::from);

    Config::new(args[1].clone(), fill_mode, color_mode, output_file_path)
}
