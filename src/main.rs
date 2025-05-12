mod contour_line;
mod draw;
mod font;
mod heigtmap;

use std::{env, path::PathBuf, process};

use font::load_sans;
use heigtmap::HeightMap;
use imageproc::window::display_image;

struct Config {
    file_path: String,
    fill_mode: i32,
    output_file_path: Option<PathBuf>,
}

fn main() {
    // Read command line arguments
    let config = get_config();

    let (contour_lines, image_width, image_heihgt) =
        contour_line::get_contour_lines_from(&config.file_path);
    println!("Contour lines count = {}", contour_lines.len());

    let heightmap = match config.fill_mode {
        0 => {
            println!("Using flat fill.");
            HeightMap::new_flat(contour_lines, image_width as usize, image_heihgt as usize)
        },
        1 => {
            println!("Using linear fill");    
            HeightMap::new_linear(contour_lines, image_width as usize, image_heihgt as usize)
        },
        other => {
            eprintln!("Incorrect fill mode: {other}");
            process::exit(1);
        }
    };

    let font = load_sans();
    let heightmap_gray_image = heightmap.to_gray_image();
    if let Some(output_path) = config.output_file_path {
        heightmap_gray_image.save(&output_path).expect("Failed to save file.");
        println!("File saved to {:?}", &output_path);
    }

    let mut canvas = draw::gray_to_rgb(&heightmap_gray_image);

    draw::draw_contour_lines_with_text(&mut canvas, &heightmap.contour_lines, &font);
    display_image("Height Map", &canvas, image_width, image_heihgt);
}

fn get_config() -> Config{
    let args: Vec<String> = env::args().collect();
    if args.len() < 2 {
        eprintln!("Please provide a file path as an argument.");
        process::exit(1);
    }

    let fill_mode: i32 = env::var("FILL_MODE")
    .ok()
    .and_then(|v| v.parse::<i32>().ok())
    .unwrap_or(0);

    let output_file_path: Option<PathBuf> = env::var("OUTPUT_PATH")
    .ok()
    .map(PathBuf::from);
    Config { file_path: args[1].clone(), fill_mode, output_file_path }
}