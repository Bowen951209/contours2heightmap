mod contour_line;
mod draw;
mod font;
mod heightmap;

use std::{env, path::PathBuf, process};

use font::load_sans;
use heightmap::HeightMap;
use imageproc::{image::DynamicImage, window::display_image};

enum FillMode {
    Flat,
    Linear,
}

enum ColorMode {
    Gray,
    RGB,
}

struct Config {
    file_path: String,
    fill_mode: FillMode,
    color_mode: ColorMode,
    output_file_path: Option<PathBuf>,
}

fn main() {
    // Read command line arguments
    let config = get_config();

    let (contour_lines, image_width, image_heihgt) =
        contour_line::get_contour_line_tree_from(&config.file_path);
    println!("Contour lines count = {}", contour_lines.size());

    let heightmap = match config.fill_mode {
        FillMode::Flat => {
            println!("Using flat fill.");
            HeightMap::new_flat(contour_lines, image_width as usize, image_heihgt as usize)
        }
        FillMode::Linear => {
            println!("Using linear fill");
            HeightMap::new_linear(contour_lines, image_width as usize, image_heihgt as usize)
        }
    };

    let heightmap_image = match config.color_mode {
        ColorMode::Gray => DynamicImage::from(heightmap.to_gray_image()),
        ColorMode::RGB => DynamicImage::from(heightmap.to_rgb_image()),
    };

    if let Some(output_path) = config.output_file_path {
        heightmap_image
            .save(&output_path)
            .expect("Failed to save file.");
        println!("File saved to {:?}", &output_path);
    }

    let mut canvas = heightmap_image.into_rgb8();

    let font = load_sans();
    draw::draw_contour_lines_with_text(&mut canvas, &heightmap.contour_line_tree, &font);
    display_image("Height Map", &canvas, image_width, image_heihgt);
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

    Config {
        file_path: args[1].clone(),
        fill_mode,
        color_mode,
        output_file_path,
    }
}
