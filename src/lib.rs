mod contour_line;
mod draw;
mod font;
mod heightmap;

use std::{path::PathBuf, time::Instant};

use font::load_sans;
use heightmap::HeightMap;
use imageproc::{image::DynamicImage, window::display_image};

#[derive(Debug, Clone)]
pub enum FillMode {
    Flat,
    Linear,
}

#[derive(Debug, Clone)]
pub enum ColorMode {
    Gray,
    RGB,
}

#[derive(Debug, Clone)]
pub struct Config {
    pub file_path: String,
    pub fill_mode: FillMode,
    pub color_mode: ColorMode,
    pub output_file_path: Option<PathBuf>,
}

impl Config {
    pub fn new(
        file_path: String,
        fill_mode: FillMode,
        color_mode: ColorMode,
        output_file_path: Option<PathBuf>,
    ) -> Self {
        Self {
            file_path,
            fill_mode,
            color_mode,
            output_file_path,
        }
    }
}

pub fn process_contours(config: Config) {
    let start = Instant::now();
    println!("Creating contour line tree...");
    let (contour_lines, image_width, image_heihgt) =
        contour_line::get_contour_line_tree_from(&config.file_path);
    println!("Contour lines count = {}", contour_lines.size());
    println!("Contour line tree created in {:?}", start.elapsed());

    let start = Instant::now();
    println!("Filling heightmap...");
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
    println!("Heightmap filled in {:?}", start.elapsed());

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
