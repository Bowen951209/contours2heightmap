mod contour_line;
mod draw;
mod font;
mod heightmap;

use std::{path::PathBuf, time::Instant};

use clap::{Parser, ValueEnum, command};
use font::load_sans;
use heightmap::HeightMap;
use imageproc::image::DynamicImage;

#[derive(Debug, Clone, ValueEnum)]
pub enum FillMode {
    Flat,
    Linear,
}

#[derive(Debug, Clone, ValueEnum)]
pub enum ColorMode {
    Gray,
    Rgb,
}

#[derive(Parser)]
#[command(name = "c2h")]
#[command(about = "Convert contour images to heightmaps")]
#[command(version)]
pub struct Args {
    /// Input contour image file
    input_path: PathBuf,

    /// Output image file
    output_path: PathBuf,

    /// Fill mode for heightmap generation
    #[arg(short, long, value_enum, default_value_t = FillMode::Flat)]
    fill_mode: FillMode,

    /// Color mode for output
    #[arg(short, long, value_enum, default_value_t = ColorMode::Gray)]
    color_mode: ColorMode,

    /// Draw contour lines and mark height text on the heightmap
    #[arg(long, action = clap::ArgAction::SetTrue, default_value_t = false)]
    draw_contours: bool,

    /// The height gap between contour lines
    #[arg(short, long, default_value_t = 50.0)]
    gap: f64,
}

pub fn run() {
    let args = Args::parse();

    let start = Instant::now();
    println!("Creating contour line tree...");
    let (contour_lines, image_width, image_height) =
        contour_line::get_contour_line_tree_from(&args.input_path, args.gap);
    println!("Contour lines count = {}", contour_lines.size());
    println!("Contour line tree created in {:?}", start.elapsed());

    let start = Instant::now();
    println!("Filling heightmap...");
    let heightmap = match args.fill_mode {
        FillMode::Flat => {
            println!("Using flat fill.");
            HeightMap::new_flat(contour_lines, args.gap, image_width, image_height)
        }
        FillMode::Linear => {
            println!("Using linear fill");
            HeightMap::new_linear(contour_lines, args.gap, image_width, image_height)
        }
    };
    println!("Heightmap filled in {:?}", start.elapsed());

    let mut heightmap_image = match args.color_mode {
        ColorMode::Gray => DynamicImage::from(heightmap.to_gray_image()),
        ColorMode::Rgb => DynamicImage::from(heightmap.to_rgb_image()),
    };

    if args.draw_contours {
        println!("Drawing contour lines...");

        // Ensure the image is in RGB format before drawing colored contour lines.
        // If the image is already RGB, this has no effect; if grayscale, it converts to RGB.
        heightmap_image = DynamicImage::from(heightmap_image.into_rgb8());

        let font = load_sans();
        draw::draw_contour_lines_with_text(
            heightmap_image.as_mut_rgb8().unwrap(),
            &heightmap.contour_line_tree,
            &font,
        );
        println!("Contour lines drawn.");
    }

    println!("Saving file...");
    heightmap_image
        .save(&args.output_path)
        .expect("Failed to save file.");
    println!("File saved to {:?}", &args.output_path);
}
