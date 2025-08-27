mod contour_line;
mod draw;
mod font;
mod heightmap;

use std::{path::PathBuf, time::Instant};

use clap::{Parser, ValueEnum, command};
use font::load_sans;
use heightmap::HeightMap;
use imageproc::{image::DynamicImage, window::display_image};

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
    file_path: PathBuf,

    /// Fill mode for heightmap generation
    #[arg(short, long, value_enum, default_value_t = FillMode::Flat)]
    fill_mode: FillMode,

    /// Color mode for output
    #[arg(short, long, value_enum, default_value_t = ColorMode::Gray)]
    color_mode: ColorMode,

    /// Output file path (optional)
    #[arg(short, long)]
    output: Option<PathBuf>,

    /// Draw contour lines and mark height text on the heightmap
    #[arg(short, long, action = clap::ArgAction::Set, default_value_t = true)]
    draw_contours: bool,
}

pub fn run() {
    let args = Args::parse();

    let start = Instant::now();
    println!("Creating contour line tree...");
    let (contour_lines, image_width, image_heihgt) =
        contour_line::get_contour_line_tree_from(&args.file_path);
    println!("Contour lines count = {}", contour_lines.size());
    println!("Contour line tree created in {:?}", start.elapsed());

    let start = Instant::now();
    println!("Filling heightmap...");
    let heightmap = match args.fill_mode {
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

    let mut heightmap_image = match args.color_mode {
        ColorMode::Gray if !args.draw_contours => DynamicImage::from(heightmap.to_gray_image()),
        _ => DynamicImage::from(heightmap.to_rgb_image()),
    };

    if args.draw_contours {
        println!("Drawing contour lines...");
        let font = load_sans();
        draw::draw_contour_lines_with_text(
            heightmap_image
                .as_mut_rgb8()
                .expect("Failed to get RGB image."),
            &heightmap.contour_line_tree,
            &font,
        );
        println!("Contour lines drawn.");
    }

    if let Some(output_path) = args.output {
        println!("Saving file...");
        heightmap_image
            .save(&output_path)
            .expect("Failed to save file.");
        println!("File saved to {:?}", &output_path);
    }

    println!("Displaying image...");
    display_image(
        "Height Map",
        &heightmap_image.into_rgb8(),
        image_width,
        image_heihgt,
    );
}
