mod contour_line;
mod draw;
mod font;
mod heightmap;

use std::{path::PathBuf, time::Instant};

use clap::{Parser, ValueEnum, command};
use colorous::Gradient;
use font::load_sans;
use heightmap::HeightMap;
use imageproc::image::DynamicImage;

// Rust currently does not support reflection to constants, so we need to manually keep these.
macro_rules! define_colormode {
    ( $( $name:ident => $grad:ident ),* $(,)? ) => {
        #[derive(Debug, Clone, ValueEnum)]
        pub enum ColorMode {
            $( $name ),*
        }

        impl From<ColorMode> for Gradient {
            fn from(mode: ColorMode) -> Gradient {
                match mode {
                    $( ColorMode::$name => colorous::$grad ),*
                }
            }
        }
    };
}

define_colormode! {
    Blues => BLUES, BlueGreen => BLUE_GREEN, BluePurple => BLUE_PURPLE, BrownGreen => BROWN_GREEN,
    Cividis => CIVIDIS, Cool => COOL, Cubehelix => CUBEHELIX, Greens => GREENS,
    GreenBlue => GREEN_BLUE, Greys => GREYS, Inferno => INFERNO, Magma => MAGMA,
    Oranges => ORANGES, OrangeRed => ORANGE_RED, PinkGreen => PINK_GREEN, Plasma => PLASMA,
    Purples => PURPLES, PurpleBlue => PURPLE_BLUE, PurpleBlueGreen => PURPLE_BLUE_GREEN,
    PurpleGreen => PURPLE_GREEN, PurpleOrange => PURPLE_ORANGE, PurpleRed => PURPLE_RED,
    Rainbow => RAINBOW, Reds => REDS, RedBlue => RED_BLUE, RedGrey => RED_GREY,
    RedPurple => RED_PURPLE, RedYellowBlue => RED_YELLOW_BLUE, RedYellowGreen => RED_YELLOW_GREEN,
    Sinebow => SINEBOW, Spectral => SPECTRAL, Turbo => TURBO, Viridis => VIRIDIS,
    Warm => WARM, YellowGreen => YELLOW_GREEN, YellowGreenBlue => YELLOW_GREEN_BLUE,
    YellowOrangeBrown => YELLOW_ORANGE_BROWN, YellowOrangeRed => YELLOW_ORANGE_RED,
}

#[derive(Debug, Clone, ValueEnum)]
pub enum FillMode {
    Flat,
    Linear,
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
    #[arg(short, long, value_enum, default_value_t = ColorMode::Greys)]
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
        ColorMode::Greys => DynamicImage::from(heightmap.to_gray16()), // we want gray16 image, not rgb
        _ => DynamicImage::from(heightmap.to_rgb_8(args.color_mode.into())),
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
