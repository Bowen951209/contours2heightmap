mod contour_line;
mod draw;
mod heightmap;

use std::{env, fmt, path::PathBuf, time::Instant};

use crate::contour_line::ContourLine;
use ab_glyph::FontRef;
use clap::{Parser, ValueEnum, command};
use colorous::Gradient;
use heightmap::HeightMap;
use imageproc::image::DynamicImage;
use log::{debug, error, info};
use rstar::RTree;

// Rust currently does not support reflection to constants, so we need to manually keep these.
macro_rules! define_colormode {
    ( $( $name:ident => $grad:ident ),* $(,)? ) => {
        #[derive(Debug, Clone, Copy, ValueEnum)]
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

#[derive(Debug, Clone)]
pub enum CreateContourLineError {
    ContoursNotFound,
}

impl fmt::Display for CreateContourLineError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            CreateContourLineError::ContoursNotFound => {
                write!(f, "No contour lines found in the input image.")
            }
        }
    }
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
    init_logging();
    let args = Args::parse();

    let (contour_lines, image_width, image_height) = create_contour_line_tree(&args)
        .unwrap_or_else(|e| {
            error!("Error creating contour line tree: {e}");
            std::process::exit(1);
        });
    let heightmap = fill_heightmap(
        args.fill_mode,
        args.gap,
        contour_lines,
        image_width,
        image_height,
    );
    let mut image = create_heightmap_image(args.color_mode, &heightmap);

    if args.draw_contours {
        draw_contours_on_image(&mut image, &heightmap);
    }

    save_image(&image, &args.output_path);
    info!("Program finished.");
}

fn init_logging() {
    if env::var("RUST_LOG").is_err() {
        unsafe {
            env::set_var("RUST_LOG", "trace");
        }
        pretty_env_logger::init();
        info!("RUST_LOG not set, defaulting to 'trace'");
    } else {
        pretty_env_logger::init();
    }
}

fn create_contour_line_tree(
    args: &Args,
) -> Result<(RTree<ContourLine>, u32, u32), CreateContourLineError> {
    let start = Instant::now();
    info!("Creating contour line tree...");
    let (contour_lines, image_width, image_height) =
        contour_line::get_contour_line_tree_from(&args.input_path, args.gap);

    if contour_lines.size() == 0 {
        return Err(CreateContourLineError::ContoursNotFound);
    }

    debug!("Contour lines count = {}", contour_lines.size());
    info!("Contour line tree created in {:?}", start.elapsed());

    Ok((contour_lines, image_width, image_height))
}

fn fill_heightmap(
    fill_mode: FillMode,
    gap: f64,
    contour_lines: RTree<ContourLine>,
    image_width: u32,
    image_height: u32,
) -> HeightMap {
    let start = Instant::now();
    info!("Filling heightmap...");
    let heightmap = match fill_mode {
        FillMode::Flat => HeightMap::new_flat(contour_lines, gap, image_width, image_height),
        FillMode::Linear => HeightMap::new_linear(contour_lines, gap, image_width, image_height),
    };

    info!("Heightmap filled in {:?}", start.elapsed());
    heightmap
}

fn create_heightmap_image(color_mode: ColorMode, heightmap: &HeightMap) -> DynamicImage {
    match color_mode {
        ColorMode::Greys => DynamicImage::from(heightmap.to_gray16()),
        _ => DynamicImage::from(heightmap.to_rgb_8(color_mode.into())),
    }
}

fn draw_contours_on_image(heightmap_image: &mut DynamicImage, heightmap: &HeightMap) {
    info!("Drawing contour lines...");
    *heightmap_image = DynamicImage::from(heightmap_image.to_rgb8());

    let font = FontRef::try_from_slice(include_bytes!("OpenSans-Medium.ttf"))
        .expect("Failed to load font");
    debug!("OpenSans-Medium font loaded");

    draw::draw_contour_lines_with_text(
        heightmap_image.as_mut_rgb8().unwrap(),
        &heightmap.contour_line_tree,
        &font,
    );
    info!("Contour lines drawn.");
}

fn save_image(heightmap_image: &DynamicImage, output_path: &PathBuf) {
    info!("Saving file...");
    heightmap_image
        .save(output_path)
        .expect("Failed to save file.");
    info!("File saved to {:?}", output_path);
}
