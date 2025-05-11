mod contour_line;
mod draw;
mod font;
mod heigtmap;

use font::load_sans;
use heigtmap::HeightMap;
use imageproc::contours::{self, Contour};
use imageproc::image::ImageReader;
use imageproc::window::display_image;

fn main() {
    // Read command line arguments
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 2 {
        eprintln!("Please provide a file path as an argument.");
        std::process::exit(1);
    }
    let filepath = &args[1];

    // Load the image
    let dyn_img = ImageReader::open(filepath)
        .expect("Failed to open image file")
        .decode()
        .expect("Failed to decode image");

    let (w, h) = (dyn_img.width(), dyn_img.height());

    // To grayscale, and then we can find contours
    let grayscale = dyn_img.to_luma8();
    let mut contours: Vec<Contour<u32>> = contours::find_contours(&grayscale);

    // find_contours finds outer and inner contours. We only retain outers as representation
    contour_line::retain_outer(&mut contours);

    // Convert Contours to ContourLines. The heights of each contour line are then set
    let contour_lines = contour_line::to_contour_lines(contours);
    println!("Contour lines count = {}", contour_lines.len());

    let flat_heightmap = HeightMap::new_flat(contour_lines, w as usize, h as usize);

    let font = load_sans();
    let heightmap_gray_image = flat_heightmap.to_gray_image();
    let contour_lines_image =
        contour_line::get_contour_lines_image(&flat_heightmap.contour_lines, &font, w, h);
    display_image("HeightMap", &heightmap_gray_image, w, h);
    display_image("Contour Lines", &contour_lines_image, w, h);
}
