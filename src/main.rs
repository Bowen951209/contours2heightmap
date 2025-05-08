mod contour_line;
mod font;
mod display;
mod heigtmap;

use crate::display::display_contour_lines;
use imageproc::contours::Contour;
use imageproc::drawing::Canvas;
use imageproc::image::ImageReader;

fn main() {
    // Read command line arguments
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 2 {
        println!("Please provide a file path as an argument.");
        return;
    }
    let filepath = &args[1];

    // Load the image
    let dyn_img = ImageReader::open(filepath)
        .expect("Failed to open image file")
        .decode()
        .expect("Failed to decode image");

    let (w, h) = dyn_img.dimensions();

    // To grayscale, and then we can find contours
    let grayscale = dyn_img.to_luma8();
    let mut contours: Vec<Contour<u32>> = imageproc::contours::find_contours(&grayscale);

    // find_contours finds outer and inner contours. We only retain outers as representation
    contour_line::retain_outer(&mut contours);

    // Convert Contours to ContourLines. The heights of each contour line are then set
    let contour_lines = contour_line::to_contour_lines(contours);
    println!("Contour lines count = {}", contour_lines.len());

    // Display the contour lines
    let font = font::load_sans();
    display_contour_lines(&contour_lines, "Contour Lines", w, h, &font);
}
