mod contour_processing;
mod font;

use crate::contour_processing::ContourLine;
use ab_glyph::{Font, PxScale};
use imageproc::contours::Contour;
use imageproc::drawing::Canvas;
use imageproc::image::{DynamicImage, ImageReader, Rgba};

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
    contour_processing::retain_outer(&mut contours);

    // Convert Contours to ContourLines. The heights of each contour line are then set
    let contour_lines = contour_processing::to_contour_lines(contours);
    println!("Contour lines count = {}", contour_lines.len());

    // Display the contour lines
    let font = font::load_sans();
    display_contour_lines(&contour_lines, "Contour Lines", w, h, &font);
}

fn display_img(img: &DynamicImage, title: &str, w: u32, h: u32) {
    let rgb_img = img
        .as_rgb8()
        .expect("Failed to convert image to RGB format");
    imageproc::window::display_image(title, rgb_img, w, h);
}

fn display_contour_lines<T: Font>(
    contour_lines: &Vec<ContourLine<u32>>,
    title: &str,
    w: u32,
    h: u32,
    font: &T,
) {
    let mut img = DynamicImage::new_rgb8(w, h);

    for contour_line in contour_lines {
        // Draw all contour line points
        let points = &contour_line.contour().points;
        for point in points {
            img.draw_pixel(point.x, point.y, Rgba::from([255, 0, 0, 255]));
        }

        // Mark the first point of the contour line with height value
        let point0 = points[0];
        let scale = PxScale::from(24.0);
        let text = contour_line
            .height()
            .expect("Contour line does not have height")
            .to_string();

        imageproc::drawing::draw_text_mut(
            &mut img,
            Rgba::from([0, 255, 0, 255]),
            point0.x as i32,
            point0.y as i32,
            scale,
            font,
            text.as_str(),
        );
    }

    display_img(&img, title, w, h);
}
