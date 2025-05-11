mod contour_line;
mod draw;
mod font;
mod heigtmap;

use font::load_sans;
use heigtmap::HeightMap;
use imageproc::window::display_image;

fn main() {
    // Read command line arguments
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 2 {
        eprintln!("Please provide a file path as an argument.");
        std::process::exit(1);
    }
    let filepath = &args[1];

    let (contour_lines, image_width, image_heihgt) =
        contour_line::get_contour_lines_from(&filepath);
    println!("Contour lines count = {}", contour_lines.len());

    let flat_heightmap =
        HeightMap::new_linear(contour_lines, image_width as usize, image_heihgt as usize);

    let font = load_sans();
    let heightmap_gray_image = flat_heightmap.to_gray_image();
    let contour_lines_image = contour_line::get_contour_lines_image(
        &flat_heightmap.contour_lines,
        &font,
        image_width,
        image_heihgt,
    );

    heightmap_gray_image.save("heightmap.png").unwrap();
    println!("Saved heightmap to heightmap.png");

    display_image(
        "HeightMap",
        &heightmap_gray_image,
        image_width,
        image_heihgt,
    );
    display_image(
        "Contour Lines",
        &contour_lines_image,
        image_width,
        image_heihgt,
    );
}
