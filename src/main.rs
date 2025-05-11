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

    let heightmap = HeightMap::new_flat(contour_lines, image_width as usize, image_heihgt as usize);

    let font = load_sans();
    let heightmap_gray_image = heightmap.to_gray_image();
    let mut canvas = draw::gray_to_rgb(&heightmap_gray_image);

    draw::draw_contour_lines_with_text(&mut canvas, &heightmap.contour_lines, &font);
    display_image("Height Map", &canvas, image_width, image_heihgt);
}
