use imageproc::image::ImageReader;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 2 {
        println!("Please provide a file path as an argument.");
        return;
    }
    let filepath = &args[1];
    let img = ImageReader::open(filepath)
        .expect("Failed to open image file")
        .decode()
        .expect("Failed to decode image");
    let img = img.into_rgb32f();
    imageproc::window::display_image(filepath, &img, img.width(), img.height());
}
