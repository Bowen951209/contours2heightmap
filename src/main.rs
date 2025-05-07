use imageproc::contours::Contour;
use imageproc::drawing::Canvas;
use imageproc::image::{DynamicImage, ImageReader, Rgba};

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 2 {
        println!("Please provide a file path as an argument.");
        return;
    }
    let filepath = &args[1];
    let dyn_img = ImageReader::open(filepath)
        .expect("Failed to open image file")
        .decode()
        .expect("Failed to decode image");

    let (w, h) = dyn_img.dimensions();
    
    let grayscale = dyn_img.to_luma8();
    let contours: Vec<Contour<u32>> = imageproc::contours::find_contours(&grayscale);
    println!("Contours count = {}", contours.len());
    
    display_contours(&contours.as_slice(), "Contours", w, h);
    display_img(&dyn_img, filepath, w, h);
}

fn display_img(img: &DynamicImage, title: &str, w: u32, h: u32) {
    let rgb_img = img
        .as_rgb8()
        .expect("Failed to convert image to RGB format");
    imageproc::window::display_image(title, rgb_img, w, h);
}

fn display_contours(contours: &[Contour<u32>], title: &str, w: u32, h: u32) {
    let mut img = DynamicImage::new_rgb8(w, h);
    for contour in contours {
        for point in &contour.points {
            img.draw_pixel(point.x, point.y, Rgba::from([255, 0, 0, 255]));
        }
    }
    
    display_img(&img, title, w, h);
}
