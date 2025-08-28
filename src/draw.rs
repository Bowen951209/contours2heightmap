use ab_glyph::{Font, PxScale, ScaleFont};
use imageproc::definitions::Clamp;
use imageproc::drawing::Canvas;
use imageproc::image::{Pixel, Rgb, RgbImage};
use rstar::RTree;

use crate::contour_line::ContourLine;

pub fn draw_contour_lines_with_text(
    image: &mut RgbImage,
    contour_lines: &RTree<ContourLine>,
    font: &impl Font,
) {
    for contour_line in contour_lines {
        // Draw all contour line points
        let points = &contour_line.contour.points;
        for point in points {
            image.draw_pixel(point.x as u32, point.y as u32, Rgb::from([255, 0, 0]));
        }

        // Mark the first point of the contour line with height value
        let point0 = points[0];
        let scale = PxScale::from(24.0);
        let text = contour_line.height.to_string();

        draw_text_on_center(
            image,
            Rgb::from([0, 255, 0]),
            point0.x as i32,
            point0.y as i32,
            scale,
            font,
            &text,
        );
    }
}

pub fn draw_text_on_center<C>(
    canvas: &mut C,
    color: C::Pixel,
    x: i32,
    y: i32,
    scale: impl Into<PxScale> + Copy,
    font: &impl Font,
    text: &str,
) where
    C: Canvas,
    <C::Pixel as Pixel>::Subpixel: Into<f32> + Clamp<f32>,
{
    let (text_width, text_height) = measure_text(scale, font, text);
    let x = x - text_width as i32 / 2;
    let y = y - text_height as i32 / 2;

    imageproc::drawing::draw_text_mut(canvas, color, x, y, scale, font, text);
}

pub fn measure_text(scale: impl Into<PxScale>, font: &impl Font, text: &str) -> (f32, f32) {
    let scaled_font = font.as_scaled(scale);

    let width = text
        .chars()
        .map(|ch| scaled_font.h_advance(scaled_font.glyph_id(ch)))
        .sum();

    let height = scaled_font.ascent() - scaled_font.descent();

    (width, height)
}
