use ab_glyph::{Font, PxScale, ScaleFont};
use imageproc::definitions::Clamp;
use imageproc::drawing::Canvas;
use imageproc::image::Pixel;

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
    let (text_width, text_height) = measure_text(scale, font, &text);
    let x = x - text_width as i32 / 2;
    let y = y - text_height as i32 / 2;

    imageproc::drawing::draw_text_mut(canvas, color, x, y, scale, font, &text);
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
