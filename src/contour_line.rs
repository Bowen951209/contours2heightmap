use std::u32;

use ab_glyph::{Font, PxScale};
use imageproc::{
    contours::{BorderType, Contour}, drawing::Canvas, image::{Rgb, RgbImage}, point::Point
};

use crate::draw::draw_text_on_center;

pub struct ContourLine<T> {
    contour: Contour<T>,
    height: Option<i32>,
}

impl<T> ContourLine<T> {
    pub fn contour(&self) -> &Contour<T> {
        &self.contour
    }

    pub fn height(&self) -> Option<i32> {
        self.height
    }
}

pub fn get_contour_lines_image<T: Font>(contour_lines: &[ContourLine<u32>], font: &T, w: u32, h: u32) -> RgbImage {
    let mut image = RgbImage::new(w, h);

    for contour_line in contour_lines {
        // Draw all contour line points
        let points = &contour_line.contour().points;
        for point in points {
            image.draw_pixel(point.x, point.y, Rgb::from([255, 0, 0]));
        }

        // Mark the first point of the contour line with height value
        let point0 = points[0];
        let scale = PxScale::from(24.0);
        let text = contour_line
            .height()
            .expect("Contour line does not have height")
            .to_string();

        draw_text_on_center(
            &mut image,
            Rgb::from([0, 255, 0]),
            point0.x as i32,
            point0.y as i32,
            scale,
            font,
            &text,
        );
    }

    image
}

pub fn retain_outer<T>(contours: &mut Vec<Contour<T>>) {
    contours.retain(|c| c.border_type == BorderType::Outer);
}

pub fn to_contour_lines<T>(contours: Vec<Contour<T>>) -> Vec<ContourLine<T>> {
    let mut contour_lines = Vec::new();
    for contour in contours {
        let contour_line = ContourLine {
            contour,
            height: None,
        };
        contour_lines.push(contour_line);
    }

    set_heights(&mut contour_lines);
    contour_lines
}

pub fn find_contour_line_height_interval(
    point: Point<usize>,
    line_height_matrix: &[Vec<Option<i32>>],
    x_range: (usize, usize)
) -> (Option<i32>, Option<i32>) {
    let (min, max) = x_range;

    let left_height = find_first_contour_line_height(
        point,
        line_height_matrix,
        (min..point.x).rev(),
    );
    let right_height =
        find_first_contour_line_height(point, line_height_matrix, point.x..max as usize);

    (left_height, right_height)
}

fn set_heights<T>(contour_lines: &mut [ContourLine<T>]) {
    let mut sorted: Vec<&mut ContourLine<T>> = contour_lines.iter_mut().collect();
    sorted.sort_by(|a, b| a.contour.parent.unwrap().cmp(&b.contour.parent.unwrap()));

    let mut height = 0;
    const GAP: i32 = 50;

    for contour_line in sorted {
        height += GAP;
        contour_line.height = Some(height);
    }
}

fn find_first_contour_line_height<I: Iterator<Item = usize>>(
    point: Point<usize>,
    line_height_matrix: &[Vec<Option<i32>>],
    range: I,
) -> Option<i32> {
    for x in range {
        let val = line_height_matrix[point.y as usize][x as usize];
        if val != None {
            return val;
        }
    }

    None
}
