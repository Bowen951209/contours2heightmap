use imageproc::{
    drawing::Canvas,
    image::{GrayImage, Luma},
    point::Point,
};
use std::{cmp::min, collections::VecDeque, vec};

use crate::contour_line::ContourLine;

pub struct HeightMap {
    pub data: Vec<Vec<Option<i32>>>,
    pub contour_lines: Vec<ContourLine<u32>>,
}

impl HeightMap {
    /// Return a flat-filled heightmap based on the passed in `contour_lines` and the given width and height.
    /// This function will call `flat_fill` to fill the heightmap. The resulting heightmap will look like stairs or river terrace.
    pub fn new_flat(contour_lines: Vec<ContourLine<u32>>, w: usize, h: usize) -> Self {
        let mut heightmap = Self::new_with_contour_lines_drawn(contour_lines, w, h);
        heightmap.flat_fill();
        heightmap
    }

    pub fn to_gray_image(&self) -> GrayImage {
        let w = self.data[0].len();
        let h = self.data.len();
        let mut image = GrayImage::new(w as u32, h as u32);
        for y in 0..h {
            for x in 0..w {
                let val = self.data[y][x].unwrap();
                let gray = height_to_u8(val, 200);
                image.draw_pixel(x as u32, y as u32, Luma([gray]));
            }
        }

        image
    }

    /// Create a new `HeightMap` with contour lines drawn on it. It simply calls `draw_contour_lines` to set the points and height value to `data` for each contour line.
    fn new_with_contour_lines_drawn(
        contour_lines: Vec<ContourLine<u32>>,
        w: usize,
        h: usize,
    ) -> Self {
        let mut heightmap = Self {
            data: vec![vec![None; w]; h],
            contour_lines,
        };

        heightmap.draw_contour_lines();
        heightmap
    }
}

impl HeightMap {
    fn flat_fill(&mut self) {
        let w = self.data[0].len();
        let h = self.data.len();

        for y in 0..h {
            for x in 0..w {
                if self.data[y][x].is_some() {
                    continue;
                }

                let (left, right) = find_height_interval(Point::new(x, y), &self.data);
                let height = min(left, right).unwrap_or(0);

                self.flood_fill(x, y, height);

                // temp debug
                println!("filled height {}", height);
            }
        }
    }

    fn flood_fill(&mut self, x: usize, y: usize, replacement_value: i32) {
        if self.data[y][x].is_some() {
            return;
        }

        let w = self.data[0].len();
        let h = self.data.len();

        let mut queue = VecDeque::new();
        queue.push_back((x, y));

        while let Some((cx, cy)) = queue.pop_front() {
            let current = self.data[cy][cx];
            if current.is_some() {
                continue;
            }

            self.data[cy][cx] = Some(replacement_value);

            if cx > 0 {
                queue.push_back((cx - 1, cy));
            }
            if cx + 1 < w {
                queue.push_back((cx + 1, cy));
            }
            if cy > 0 {
                queue.push_back((cx, cy - 1));
            }
            if cy + 1 < h {
                queue.push_back((cx, cy + 1));
            }
        }
    }

    /// Set the points and height value to `data` for each contour line.
    fn draw_contour_lines(&mut self) {
        for cl in &self.contour_lines {
            for p in &cl.contour().points {
                self.data[p.y as usize][p.x as usize] = Some(cl.height().unwrap());
            }
        }
    }
}

/// Find the contour line height interval where `point` is located.
/// * `point` is the point to check.
/// * `line_height_data` is the data directly after `draw_contour_lines` call.
fn find_height_interval(
    point: Point<usize>,
    line_height_data: &[Vec<Option<i32>>],
) -> (Option<i32>, Option<i32>) {
    let (min, max) = (0, line_height_data[0].len());

    let left_height = find_first_contour_line_height(point, line_height_data, (min..point.x).rev());
    let right_height =
        find_first_contour_line_height(point, line_height_data, point.x..max as usize);

    (left_height, right_height)
}

/// Start from `point` and find the first encounter contour line's height in the given range.
/// * `point` is the point to check.
/// * `line_height_data` is the data directly after `draw_contour_lines` call.
/// * `range` is the range to check.
fn find_first_contour_line_height<I: Iterator<Item = usize>>(
    point: Point<usize>,
    line_height_data: &[Vec<Option<i32>>],
    range: I,
) -> Option<i32> {
    for x in range {
        let val = line_height_data[point.y as usize][x as usize];
        if val != None {
            return val;
        }
    }

    None
}

fn height_to_u8(h: i32, max_h: i32) -> u8 {
    (h as f32 / max_h as f32 * 255.0) as u8
}
