use imageproc::{
    drawing::Canvas,
    image::{GrayImage, Luma},
    point::Point,
};
use std::{cmp::min, collections::VecDeque, vec};

use crate::contour_line::{ContourLine, find_contour_line_height_interval, get_bbox};

pub struct HeightMap {
    pub data: Vec<Vec<Option<i32>>>,
    pub contour_lines: Vec<ContourLine<u32>>,
}

impl HeightMap {
    pub fn new(contour_lines: Vec<ContourLine<u32>>, w: usize, h: usize) -> Self {
        Self {
            data: vec![vec![None; w]; h],
            contour_lines,
        }
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
}

impl HeightMap {
    pub fn flat_fill(&mut self) {
        let w = self.data[0].len();
        let h = self.data.len();
        let mut filled = vec![vec![false; w as usize]; h as usize];
        self.draw_contour_lines(&mut filled);

        for y in 0..h {
            for x in 0..w {
                if filled[y as usize][x as usize] || self.data[y][x] != None {
                    continue;
                }

                let bbox = get_bbox(&self.contour_lines);
                let (left, right) =
                    find_contour_line_height_interval(Point::new(x, y), &self.data, &bbox);
                let height = min(left, right).unwrap_or(0);

                self.flood_fill(x, y, height, &mut filled);

                // temp debug
                println!("filled height {}", height);
            }
        }
    }

    pub fn flood_fill(
        &mut self,
        x: usize,
        y: usize,
        replacement_value: i32,
        filled: &mut [Vec<bool>],
    ) {
        if !self.data[y][x].is_none() {
            return;
        }

        let w = self.data[0].len();
        let h = self.data.len();

        let mut queue = VecDeque::new();
        queue.push_back((x, y));

        while let Some((cx, cy)) = queue.pop_front() {
            let current = self.data[cy][cx];
            if !current.is_none() {
                continue;
            }

            self.data[cy][cx] = Some(replacement_value);
            filled[cy as usize][cx as usize] = true;

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

    fn draw_contour_lines(&mut self, filled: &mut [Vec<bool>]) {
        for cl in &self.contour_lines {
            for p in &cl.contour().points {
                self.data[p.y as usize][p.x as usize] = Some(cl.height().unwrap());
                filled[p.y as usize][p.x as usize] = true;
            }
        }
    }
}

fn height_to_u8(h: i32, max_h: i32) -> u8 {
    (h as f32 / max_h as f32 * 255.0) as u8
}
