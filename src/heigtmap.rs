use imageproc::{
    drawing::Canvas,
    image::{GrayImage, Luma},
    point::Point,
};
use std::{
    cmp::{max, min},
    collections::VecDeque,
};

use crate::contour_line::{ContourLine, find_contour_line_height_interval_gray, get_bbox};

pub struct HeightMap {
    pub img: GrayImage,
}

impl HeightMap {
    pub fn flat_fill(&mut self, contour_lines: &[ContourLine<u32>]) {
        let (w, h) = self.img.dimensions();
        let mut filled = vec![vec![false; w as usize]; h as usize];
        draw_contour_lines_gray(&mut self.img, &mut filled, contour_lines);

        for y in 0..h {
            for x in 0..w {
                if filled[y as usize][x as usize] || self.img.get_pixel(x, y)[0] != 0 {
                    continue;
                }

                let bbox = get_bbox(contour_lines);
                let (left, right) =
                    find_contour_line_height_interval_gray(Point::new(x, y), &self.img, &bbox);
                let gray = max(min(left, right), 1); // 1 for flood fill checker
                flood_fill_gray(&mut self.img, &mut filled, x, y, 0, gray);

                // temp debug
                println!("filled gray {}", gray);
            }
        }
    }
}

fn draw_contour_lines_gray(
    img: &mut GrayImage,
    filled: &mut [Vec<bool>],
    contour_lines: &[ContourLine<u32>],
) {
    for cl in contour_lines {
        for p in &cl.contour().points {
            let color = height_to_u8(cl.height().unwrap(), 200);
            img.draw_pixel(p.x, p.y, Luma([color]));
            filled[p.y as usize][p.x as usize] = true;
        }
    }
}

fn flood_fill_gray(
    img: &mut GrayImage,
    filled: &mut [Vec<bool>],
    x: u32,
    y: u32,
    target_value: u8,
    replacement_value: u8,
) {
    if target_value == replacement_value {
        return;
    }

    let (width, height) = img.dimensions();

    let mut queue = VecDeque::new();
    queue.push_back((x, y));

    while let Some((cx, cy)) = queue.pop_front() {
        let current = img.get_pixel(cx, cy)[0];
        if current != target_value {
            continue;
        }

        img.put_pixel(cx, cy, Luma([replacement_value]));
        filled[cy as usize][cx as usize] = true;

        if cx > 0 {
            queue.push_back((cx - 1, cy));
        }
        if cx + 1 < width {
            queue.push_back((cx + 1, cy));
        }
        if cy > 0 {
            queue.push_back((cx, cy - 1));
        }
        if cy + 1 < height {
            queue.push_back((cx, cy + 1));
        }
    }
}

fn height_to_u8(h: i32, max_h: i32) -> u8 {
    (h as f32 / max_h as f32 * 255.0) as u8
}
