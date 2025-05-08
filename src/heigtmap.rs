use imageproc::image::{GrayImage, Luma};
use std::collections::VecDeque;

use crate::contour_line::ContourLine;

struct HeightMap {
    img: GrayImage,
}

impl HeightMap {
    pub fn flat_from<T>(contour_lines: ContourLine<T>, w: u32, h: u32) -> Self {
        let mut img = GrayImage::new(w, h);

        let mut filled = vec![vec![false; w as usize]; h as usize];
        for y in 0..h {
            for x in 0..w {
                if filled[y as usize][x as usize] {
                    continue;
                }

                flood_fill_gray(&mut img, &mut filled, x, y, 0, height);
            }
        }

        Self { img }
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
    // if x >= width || y >= height {
    //     return;
    // }

    let mut queue = VecDeque::new();
    queue.push_back((x, y));

    while let Some((cx, cy)) = queue.pop_front() {
        // if cx >= width || cy >= height {
        //     continue;
        // }

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
