use imageproc::{
    drawing::Canvas,
    image::{GrayImage, Luma},
    point::Point,
};
use indicatif::{ProgressBar, ProgressStyle};
use std::{collections::VecDeque, io::Write, vec};

use crate::contour_line::ContourLine;

pub struct HeightMap {
    pub data: Vec<Vec<Option<i32>>>,
    pub contour_lines: Vec<ContourLine>,
}

impl HeightMap {
    /// Return a flat-filled heightmap based on the passed in `contour_lines` and the given width and height.
    /// This function will call `flat_fill` to fill the heightmap. The resulting heightmap will look like stairs or river terrace.
    pub fn new_flat(contour_lines: Vec<ContourLine>, w: usize, h: usize) -> Self {
        let mut heightmap = Self::new_with_contour_lines_drawn(contour_lines, w, h);
        heightmap.flat_fill();
        heightmap
    }

    pub fn new_linear(contour_lines: Vec<ContourLine>, w: usize, h: usize) -> Self {
        let mut heightmap = Self::new_with_contour_lines_drawn(contour_lines, w, h);
        heightmap.linear_fill();
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
    fn new_with_contour_lines_drawn(contour_lines: Vec<ContourLine>, w: usize, h: usize) -> Self {
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

                let (inside, _) = find_contour_line_interval(Point::new(x, y), &self.contour_lines);
                let height = match inside {
                    Some(_) => inside.unwrap().height().unwrap(),
                    None => 0,
                };

                self.flood_fill(x, y, height);

                // temp debug
                println!("filled height {}", height);
            }
        }
    }

    fn linear_fill(&mut self) {
        let w = self.data[0].len();
        let h = self.data.len();
        let elapsed = std::time::Instant::now().elapsed();
        let pb = ProgressBar::new((w * h) as u64).with_elapsed(elapsed);
        pb.set_style(
            ProgressStyle::default_bar()
                .template("{percent:.2}% {bar:20.cyan/blue} {pos}/{len} [{elapsed_precise}] {msg}")
                .unwrap(),
        );

        for y in 0..h {
            for x in 0..w {
                if self.data[y][x].is_some() {
                    continue;
                }

                let height = self.linear_at(&Point::new(x, y));
                self.data[y][x] = Some(height);
                pb.set_position((y * w + x) as u64);
                pb.set_message(format!("linear fill at ({}, {})", x, y));
            }
        }

        pb.finish_with_message("linear fill done");
    }

    fn linear_at(&self, point: &Point<usize>) -> i32 {
        let (inside, outside) = find_contour_line_interval(*point, &self.contour_lines);
        if let (Some(inside), Some(outside)) = (inside, outside) {
            let inside_height = inside.height().unwrap();
            let outside_height = outside.height().unwrap();
            let distance_inside = find_nearest_distance(inside, *point);
            let distance_outside = find_nearest_distance(outside, *point);
            let distance_whole = distance_inside + distance_outside;
            let t = distance_inside / distance_whole;
            return lerp(inside_height, outside_height, t) as i32;
        }

        match inside {
            Some(cl) => cl.height().unwrap(),
            None => 0,
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

/// Find the contour line interval where `point` is located.
fn find_contour_line_interval(
    point: Point<usize>,
    sorted_contour_lines: &[ContourLine],
) -> (Option<&ContourLine>, Option<&ContourLine>) {
    let mut inside = None;
    let mut outside = None;

    for cl in sorted_contour_lines {
        if cl.is_point_inside(&point) {
            inside = Some(cl);
        } else {
            outside = Some(cl);
            break;
        }
    }

    (inside, outside)
}

fn find_nearest_distance(contour_line: &ContourLine, point: Point<usize>) -> f32 {
    contour_line
        .contour()
        .points
        .iter()
        .map(|p| distance(&point, p))
        .fold(f32::MAX, f32::min)
}

fn distance(a: &Point<usize>, b: &Point<usize>) -> f32 {
    let dx = a.x as f32 - b.x as f32;
    let dy = a.y as f32 - b.y as f32;
    (dx * dx + dy * dy).sqrt()
}

fn lerp(a: i32, b: i32, t: f32) -> f32 {
    a as f32 * (1.0 - t) + b as f32 * t
}

fn height_to_u8(h: i32, max_h: i32) -> u8 {
    (h as f32 / max_h as f32 * 255.0) as u8
}
