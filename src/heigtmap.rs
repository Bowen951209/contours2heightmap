use imageproc::{
    drawing::Canvas,
    image::{GrayImage, Luma},
    point::Point,
};
use indicatif::{MultiProgress, ProgressBar, ProgressStyle};
use rayon::iter::{IndexedParallelIterator, IntoParallelRefMutIterator, ParallelIterator};
use std::{collections::VecDeque, vec};

use crate::contour_line::{ContourLine, find_contour_line_interval};

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

                flood_fill(&mut self.data, x, y, height);

                // temp debug
                println!("filled height {}", height);
            }
        }
    }

    fn linear_fill(&mut self) {
        let w = self.data[0].len();
        let h = self.data.len();
        let mut intervals: Vec<Vec<Option<(Option<&ContourLine>, Option<&ContourLine>)>>> =
            vec![vec![None; w]; h];

        // Progress bar
        let m = MultiProgress::new();
        let sty = ProgressStyle::with_template(
            "{percent:.2}% {bar:20.cyan/blue} {pos}/{len} [{elapsed_precise}] {msg}",
        )
        .unwrap();
        let n = (w * h) as u64;
        let pb1 = m.add(ProgressBar::new(n));
        pb1.set_style(sty.clone());
        let pb2 = m.add(ProgressBar::new(n));
        pb2.set_style(sty);

        // Set the points on contour lines to have same outer and inner. This is for building walls between different levels for the following flood fill.
        pb1.set_message("Setting points on contour lines...");
        for cl in &self.contour_lines {
            for p in &cl.contour().points {
                intervals[p.y][p.x] = Some((Some(cl), Some(cl)));
            }
        }

        // Use flood fill to set intervals
        pb1.set_message("Finding intervals for points...");
        for y in 0..h {
            for x in 0..w {
                pb1.inc(1);
                if intervals[y][x].is_some() {
                    continue;
                }

                let interval = find_contour_line_interval(Point::new(x, y), &self.contour_lines);
                flood_fill(&mut intervals, x, y, interval);
                pb1.set_message(format!("flood fill at ({}, {})", x, y));
            }
        }
        pb1.finish_with_message("Intervals finding done.");

        // Linear fill data
        pb2.set_message("Filling data...");

        self.data.par_iter_mut().enumerate().for_each(|(y, row)| {
            row.iter_mut().enumerate().for_each(|(x, val)| {
                if !val.is_some() {
                    let height = linear_at(&Point::new(x, y), &intervals);
                    *val = Some(height);
                }
                pb2.inc(1);
            });
        });

        pb2.finish_with_message("linear fill done");
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

fn flood_fill<T: Clone>(data: &mut [Vec<Option<T>>], x: usize, y: usize, replacement_value: T) {
    let w = data[0].len();
    let h = data.len();

    let mut queue = VecDeque::new();
    queue.push_back((x, y));

    while let Some((cx, cy)) = queue.pop_front() {
        let current = &data[cy][cx];
        if current.is_some() {
            continue;
        }

        data[cy][cx] = Some(replacement_value.clone());

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

fn linear_at(
    point: &Point<usize>,
    intervals: &[Vec<Option<(Option<&ContourLine>, Option<&ContourLine>)>>],
) -> i32 {
    let (inside, outside) =
        intervals[point.y][point.x].expect("Interval not properly filled, found a None.");
    if let (Some(inside), Some(outside)) = (inside, outside) {
        let inside_height = inside.height().unwrap();
        let outside_height = outside.height().unwrap();
        let distance_inside = inside.find_nearest_distance(point);
        let distance_outside = outside.find_nearest_distance(point);
        let distance_whole = distance_inside + distance_outside;
        let t = distance_inside / distance_whole;
        return lerp(inside_height, outside_height, t) as i32;
    }

    match inside {
        Some(cl) => cl.height().unwrap(),
        None => 0,
    }
}

fn lerp(a: i32, b: i32, t: f32) -> f32 {
    a as f32 * (1.0 - t) + b as f32 * t
}

fn height_to_u8(h: i32, max_h: i32) -> u8 {
    (h as f32 / max_h as f32 * 255.0) as u8
}
