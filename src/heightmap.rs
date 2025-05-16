use imageproc::{
    drawing::Canvas,
    image::{GrayImage, Luma},
    point::Point,
};
use indicatif::{MultiProgress, ProgressBar, ProgressStyle};
use rayon::iter::{IndexedParallelIterator, IntoParallelRefMutIterator, ParallelIterator};
use rstar::RTree;
use std::{collections::VecDeque, vec};

use crate::contour_line::{self, ContourLine, ContourLineInterval, find_contour_line_interval};

pub struct HeightMap {
    pub data: Vec<Vec<Option<i32>>>,
    pub contour_line_tree: RTree<ContourLine>,
    max_height: i32,
}

impl HeightMap {
    /// Return a flat-filled heightmap based on the passed in `contour_lines` and the given width and height.
    /// This function will call `flat_fill` to fill the heightmap. The resulting heightmap will look like stairs or river terrace.
    pub fn new_flat(contour_line_tree: RTree<ContourLine>, w: usize, h: usize) -> Self {
        let mut heightmap = Self::new_with_contour_lines_drawn(contour_line_tree, w, h);
        heightmap.flat_fill();
        heightmap
    }

    pub fn new_linear(contour_line_tree: RTree<ContourLine>, w: usize, h: usize) -> Self {
        let mut heightmap = Self::new_with_contour_lines_drawn(contour_line_tree, w, h);
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
                let gray = self.height_to_u8(val);
                image.draw_pixel(x as u32, y as u32, Luma([gray]));
            }
        }

        image
    }

    /// Create a new `HeightMap` with contour lines drawn on it. It simply calls `draw_contour_lines` to set the points and height value to `data` for each contour line.
    fn new_with_contour_lines_drawn(
        contour_line_tree: RTree<ContourLine>,
        w: usize,
        h: usize,
    ) -> Self {
        let max_height = contour_line_tree
            .iter()
            .map(|cl| cl.height().unwrap())
            .max()
            .unwrap();
        let mut heightmap = Self {
            data: vec![vec![None; w]; h],
            contour_line_tree,
            max_height,
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

                let interval =
                    find_contour_line_interval(Point::new(x, y), &self.contour_line_tree);

                let height = match interval.outer() {
                    Some(outside) => outside.height().unwrap(),
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
        let mut intervals = vec![vec![None; w]; h];

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
        for cl in &self.contour_line_tree {
            for p in &cl.contour().points {
                intervals[p.y][p.x] = Some(ContourLineInterval::new(cl, cl));
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

                let interval =
                    find_contour_line_interval(Point::new(x, y), &self.contour_line_tree);
                flood_fill(&mut intervals, x, y, interval);
                pb1.set_message(format!("flood fill at ({}, {})", x, y));
            }
        }
        pb1.finish_with_message("Intervals finding done.");

        // Linear fill data
        pb2.set_message("Filling data...");

        self.data.par_iter_mut().enumerate().for_each(|(y, row)| {
            row.iter_mut().enumerate().for_each(|(x, val)| {
                if val.is_none() {
                    let interval = intervals[y][x].as_ref().expect("interval should exist");
                    let height = linear_at(&Point::new(x, y), interval);
                    *val = Some(height);
                }
                pb2.inc(1);
            });
        });

        pb2.finish_with_message("linear fill done");
    }

    /// Set the points and height value to `data` for each contour line.
    fn draw_contour_lines(&mut self) {
        for cl in &self.contour_line_tree {
            for p in &cl.contour().points {
                self.data[p.y as usize][p.x as usize] = Some(cl.height().unwrap());
            }
        }
    }

    fn height_to_u8(&self, h: i32) -> u8 {
        (h as f32 / self.max_height as f32 * 255.0) as u8
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

fn linear_at(point: &Point<usize>, interval: &ContourLineInterval) -> i32 {
    if let (Some(outer), inners) = (interval.outer(), interval.inners()) {
        if !inners.is_empty() {
            let outer_height = outer.height().unwrap();
            let (_, distance_to_outer) = outer.find_nearest_point(point);

            let inner_height = inners[0].height().unwrap();
            let (_, distance_to_inner) = contour_line::find_nearest_point(point, inners);
            let total_distance = distance_to_outer + distance_to_inner;
            let t = distance_to_outer / total_distance;
            return lerp(outer_height, inner_height, t) as i32;
        }
    }

    interval.outer().map_or(0, |cl| cl.height().unwrap())
}

fn lerp(a: i32, b: i32, t: f32) -> f32 {
    a as f32 * (1.0 - t) + b as f32 * t
}

mod test {
    use std::path::Path;

    use imageproc::point::Point;

    use crate::{
        contour_line,
        heightmap::{HeightMap, linear_at},
    };

    #[test]
    fn test_one_hill_removed_and_two_hills_have_same_linear_height_at_same_point() {
        let file_path_one_hill_removed = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("assets/one_hill_removed_from_two_hills.png");
        let (contour_lines_one_hill_removed, w, h) =
            contour_line::get_contour_line_tree_from(file_path_one_hill_removed.to_str().unwrap());
        let one_hill_removed_heightmap = HeightMap::new_with_contour_lines_drawn(
            contour_lines_one_hill_removed,
            w as usize,
            h as usize,
        );

        let file_path_two_hills =
            Path::new(env!("CARGO_MANIFEST_DIR")).join("assets/two_hills.png");
        let (contour_lines_two_hills, w, h) =
            contour_line::get_contour_line_tree_from(file_path_two_hills.to_str().unwrap());
        let two_hills_heightmap = HeightMap::new_with_contour_lines_drawn(
            contour_lines_two_hills,
            w as usize,
            h as usize,
        );

        let point = Point::new(228, 114);
        let one_hill_removed_interval = contour_line::find_contour_line_interval(
            point,
            &one_hill_removed_heightmap.contour_line_tree,
        );
        let two_hills_interval =
            contour_line::find_contour_line_interval(point, &two_hills_heightmap.contour_line_tree);

        assert_eq!(
            linear_at(&point, &one_hill_removed_interval),
            linear_at(&point, &two_hills_interval)
        );
    }

    #[test]
    fn test_flat_fill_layer_one_hill() {
        let file_path = Path::new(env!("CARGO_MANIFEST_DIR")).join("assets/one_hill.png");
        let (contour_lines, w, h) =
            contour_line::get_contour_line_tree_from(file_path.to_str().unwrap());
        let heightmap = HeightMap::new_flat(contour_lines, w as usize, h as usize);

        // x=15, y=79
        assert_eq!(heightmap.data[79][15].unwrap(), 0);

        // x=157, y=182
        assert_eq!(heightmap.data[182][157].unwrap(), contour_line::GAP);

        // x=185, y=109
        assert_eq!(heightmap.data[109][185].unwrap(), contour_line::GAP * 2);

        // x=110, y=85
        assert_eq!(heightmap.data[85][110].unwrap(), contour_line::GAP * 3);

        // x=128, y=89
        assert_eq!(heightmap.data[89][128].unwrap(), contour_line::GAP * 4);
    }

    #[test]
    fn test_flat_fill_layer_two_hills() {
        let file_path = Path::new(env!("CARGO_MANIFEST_DIR")).join("assets/two_hills.png");
        let (contour_lines, w, h) =
            contour_line::get_contour_line_tree_from(file_path.to_str().unwrap());
        let heightmap = HeightMap::new_flat(contour_lines, w as usize, h as usize);

        // x=24, y=185
        assert_eq!(heightmap.data[185][24].unwrap(), 0);

        // x=148, y=174
        assert_eq!(heightmap.data[174][148].unwrap(), contour_line::GAP);

        // x=110, y=164
        assert_eq!(heightmap.data[164][110].unwrap(), contour_line::GAP * 2);

        // x=129, y=147
        assert_eq!(heightmap.data[147][129].unwrap(), contour_line::GAP * 3);
        // x=174, y=114
        assert_eq!(heightmap.data[114][174].unwrap(), contour_line::GAP * 3);

        // x=177, y=122
        assert_eq!(heightmap.data[122][177].unwrap(), contour_line::GAP * 4);
        // x=133, y=110
        assert_eq!(heightmap.data[110][133].unwrap(), contour_line::GAP * 4);

        // x=121, y=104
        assert_eq!(heightmap.data[104][121].unwrap(), contour_line::GAP * 5);
        // x=197, y=161
        assert_eq!(heightmap.data[161][197].unwrap(), contour_line::GAP * 5);

        // x=194, y=132
        assert_eq!(heightmap.data[132][194].unwrap(), contour_line::GAP * 6);
        // x=113, y=122
        assert_eq!(heightmap.data[122][113].unwrap(), contour_line::GAP * 6);

        // x=198, y=143
        assert_eq!(heightmap.data[143][198].unwrap(), contour_line::GAP * 7);
        // x=104, y=118
        assert_eq!(heightmap.data[118][104].unwrap(), contour_line::GAP * 7);

        // x=91, y=111
        assert_eq!(heightmap.data[111][91].unwrap(), contour_line::GAP * 8);
    }
}
