use imageproc::{
    definitions::Image,
    drawing::Canvas,
    image::{GrayImage, Luma, Rgb, RgbImage},
    point::Point,
};
use rstar::RTree;
use std::{collections::VecDeque, vec};

use crate::contour_line::{ContourLine, ContourLineInterval, GAP, find_contour_line_interval};

/// The mode of distance calculation.
/// This is for [`HeightMap::distance_transform`]
/// , where you can decide whether you want to generate
/// a distance field to the nearest **inner** or **outer** contour line.
#[derive(Debug, Clone, Copy)]
enum DistanceMode {
    ToInner,
    ToOuter,
}

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

    pub fn to_rgb_image(&self) -> RgbImage {
        let w = self.data[0].len();
        let h = self.data.len();
        let mut image = RgbImage::new(w as u32, h as u32);
        for y in 0..h {
            for x in 0..w {
                let val = self.data[y][x].unwrap();
                let rgb = self.height_to_rgb(val);
                image.draw_pixel(x as u32, y as u32, rgb);
            }
        }

        image
    }

    /// Create a new [`HeightMap`] with contour lines drawn on it.
    ///
    /// It simply calls [`HeightMap::draw_contour_lines`] to set
    /// the points and height value to [`HeightMap::data`] for each contour line.
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

        // A map where you can look up what are a pixel's inner and outer contour lines
        let mut interval_map = vec![vec![None; w]; h];

        // Set the points on contour lines to have same outer and inner.
        // This is for building walls between different levels for the following flood fill.
        for cl in &self.contour_line_tree {
            for p in &cl.contour().points {
                interval_map[p.y][p.x] = Some(ContourLineInterval::new(cl, cl));
            }
        }

        // Flood fill interval_map
        for y in 0..h {
            for x in 0..w {
                if interval_map[y][x].is_some() {
                    continue;
                }

                let interval =
                    find_contour_line_interval(Point::new(x, y), &self.contour_line_tree);

                flood_fill(&mut interval_map, x, y, interval);
            }
        }

        // Generate distance fields

        // Each pixel stores the squared distance to the nearest OUTER contour line
        let mut outer_distance_field = Image::new(w as u32, h as u32);
        self.euclidean_distance_transform(
            &interval_map,
            &mut outer_distance_field,
            DistanceMode::ToOuter,
        );

        // Each pixel stores the squared distance to the nearest INNER contour line
        let mut inner_distance_field = Image::new(w as u32, h as u32);
        self.euclidean_distance_transform(
            &interval_map,
            &mut inner_distance_field,
            DistanceMode::ToInner,
        );

        // Linear interpolation and fill to self.data
        for y in 0..h {
            for x in 0..w {
                let point = Point::new(x, y);
                let interval = interval_map[y][x].as_ref().unwrap();
                let val = linear_at(
                    &point,
                    interval,
                    &outer_distance_field,
                    &inner_distance_field,
                );
                self.data[y][x] = Some(val as i32);
            }
        }
    }

    /// Computes the squared Euclidean distance field and writes it to `buffer` in linear time.
    /// The algorithm is based on [Distance Transforms of Sampled Functions].
    ///
    /// This is not a direct use of [`imageproc::distance_transform::euclidean_squared_distance_transform`]
    /// because more control is required.
    ///
    /// This function calculates the distance from each pixel to the nearest outer or inner contour line,
    /// depending on the `distance_mode` parameter.
    ///
    /// Unlike traditional distance transforms that generate a distance field from static features,
    /// this implementation builds the field layer by layer, where each layer corresponds to the region
    /// between contour lines.
    ///
    /// Alternatively, you could build separate distance fields for each contour line height (as in [738b3e1]),
    /// but that approach computes unnecessary pixels and uses more time and memory.
    ///
    /// [738b3e1]: https://github.com/Bowen951209/contours2heightmap/commit/738b3e10e5a4b0a5bf77b76e7a9bec5e16e28e65
    /// [Distance Transforms of Sampled Functions]: https://www.cs.cornell.edu/~dph/papers/dt.pdf
    fn euclidean_distance_transform(
        &self,
        interval_map: &Vec<Vec<Option<ContourLineInterval>>>,
        buffer: &mut Image<Luma<f32>>,
        distance_mode: DistanceMode,
    ) {
        let (w, h) = buffer.dimensions();
        let (w, h) = (w as usize, h as usize);

        // Put these buffer outside the loop to avoid reallocating them every iteration
        let mut y_envelope = Envelope::new(h);
        let mut y_result_buffer = vec![f32::NAN; h];
        let mut x_envelope = Envelope::new(w);
        let mut x_result_buffer = vec![f32::NAN; w];

        // Loop through each height level.
        // Regions regarding the current height will be filled.
        let mut current_height = GAP;
        while current_height <= self.max_height {
            for x in 0..w {
                // `f` is directly named from the paper.
                // Return 0 if a point is a "feature" or "boundary", and INFINITY otherwise.
                // Basically the algorithm is trying to minimize values, so adding INFINITY
                // is to avoid minimizing on a non-feature point.
                let f = |y: usize| {
                    let interval = interval_map[y][x].as_ref().unwrap();
                    let point = Point::new(x, y);

                    match distance_mode {
                        DistanceMode::ToInner => {
                            if interval
                                .inners()
                                .iter()
                                .any(|inner| inner.contour().points.contains(&point))
                            {
                                0.0
                            } else {
                                f32::INFINITY
                            }
                        }

                        DistanceMode::ToOuter => interval
                            .outer()
                            .filter(|outer| outer.contour().points.contains(&point))
                            .map(|_| 0.0)
                            .unwrap_or(f32::INFINITY),
                    }
                };

                // Because we don't need the whole buffer to be filled in reference to the current height,
                // we can ignore certain pixels. If a pixel's interval is not the interval this iteration
                // is dealing with, we can skip it.
                let ignore =
                    |y| is_height_match(x, y, &interval_map, current_height, distance_mode);

                distance_transform_1d(&f, &mut y_envelope, &mut y_result_buffer, &ignore);

                // copy column buffer (y_result_buffer) to main buffer
                for y in 0..h {
                    if !ignore(y) {
                        // Only copy the not ignored pixels. Do not overwrite other pixels.
                        buffer.put_pixel(x as u32, y as u32, Luma([y_result_buffer[y]]));
                    }
                }
            }

            for y in 0..h {
                let f = |x| buffer.get_pixel(x as u32, y as u32).0[0] as f32;
                let ignore =
                    |x| is_height_match(x, y, &interval_map, current_height, distance_mode);

                distance_transform_1d(&f, &mut x_envelope, &mut x_result_buffer, &ignore);

                // copy from buffer to data
                for x in 0..w {
                    if !ignore(x) {
                        buffer.put_pixel(x as u32, y as u32, Luma([x_result_buffer[x]]));
                    }
                }
            }

            current_height += GAP;
        }
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

    fn height_to_rgb(&self, h: i32) -> Rgb<u8> {
        let t = h as f32 / self.max_height as f32;

        let r = lerp(128, 255, t) as u8;
        let g = lerp(0, 0, t) as u8;
        let b = lerp(128, 0, t) as u8;

        Rgb([r, g, b])
    }
}

/// The parabola lower-envelope structure describe in
/// [Distance Transforms of Sampled Functions](https://www.cs.cornell.edu/~dph/papers/dt.pdf)
struct Envelope {
    /// x coordinates of the parabola lowest points
    v: Vec<usize>,
    /// x coordinates of parabola intersections
    z: Vec<f32>,
}

impl Envelope {
    fn new(n: usize) -> Self {
        Self {
            v: vec![0; n],
            z: vec![f32::NAN; n + 1],
        }
    }
}

// TODO: Fix flood fill performance.
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
    interval: &ContourLineInterval,
    outer_distance_field: &Image<Luma<f32>>,
    inner_distance_field: &Image<Luma<f32>>,
) -> i32 {
    let (Some(outer), inners) = (interval.outer(), interval.inners()) else {
        return 0;
    };

    if inners.is_empty() {
        return outer.height().unwrap();
    }

    let outer_height = outer.height().unwrap();
    let distance_to_outer = outer_distance_field
        .get_pixel(point.x as u32, point.y as u32)
        .0[0]
        .sqrt();

    let inner_height = inners[0].height().unwrap();
    let distance_to_inner = inner_distance_field
        .get_pixel(point.x as u32, point.y as u32)
        .0[0]
        .sqrt();

    let total_distance = distance_to_outer + distance_to_inner;
    let t = distance_to_outer / total_distance;
    return lerp(outer_height, inner_height, t as f32) as i32;
}

fn lerp(a: i32, b: i32, t: f32) -> f32 {
    a as f32 * (1.0 - t) + b as f32 * t
}

/// Based on [Distance Transforms of Sampled Functions](https://www.cs.cornell.edu/~dph/papers/dt.pdf).
fn distance_transform_1d(
    f: &impl Fn(usize) -> f32,
    envelope: &mut Envelope,
    result: &mut [f32],
    ignore: &impl Fn(usize) -> bool,
) {
    let n = result.len();

    // Index of rightmost parabola
    let mut k = 0;

    // x coordinates of the parabola lowest points
    let v = &mut envelope.v;
    v[0] = 0;

    // x coordinates of parabola intersections
    let z = &mut envelope.z;
    z[0] = f32::NEG_INFINITY;
    z[1] = f32::INFINITY;

    for q in 1..n {
        if ignore(q) {
            continue;
        }

        if f(q) == f32::INFINITY {
            continue;
        }

        if k == 0 && f(v[k]) == f32::INFINITY {
            v[k] = q;
            z[k] = f32::NEG_INFINITY;
            z[k + 1] = f32::INFINITY;
            continue;
        }

        let mut s = parabola_intersection(f, v[k], q);
        while s <= z[k] {
            k -= 1;
            s = parabola_intersection(f, v[k], q);
        }

        k += 1;
        v[k] = q;
        z[k] = s;
        z[k + 1] = f32::INFINITY;
    }

    k = 0;
    for q in 0..n {
        if ignore(q) {
            continue;
        }

        while z[k + 1] < q as f32 {
            k += 1;
        }
        let dist = q as f32 - v[k] as f32;
        result[q] = dist * dist + f(v[k]) as f32;
    }
}

// Modified from imageproc::distance_transform::intersection
fn parabola_intersection(f: impl Fn(usize) -> f32, p: usize, q: usize) -> f32 {
    // The intersection s of the two parabolas satisfies:
    //
    // f[q] + (q - s) ^ 2 = f[p] + (s - q) ^ 2
    //
    // Rearranging gives:
    //
    // s = [( f[q] + q ^ 2 ) - ( f[p] + p ^ 2 )] / (2q - 2p)
    let fq = f(q);
    let fp = f(p);
    let p = p as f32;
    let q = q as f32;

    ((fq + q * q) - (fp + p * p)) / (2.0 * q - 2.0 * p)
}

// A very specific function for euclidean_distance_transform
// to decide wheter to ignore a pixel.
fn is_height_match(
    x: usize,
    y: usize,
    interval_map: &Vec<Vec<Option<ContourLineInterval>>>,
    current_height: i32,
    distance_mode: DistanceMode,
) -> bool {
    let interval = interval_map[y][x].as_ref().unwrap();

    match distance_mode {
        DistanceMode::ToInner => interval
            .inners()
            .first()
            .map_or(true, |inner| inner.height().unwrap() != current_height),
        DistanceMode::ToOuter => interval
            .outer()
            .map_or(true, |outer| outer.height().unwrap() != current_height),
    }
}

#[cfg(test)]
mod test {
    use std::path::Path;

    use crate::{contour_line, heightmap::HeightMap};

    #[test]
    fn test_one_hill_removed_and_two_hills_have_same_linear_height_at_same_point() {
        let file_path_one_hill_removed = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("assets/one_hill_removed_from_two_hills.png");
        let (contour_lines_one_hill_removed, w, h) =
            contour_line::get_contour_line_tree_from(file_path_one_hill_removed.to_str().unwrap());
        let one_hill_removed_heightmap =
            HeightMap::new_linear(contour_lines_one_hill_removed, w as usize, h as usize);

        let file_path_two_hills =
            Path::new(env!("CARGO_MANIFEST_DIR")).join("assets/two_hills.png");
        let (contour_lines_two_hills, w, h) =
            contour_line::get_contour_line_tree_from(file_path_two_hills.to_str().unwrap());
        let two_hills_heightmap =
            HeightMap::new_linear(contour_lines_two_hills, w as usize, h as usize);

        let point_y = 114;
        let point_x = 228;

        assert_eq!(
            one_hill_removed_heightmap.data[point_y][point_x],
            two_hills_heightmap.data[point_y][point_x]
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
