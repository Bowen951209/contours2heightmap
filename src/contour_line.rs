use std::u32;

use imageproc::{
    contours::{BorderType, Contour},
    image::{self},
    point::Point,
};

pub struct ContourLine {
    contour: Contour<usize>,
    height: Option<i32>,
    max_x: usize,
}

impl ContourLine {
    pub fn new(contour: Contour<usize>) -> Self {
        let max_x = contour.points.iter().map(|p| p.x).max().unwrap();
        Self {
            contour,
            height: None,
            max_x,
        }
    }

    pub fn is_point_inside(&self, point: &Point<usize>) -> bool {
        let mut hit_count: u32 = 0;
        let mut last_hit_point: Option<&Point<usize>> = None;
        for x in point.x..=self.max_x {
            for i in 0..self.contour.points.len() {
                let p = &self.contour.points[i];
                if p.x == x && p.y == point.y && !self.is_extremum(i) {
                    match last_hit_point {
                        Some(last_hit_point) => {
                            if p.x != last_hit_point.x + 1 {
                                hit_count += 1;
                            }
                        }
                        None => {
                            hit_count += 1;
                        }
                    }
                    last_hit_point = Some(p);
                }
            }
        }

        hit_count % 2 == 1
    }

    pub fn find_nearest_distance(&self, point: &Point<usize>) -> f32 {
        self.contour()
            .points
            .iter()
            .map(|p| distance(&point, p))
            .fold(f32::MAX, f32::min)
    }

    pub fn contour(&self) -> &Contour<usize> {
        &self.contour
    }

    pub fn height(&self) -> Option<i32> {
        self.height
    }

    fn is_extremum(&self, point_index: usize) -> bool {
        let points = &self.contour.points;
        let p = points[point_index];
        let mut right_index = point_index as i32 + 1;
        while self.point_index_wrap(right_index).y == p.y {
            right_index += 1;
        }
        let right_ordering = self.point_index_wrap(right_index).y.cmp(&p.y);

        let mut left_index = point_index as i32 - 1;
        while self.point_index_wrap(left_index).y == p.y {
            left_index -= 1;
        }
        let left_ordering = self.point_index_wrap(left_index).y.cmp(&p.y);

        right_ordering == left_ordering
    }

    fn point_index_wrap(&self, index: i32) -> &Point<usize> {
        let points = &self.contour.points;
        let len = points.len() as i32;
        let wrapped_index = ((index % len + len) % len) as usize;
        &points[wrapped_index]
    }
}

pub fn get_contour_lines_from(file_path: &str) -> (Vec<ContourLine>, u32, u32) {
    // Load the image
    let dyn_img = image::open(file_path).expect("Failed to open image file");
    let (w, h) = (dyn_img.width(), dyn_img.height());

    // To grayscale, and then we can find contours
    let grayscale = dyn_img.to_luma8();
    let mut contours: Vec<Contour<usize>> = imageproc::contours::find_contours(&grayscale);

    // find_contours finds outer and inner contours. We only retain outers as representation
    retain_outer(&mut contours);

    // Convert Contours to ContourLines. The heights of each contour line are then set
    (to_contour_lines(contours), w, h)
}

/// Find the contour line interval where `point` is located. The passed in `contour_lines` should be sorted with their parents order.
pub fn find_contour_line_interval(
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

fn retain_outer<T>(contours: &mut Vec<Contour<T>>) {
    contours.retain(|c| c.border_type == BorderType::Outer);
}

fn to_contour_lines(contours: Vec<Contour<usize>>) -> Vec<ContourLine> {
    let mut contour_lines: Vec<_> = contours.into_iter().map(ContourLine::new).collect();

    set_heights(&mut contour_lines);
    contour_lines
}

/// Set the heights of the contour lines by their order in the passed in vector.
/// The default order `imageproc` returns is sorted by the parent order, which starts from outer to inner.
/// Thus, by default, this function makes the outermost contour line the lowest and the innermost the highest.
fn set_heights(sorted_contour_lines: &mut [ContourLine]) {
    let mut height = 0;
    const GAP: i32 = 50;

    for contour_line in sorted_contour_lines {
        height += GAP;
        contour_line.height = Some(height);
    }
}

fn distance(a: &Point<usize>, b: &Point<usize>) -> f32 {
    let dx = a.x as f32 - b.x as f32;
    let dy = a.y as f32 - b.y as f32;
    (dx * dx + dy * dy).sqrt()
}

#[cfg(test)]
mod tests {
    use std::{env, path::Path};

    use super::get_contour_lines_from;
    use imageproc::point::Point;

    #[test]
    fn points_inside_simple_contour_lines() {
        let file_path = Path::new(env!("CARGO_MANIFEST_DIR")).join("assets/contour_lines.png");
        let (contour_lines, _, _) = get_contour_lines_from(file_path.to_str().unwrap());

        assert!(contour_lines[0].is_point_inside(&Point::new(94, 23)));

        assert!(contour_lines[0].is_point_inside(&Point::new(117, 29)));
        assert!(contour_lines[1].is_point_inside(&Point::new(117, 29)));

        assert!(contour_lines[0].is_point_inside(&Point::new(109, 55)));
        assert!(contour_lines[1].is_point_inside(&Point::new(109, 55)));
        assert!(contour_lines[2].is_point_inside(&Point::new(109, 55)));

        assert!(contour_lines[0].is_point_inside(&Point::new(151, 111)));
        assert!(contour_lines[1].is_point_inside(&Point::new(151, 111)));
        assert!(contour_lines[2].is_point_inside(&Point::new(151, 111)));
        assert!(contour_lines[3].is_point_inside(&Point::new(151, 111)));
    }

    #[test]
    fn extremum_points_outside_simple_contour_lines() {
        let file_path = Path::new(env!("CARGO_MANIFEST_DIR")).join("assets/contour_lines.png");
        let (contour_lines, _, _) = get_contour_lines_from(file_path.to_str().unwrap());

        assert!(!contour_lines[0].is_point_inside(&Point::new(8, 15)));
        assert!(!contour_lines[2].is_point_inside(&Point::new(63, 63)));
    }

    #[test]
    fn four_contours_lines_in_simple_contour_lines() {
        let file_path = Path::new(env!("CARGO_MANIFEST_DIR")).join("assets/contour_lines.png");
        let (contour_lines, _, _) = get_contour_lines_from(file_path.to_str().unwrap());

        assert_eq!(contour_lines.len(), 4);
    }
}
