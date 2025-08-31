use std::path::Path;

use imageproc::{
    contours::{BorderType, Contour},
    image::{self},
};
use ordered_float::OrderedFloat;

pub struct ContourLine {
    pub contour: Contour<u32>,
    pub height: f64,
}

impl ContourLine {
    fn new(contour: Contour<u32>) -> Self {
        Self {
            contour,
            height: 0.0,
        }
    }

    // Ray Casting Algorithm
    pub fn is_point_inside(&self, x: u32, y: u32) -> bool {
        let (px, py) = (x as f64, y as f64);
        let mut inside = false;
        let points = &self.contour.points;
        let n = points.len();

        let mut j = n - 1;
        for i in 0..n {
            let (xi, yi) = (points[i].x as f64, points[i].y as f64);
            let (xj, yj) = (points[j].x as f64, points[j].y as f64);

            // Check if point is on a horizontal ray from the test point
            if ((yi > py) != (yj > py)) && (px < (xj - xi) * (py - yi) / (yj - yi) + xi) {
                inside = !inside;
            }
            j = i;
        }

        inside
    }
}

/// Indicates the inner and outer contour lines for a pixel.
///
/// For a given pixel, there may be no outer contour line (the pixel is outside all contours),
/// or there may be no inner contour lines (the pixel is in the innermost region).
/// Therefore, `outer` is an [`Option`], and since there may be multiple inner contour lines,
/// `inners` is a [`Vec`]. Set it to empty if there are no inner contours.
#[derive(Clone)]
pub struct ContourLineInterval<'a> {
    pub outer: Option<&'a ContourLine>,
    pub inners: Vec<&'a ContourLine>,
}

impl<'a> ContourLineInterval<'a> {
    pub fn new(outer: &'a ContourLine, inner: &'a ContourLine) -> Self {
        Self {
            outer: Some(outer),
            inners: vec![inner],
        }
    }
}

pub fn get_contour_lines_from(
    file_path: impl AsRef<Path>,
    gap: f64,
) -> (Vec<ContourLine>, u32, u32) {
    // Load the image
    let dyn_img = image::open(file_path).expect("Failed to open image file");
    let (w, h) = (dyn_img.width(), dyn_img.height());

    // To grayscale, and then we can find contours
    let grayscale = dyn_img.into_luma8();
    let mut contours = imageproc::contours::find_contours_with_threshold(&grayscale, 64);

    // find_contours finds outer and inner contours. We only retain outers as representation
    retain_outer(&mut contours);

    // Convert Contours to ContourLines. The heights of each contour line are then set
    (to_contour_lines(contours, gap), w, h)
}

/// Find the contour line interval where `point` is located.
pub fn find_contour_line_interval(
    x: u32,
    y: u32,
    sorted_contour_lines: &Vec<ContourLine>,
    gap: f64,
) -> ContourLineInterval<'_> {
    debug_assert!(sorted_contour_lines.is_sorted_by_key(|cl| OrderedFloat(cl.height)));

    // Identify the outer contour line
    let outer_contour_line = sorted_contour_lines
        .into_iter()
        .rev()
        .find(|cl| cl.is_point_inside(x, y));

    // Collect inner contour lines
    let inner_contour_lines = outer_contour_line.map_or(vec![], |outer| {
        let target_height = outer.height + gap;
        sorted_contour_lines
            .iter()
            .filter(|cl| cl.height == target_height)
            .filter(|cl| {
                let point = cl.contour.points[0];
                outer.is_point_inside(point.x, point.y)
            })
            .collect()
    });

    ContourLineInterval {
        outer: outer_contour_line,
        inners: inner_contour_lines,
    }
}

fn retain_outer<T>(contours: &mut Vec<Contour<T>>) {
    contours.retain(|c| c.border_type == BorderType::Outer);
}

fn to_contour_lines(contours: Vec<Contour<u32>>, gap: f64) -> Vec<ContourLine> {
    let mut contour_lines = contours.into_iter().map(ContourLine::new).collect();
    set_heights(&mut contour_lines, gap);
    contour_lines
}

/// Use the passed in contour lines to set the heights of each contour line from outer to inner.
/// The outmost contour line will have the lowest height, and the innermost contour line will have the highest height.
/// This function also sorts `contour_lines` by the height.
fn set_heights(contour_lines: &mut Vec<ContourLine>, gap: f64) {
    unsafe {
        let ptr = contour_lines.as_ptr();
        let len = contour_lines.len();

        for (i, cl) in contour_lines.iter_mut().enumerate() {
            let this_point = cl.contour.points[0];
            let mut count_plus_1 = 1; // we want the first contour line to have height `gap`
            for j in 0..len {
                if j == i {
                    continue; // this make sure it's safe
                }

                let other = &*ptr.add(j);
                if other.is_point_inside(this_point.x, this_point.y) {
                    count_plus_1 += 1;
                }
            }
            cl.height = gap * count_plus_1 as f64;
        }
    }

    contour_lines.sort_by_key(|cl| OrderedFloat(cl.height));
}

#[cfg(test)]
mod tests {
    use super::{find_contour_line_interval, get_contour_lines_from};
    use ordered_float::OrderedFloat;

    const GAP: f64 = 50.0;

    #[test]
    fn test_points_inside_one_hill() {
        let file_path = "test_assets/one_hill.jpg";
        let (contour_lines, _, _) = get_contour_lines_from(file_path, GAP);
        let mut contour_lines = contour_lines.iter().collect::<Vec<_>>();
        contour_lines.sort_by_key(|cl| OrderedFloat(cl.height));

        assert!(contour_lines[0].is_point_inside(94, 23));

        assert!(contour_lines[0].is_point_inside(117, 29));
        assert!(contour_lines[1].is_point_inside(117, 29));

        assert!(contour_lines[0].is_point_inside(109, 55));
        assert!(contour_lines[1].is_point_inside(109, 55));
        assert!(contour_lines[2].is_point_inside(109, 55));

        assert!(contour_lines[0].is_point_inside(151, 111));
        assert!(contour_lines[1].is_point_inside(151, 111));
        assert!(contour_lines[2].is_point_inside(151, 111));
        assert!(contour_lines[3].is_point_inside(151, 111));
    }

    #[test]
    fn test_extremum_points_outside_one_hill() {
        let file_path = "test_assets/one_hill.jpg";
        let (contour_lines, _, _) = get_contour_lines_from(file_path, GAP);
        let mut contour_lines = contour_lines.iter().collect::<Vec<_>>();
        contour_lines.sort_by_key(|cl| OrderedFloat(cl.height));

        assert!(!contour_lines[0].is_point_inside(8, 15));
        assert!(!contour_lines[2].is_point_inside(63, 63));
    }

    #[test]
    fn test_four_contour_lines_in_one_hill() {
        let file_path = "test_assets/one_hill.jpg";
        let (contour_lines, _, _) = get_contour_lines_from(file_path, GAP);

        assert_eq!(contour_lines.len(), 4);
    }

    #[test]
    fn test_points_interval_two_hills() {
        let file_path = "test_assets/two_hills.jpg";
        let (contour_lines, _, _) = get_contour_lines_from(file_path, GAP);

        let interval = find_contour_line_interval(242, 184, &contour_lines, GAP);
        assert_eq!(interval.outer.unwrap().height, GAP);
        assert_eq!(interval.inners[0].height, GAP * 2.0);

        let interval = find_contour_line_interval(151, 135, &contour_lines, GAP);
        assert_eq!(interval.outer.unwrap().height, GAP * 2.0);
        assert_eq!(interval.inners[0].height, GAP * 3.0);
    }

    #[test]
    fn test_232_117_two_inners_two_hills() {
        let file_path = "test_assets/two_hills.jpg";
        let (contour_lines, _, _) = get_contour_lines_from(file_path, GAP);

        let interval = find_contour_line_interval(232, 117, &contour_lines, GAP);
        let inners = interval.inners;
        assert_eq!(inners.len(), 2);
    }

    #[test]
    fn test_thirteen_contour_lines_in_two_hills() {
        let file_path = "test_assets/two_hills.jpg";
        let (contour_lines, _, _) = get_contour_lines_from(file_path, GAP);

        assert_eq!(contour_lines.len(), 13);
    }

    #[test]
    fn test_concave_height_on_contour() {
        let file_path = "test_assets/concave.jpg";
        let (contour_lines, _, _) = get_contour_lines_from(file_path, GAP);

        assert_eq!(contour_lines[0].height, GAP);
        assert_eq!(contour_lines[1].height, GAP);
    }
}
