use std::{path::Path, u32};

use imageproc::{
    contours::{BorderType, Contour},
    image::{self},
    point::Point,
};
use rstar::{AABB, Envelope, RTree, RTreeObject};

pub struct ContourLine {
    contour: Contour<usize>,
    height: Option<i32>,
    bbox: AABB<[i32; 2]>,
}

impl ContourLine {
    pub fn new(contour: Contour<usize>) -> Self {
        let bbox = AABB::from_points(
            &contour
                .points
                .iter()
                .map(|p| [p.x as i32, p.y as i32])
                .collect::<Vec<_>>(),
        );
        Self {
            contour,
            height: None,
            bbox,
        }
    }

    // Ray Casting Algorithm
    pub fn is_point_inside(&self, point: &Point<usize>) -> bool {
        let (px, py) = (point.x as f64, point.y as f64);
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

    pub fn contour(&self) -> &Contour<usize> {
        &self.contour
    }

    pub fn height(&self) -> Option<i32> {
        self.height
    }
}

impl RTreeObject for ContourLine {
    type Envelope = AABB<[i32; 2]>;

    fn envelope(&self) -> Self::Envelope {
        self.bbox
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
    outer: Option<&'a ContourLine>,
    inners: Vec<&'a ContourLine>,
}

impl<'a> ContourLineInterval<'a> {
    pub fn new(outer: &'a ContourLine, inner: &'a ContourLine) -> Self {
        Self {
            outer: Some(outer),
            inners: vec![inner],
        }
    }

    // TODO!("get rid of these and just use pub fields");
    pub fn outer(&self) -> Option<&ContourLine> {
        self.outer
    }

    pub fn inners(&self) -> &Vec<&ContourLine> {
        &self.inners
    }
}

pub fn get_contour_line_tree_from(
    file_path: impl AsRef<Path>,
    gap: i32,
) -> (RTree<ContourLine>, u32, u32) {
    // Load the image
    let dyn_img = image::open(file_path).expect("Failed to open image file");
    let (w, h) = (dyn_img.width(), dyn_img.height());

    // To grayscale, and then we can find contours
    let grayscale = dyn_img.to_luma8();
    let mut contours: Vec<Contour<usize>> = imageproc::contours::find_contours(&grayscale);

    // find_contours finds outer and inner contours. We only retain outers as representation
    retain_outer(&mut contours);

    // Convert Contours to ContourLines. The heights of each contour line are then set
    (to_contour_lines(contours, gap), w, h)
}

/// Find the contour line interval where `point` is located.
pub fn find_contour_line_interval(
    point: Point<usize>,
    contour_line_tree: &RTree<ContourLine>,
    gap: i32,
) -> ContourLineInterval<'_> {
    // Sort contour lines by height
    let mut sorted_contour_lines = contour_line_tree.iter().collect::<Vec<_>>();
    sorted_contour_lines.sort_by_key(|cl| cl.height().unwrap());

    // Identify the outer contour line
    let outer_contour_line = sorted_contour_lines
        .into_iter()
        .rev()
        .find(|cl| cl.is_point_inside(&point));

    // Collect inner contour lines
    let inner_contour_lines = outer_contour_line.map_or(vec![], |outer| {
        let target_height = outer.height().unwrap() + gap;
        let outer_envelope = outer.envelope();
        contour_line_tree
            .iter()
            .filter(|cl| cl.height().unwrap() == target_height)
            .filter(|cl| outer_envelope.contains_envelope(&cl.envelope()))
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

fn to_contour_lines(contours: Vec<Contour<usize>>, gap: i32) -> RTree<ContourLine> {
    let contour_lines = contours.into_iter().map(ContourLine::new).collect();
    let mut tree = RTree::bulk_load(contour_lines);

    set_heights(&mut tree, gap);
    tree
}

/// Use the passed in contour lines RTree to set the heights of each contour line from outer to inner.
/// The outmost contour line will have the lowest height, and the innermost contour line will have the highest height.
fn set_heights(tree: &mut RTree<ContourLine>, gap: i32) {
    let mut heights = Vec::with_capacity(tree.size());

    for cl in tree.iter() {
        let outside_count = tree
            .iter()
            .filter(|c| c.envelope().contains_envelope(&cl.envelope()))
            .count();
        let height = outside_count as i32 * gap;
        heights.push(height);
    }

    for (cl, height) in tree.iter_mut().zip(heights) {
        cl.height = Some(height);
    }
}

#[cfg(test)]
mod tests {
    use std::{env, path::Path, ptr};

    use crate::contour_line::ContourLine;

    use super::{find_contour_line_interval, get_contour_line_tree_from};
    use imageproc::point::Point;
    use rstar::Envelope;

    const GAP: i32 = 50;

    #[test]
    fn test_points_inside_one_hill() {
        let file_path = Path::new(env!("CARGO_MANIFEST_DIR")).join("assets/one_hill.png");
        let (tree, _, _) = get_contour_line_tree_from(file_path.to_str().unwrap(), GAP);
        let mut contour_lines = tree.iter().collect::<Vec<_>>();
        contour_lines.sort_by(|a, b| a.height().unwrap().cmp(&b.height().unwrap()));

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
    fn test_extremum_points_outside_one_hill() {
        let file_path = Path::new(env!("CARGO_MANIFEST_DIR")).join("assets/one_hill.png");
        let (tree, _, _) = get_contour_line_tree_from(file_path.to_str().unwrap(), GAP);
        let mut contour_lines = tree.iter().collect::<Vec<_>>();
        contour_lines.sort_by(|a, b| a.height().unwrap().cmp(&b.height().unwrap()));

        assert!(!contour_lines[0].is_point_inside(&Point::new(8, 15)));
        assert!(!contour_lines[2].is_point_inside(&Point::new(63, 63)));
    }

    #[test]
    fn test_four_contour_lines_in_one_hill() {
        let file_path = Path::new(env!("CARGO_MANIFEST_DIR")).join("assets/one_hill.png");
        let (contour_lines, _, _) = get_contour_line_tree_from(file_path.to_str().unwrap(), GAP);

        assert_eq!(contour_lines.size(), 4);
    }

    #[test]
    fn test_points_interval_two_hills() {
        let file_path = Path::new(env!("CARGO_MANIFEST_DIR")).join("assets/two_hills.png");
        let (tree, _, _) = get_contour_line_tree_from(file_path.to_str().unwrap(), GAP);

        let interval = find_contour_line_interval(Point::new(242, 184), &tree, GAP);
        assert_eq!(interval.outer.unwrap().height().unwrap(), GAP);
        assert_eq!(interval.inners[0].height().unwrap(), GAP * 2);

        let interval = find_contour_line_interval(Point::new(151, 135), &tree, GAP);
        assert_eq!(interval.outer.unwrap().height().unwrap(), GAP * 2);
        assert_eq!(interval.inners[0].height().unwrap(), GAP * 3);
    }

    #[test]
    fn test_232_117_two_inners_two_hills() {
        let file_path = Path::new(env!("CARGO_MANIFEST_DIR")).join("assets/two_hills.png");
        let (tree, _, _) = get_contour_line_tree_from(file_path.to_str().unwrap(), GAP);

        let interval = find_contour_line_interval(Point::new(232, 117), &tree, GAP);
        let inners = interval.inners();
        assert_eq!(inners.len(), 2);
    }

    #[test]
    fn test_151_119_interval_greater_area_two_hills() {
        let file_path = Path::new(env!("CARGO_MANIFEST_DIR")).join("assets/two_hills.png");
        let (tree, _, _) = get_contour_line_tree_from(file_path.to_str().unwrap(), GAP);

        let inner = find_contour_line_interval(Point::new(151, 119), &tree, GAP).inners[0];

        let another_gap_3: Vec<&ContourLine> = tree
            .iter()
            .filter(|cl| cl.height().unwrap() == GAP * 3 && !ptr::eq(*cl, inner))
            .collect();
        assert_eq!(another_gap_3.len(), 1);

        assert!(inner.bbox.area() > another_gap_3[0].bbox.area());
    }

    #[test]
    fn test_thirteen_contour_lines_in_two_hills() {
        let file_path = Path::new(env!("CARGO_MANIFEST_DIR")).join("assets/two_hills.png");
        let (contour_lines, _, _) = get_contour_line_tree_from(file_path.to_str().unwrap(), GAP);

        assert_eq!(contour_lines.size(), 13);
    }
}
