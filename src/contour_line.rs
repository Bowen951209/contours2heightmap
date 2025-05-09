use std::u32;

use imageproc::{
    contours::{BorderType, Contour},
    point::Point,
};

pub struct ContourLine<T> {
    contour: Contour<T>,
    height: Option<i32>,
}

impl<T> ContourLine<T> {
    pub fn contour(&self) -> &Contour<T> {
        &self.contour
    }

    pub fn height(&self) -> Option<i32> {
        self.height
    }
}

pub struct Bbox {
    min: Point<u32>,
    max: Point<u32>,
}

impl Bbox {
    pub fn new() -> Self {
        Self {
            min: Point {
                x: u32::MAX,
                y: u32::MAX,
            },
            max: Point {
                x: u32::MIN,
                y: u32::MIN,
            },
        }
    }
}

pub fn retain_outer<T>(contours: &mut Vec<Contour<T>>) {
    contours.retain(|c| c.border_type == BorderType::Outer);
}

pub fn to_contour_lines<T>(contours: Vec<Contour<T>>) -> Vec<ContourLine<T>> {
    let mut contour_lines = Vec::new();
    for contour in contours {
        let contour_line = ContourLine {
            contour,
            height: None,
        };
        contour_lines.push(contour_line);
    }

    set_heights(&mut contour_lines);
    contour_lines
}

pub fn get_bbox(contour_lines: &[ContourLine<u32>]) -> Bbox {
    let mut bbox = Bbox::new();

    for cl in contour_lines {
        for p in &cl.contour.points {
            if p.x < bbox.min.x {
                bbox.min.x = p.x;
            }
            if p.x > bbox.max.x {
                bbox.max.x = p.x;
            }
            if p.y < bbox.min.y {
                bbox.min.y = p.y;
            }
            if p.y > bbox.max.y {
                bbox.max.y = p.y;
            }
        }
    }

    bbox
}

pub fn find_contour_line_interval<'a>(
    point: Point<u32>,
    contour_lines: &'a [ContourLine<u32>],
    bbox: &Bbox,
) -> (Option<&'a ContourLine<u32>>, Option<&'a ContourLine<u32>>) {
    let mut left_contour_line = None;
    println!("point = {:?}", point);
    'outer: for finder_x in (bbox.min.x..point.x).rev() {
        println!("finder_x = {}", finder_x);
        for cl in contour_lines {
            for p in &cl.contour.points {
                if *p == Point::new(finder_x, point.y) {
                    left_contour_line = Some(cl);
                    break 'outer;
                }
            }
        }
    }

    // Find right
    let mut right_contour_line = None;
    'outer: for finder_x in point.x..bbox.max.x {
        println!("finder_x = {}", finder_x);
        for cl in contour_lines {
            for p in &cl.contour.points {
                if *p == Point::new(finder_x, point.y) {
                    right_contour_line = Some(cl);
                    break 'outer;
                }
            }
        }
    }

    (left_contour_line, right_contour_line)
}

fn set_heights<T>(contour_lines: &mut [ContourLine<T>]) {
    let mut sorted: Vec<&mut ContourLine<T>> = contour_lines.iter_mut().collect();
    sorted.sort_by(|a, b| a.contour.parent.unwrap().cmp(&b.contour.parent.unwrap()));

    let mut height = 0;
    const GAP: i32 = 50;

    for contour_line in sorted {
        height += GAP;
        contour_line.height = Some(height);
    }
}
