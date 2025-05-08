use imageproc::contours::{BorderType, Contour};

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
