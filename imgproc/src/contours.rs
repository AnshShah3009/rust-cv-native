use image::GrayImage;
use std::collections::{HashSet, VecDeque};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Contour {
    pub points: Vec<(i32, i32)>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ConnectedComponentStats {
    pub label: u32,
    pub area: u32,
    pub bbox: (u32, u32, u32, u32), // x, y, width, height
    pub centroid: (f64, f64),
}

const DIRS_8: [(i32, i32); 8] = [
    (1, 0),   // E
    (1, 1),   // SE
    (0, 1),   // S
    (-1, 1),  // SW
    (-1, 0),  // W
    (-1, -1), // NW
    (0, -1),  // N
    (1, -1),  // NE
];

fn in_bounds(x: i32, y: i32, w: i32, h: i32) -> bool {
    x >= 0 && y >= 0 && x < w && y < h
}

fn is_foreground(data: &[u8], w: i32, h: i32, x: i32, y: i32) -> bool {
    in_bounds(x, y, w, h) && data[(y * w + x) as usize] > 0
}

fn is_boundary(data: &[u8], w: i32, h: i32, x: i32, y: i32) -> bool {
    if !is_foreground(data, w, h, x, y) {
        return false;
    }
    for (dx, dy) in DIRS_8 {
        let nx = x + dx;
        let ny = y + dy;
        if !in_bounds(nx, ny, w, h) || !is_foreground(data, w, h, nx, ny) {
            return true;
        }
    }
    false
}

fn trace_boundary(data: &[u8], w: i32, h: i32, sx: i32, sy: i32) -> Vec<(i32, i32)> {
    let mut contour = Vec::new();
    let mut current = (sx, sy);
    let mut prev_dir = 4usize; // Start as if we came from W.
    let start = current;
    let start_prev_dir = prev_dir;
    let max_steps = (w as usize * h as usize).saturating_mul(8).max(32);

    for _ in 0..max_steps {
        contour.push(current);

        let mut found = None;
        for step in 1..=8 {
            let k = (prev_dir + step) % 8;
            let nx = current.0 + DIRS_8[k].0;
            let ny = current.1 + DIRS_8[k].1;
            if is_foreground(data, w, h, nx, ny) {
                // Backtrack direction for next step: neighbor before k in clockwise search.
                prev_dir = (k + 6) % 8;
                found = Some((nx, ny));
                break;
            }
        }

        let Some(next) = found else { break };

        if next == start && prev_dir == start_prev_dir && contour.len() > 1 {
            break;
        }
        current = next;
    }

    // Remove trivial duplicates if any.
    if contour.len() > 1 && contour.first() == contour.last() {
        contour.pop();
    }
    contour
}

/// Find external contours in a binary image (non-zero pixels are foreground).
pub fn find_external_contours(binary: &GrayImage) -> Vec<Contour> {
    let w = binary.width() as i32;
    let h = binary.height() as i32;
    let data = binary.as_raw();
    let mut visited_boundary = vec![false; (w * h) as usize];
    let mut contours = Vec::new();

    for y in 0..h {
        for x in 0..w {
            let idx = (y * w + x) as usize;
            if visited_boundary[idx] || !is_boundary(data, w, h, x, y) {
                continue;
            }
            let points = trace_boundary(data, w, h, x, y);
            if points.len() >= 3 {
                for &(px, py) in &points {
                    visited_boundary[(py * w + px) as usize] = true;
                }
                contours.push(Contour { points });
            } else {
                visited_boundary[idx] = true;
            }
        }
    }

    contours
}

/// Polygon area (shoelace). Contour should be ordered.
pub fn contour_area(contour: &Contour) -> f64 {
    let n = contour.points.len();
    if n < 3 {
        return 0.0;
    }
    let mut area = 0.0f64;
    for i in 0..n {
        let (x0, y0) = contour.points[i];
        let (x1, y1) = contour.points[(i + 1) % n];
        area += x0 as f64 * y1 as f64 - x1 as f64 * y0 as f64;
    }
    area.abs() * 0.5
}

/// Closed-contour perimeter.
pub fn contour_perimeter(contour: &Contour) -> f64 {
    let n = contour.points.len();
    if n < 2 {
        return 0.0;
    }
    let mut p = 0.0f64;
    for i in 0..n {
        let (x0, y0) = contour.points[i];
        let (x1, y1) = contour.points[(i + 1) % n];
        let dx = (x1 - x0) as f64;
        let dy = (y1 - y0) as f64;
        p += (dx * dx + dy * dy).sqrt();
    }
    p
}

/// Bounding rectangle as (x, y, width, height).
pub fn bounding_rect(contour: &Contour) -> Option<(i32, i32, u32, u32)> {
    if contour.points.is_empty() {
        return None;
    }
    let mut min_x = i32::MAX;
    let mut min_y = i32::MAX;
    let mut max_x = i32::MIN;
    let mut max_y = i32::MIN;
    for &(x, y) in &contour.points {
        min_x = min_x.min(x);
        min_y = min_y.min(y);
        max_x = max_x.max(x);
        max_y = max_y.max(y);
    }
    Some((
        min_x,
        min_y,
        (max_x - min_x + 1) as u32,
        (max_y - min_y + 1) as u32,
    ))
}

fn cross(o: (i32, i32), a: (i32, i32), b: (i32, i32)) -> i64 {
    let (ox, oy) = o;
    let (ax, ay) = a;
    let (bx, by) = b;
    (ax - ox) as i64 * (by - oy) as i64 - (ay - oy) as i64 * (bx - ox) as i64
}

/// Monotonic chain convex hull.
pub fn convex_hull(contour: &Contour) -> Contour {
    let mut pts: Vec<(i32, i32)> = {
        let mut set = HashSet::new();
        contour
            .points
            .iter()
            .copied()
            .filter(|p| set.insert(*p))
            .collect()
    };
    if pts.len() <= 2 {
        return Contour { points: pts };
    }
    pts.sort_unstable();

    let mut lower = Vec::new();
    for &p in &pts {
        while lower.len() >= 2 && cross(lower[lower.len() - 2], lower[lower.len() - 1], p) <= 0 {
            lower.pop();
        }
        lower.push(p);
    }

    let mut upper = Vec::new();
    for &p in pts.iter().rev() {
        while upper.len() >= 2 && cross(upper[upper.len() - 2], upper[upper.len() - 1], p) <= 0 {
            upper.pop();
        }
        upper.push(p);
    }

    lower.pop();
    upper.pop();
    lower.extend(upper);
    Contour { points: lower }
}

fn point_line_distance(p: (i32, i32), a: (i32, i32), b: (i32, i32)) -> f64 {
    let (px, py) = (p.0 as f64, p.1 as f64);
    let (ax, ay) = (a.0 as f64, a.1 as f64);
    let (bx, by) = (b.0 as f64, b.1 as f64);
    let dx = bx - ax;
    let dy = by - ay;
    if dx == 0.0 && dy == 0.0 {
        let ex = px - ax;
        let ey = py - ay;
        return (ex * ex + ey * ey).sqrt();
    }
    let t = ((px - ax) * dx + (py - ay) * dy) / (dx * dx + dy * dy);
    let proj_x = ax + t * dx;
    let proj_y = ay + t * dy;
    let ex = px - proj_x;
    let ey = py - proj_y;
    (ex * ex + ey * ey).sqrt()
}

fn rdp(points: &[(i32, i32)], epsilon: f64, out: &mut Vec<(i32, i32)>) {
    if points.len() < 2 {
        return;
    }
    let first = points[0];
    let last = points[points.len() - 1];
    let mut max_dist = 0.0f64;
    let mut idx = 0usize;
    for (i, &p) in points
        .iter()
        .enumerate()
        .skip(1)
        .take(points.len().saturating_sub(2))
    {
        let d = point_line_distance(p, first, last);
        if d > max_dist {
            max_dist = d;
            idx = i;
        }
    }

    if max_dist > epsilon && idx > 0 {
        rdp(&points[..=idx], epsilon, out);
        out.pop();
        rdp(&points[idx..], epsilon, out);
    } else {
        out.push(first);
        out.push(last);
    }
}

/// Douglas-Peucker contour approximation.
pub fn approx_poly_dp(contour: &Contour, epsilon: f64, closed: bool) -> Contour {
    let mut pts = contour.points.clone();
    if pts.len() < 3 {
        return Contour { points: pts };
    }

    if closed && pts.first() != pts.last() {
        pts.push(pts[0]);
    }

    let mut out = Vec::new();
    rdp(&pts, epsilon.max(0.0), &mut out);

    out.dedup();
    if closed && out.len() > 2 && out.first() == out.last() {
        out.pop();
    }
    Contour { points: out }
}

/// Connected components labeling with component statistics.
///
/// Returns `(labels, num_labels, stats)` where:
/// - `labels` is row-major label map (0 is background)
/// - `num_labels` includes background label 0
/// - `stats` only contains foreground components (labels 1..)
pub fn connected_components_with_stats(
    binary: &GrayImage,
    connectivity: u8,
) -> (Vec<u32>, u32, Vec<ConnectedComponentStats>) {
    let w = binary.width() as i32;
    let h = binary.height() as i32;
    let data = binary.as_raw();
    let mut labels = vec![0u32; (w * h) as usize];
    let mut stats = Vec::new();
    let mut next_label = 1u32;

    let neigh_4: &[(i32, i32)] = &[(1, 0), (-1, 0), (0, 1), (0, -1)];
    let neigh_8: &[(i32, i32)] = &[
        (1, 0),
        (-1, 0),
        (0, 1),
        (0, -1),
        (1, 1),
        (1, -1),
        (-1, 1),
        (-1, -1),
    ];
    let neigh = if connectivity == 4 { neigh_4 } else { neigh_8 };

    for y in 0..h {
        for x in 0..w {
            let idx = (y * w + x) as usize;
            if data[idx] == 0 || labels[idx] != 0 {
                continue;
            }

            let label = next_label;
            next_label += 1;

            let mut q = VecDeque::new();
            q.push_back((x, y));
            labels[idx] = label;

            let mut area = 0u32;
            let mut min_x = x;
            let mut min_y = y;
            let mut max_x = x;
            let mut max_y = y;
            let mut sum_x = 0f64;
            let mut sum_y = 0f64;

            while let Some((cx, cy)) = q.pop_front() {
                area += 1;
                min_x = min_x.min(cx);
                min_y = min_y.min(cy);
                max_x = max_x.max(cx);
                max_y = max_y.max(cy);
                sum_x += cx as f64;
                sum_y += cy as f64;

                for &(dx, dy) in neigh {
                    let nx = cx + dx;
                    let ny = cy + dy;
                    if !in_bounds(nx, ny, w, h) {
                        continue;
                    }
                    let nidx = (ny * w + nx) as usize;
                    if data[nidx] == 0 || labels[nidx] != 0 {
                        continue;
                    }
                    labels[nidx] = label;
                    q.push_back((nx, ny));
                }
            }

            let centroid = (sum_x / area as f64, sum_y / area as f64);
            let bbox = (
                min_x as u32,
                min_y as u32,
                (max_x - min_x + 1) as u32,
                (max_y - min_y + 1) as u32,
            );
            stats.push(ConnectedComponentStats {
                label,
                area,
                bbox,
                centroid,
            });
        }
    }

    (labels, next_label, stats)
}

#[cfg(test)]
mod tests {
    use super::*;
    use image::Luma;

    #[test]
    fn finds_rectangle_contour_and_descriptors() {
        let mut img = GrayImage::new(32, 24);
        for y in 6..18 {
            for x in 8..22 {
                img.put_pixel(x, y, Luma([255]));
            }
        }

        let contours = find_external_contours(&img);
        assert!(!contours.is_empty());
        let c = &contours[0];

        let rect = bounding_rect(c).unwrap();
        assert_eq!(rect.0, 8);
        assert_eq!(rect.1, 6);
        assert_eq!(rect.2, 14);
        assert_eq!(rect.3, 12);

        let a = contour_area(c);
        let p = contour_perimeter(c);
        assert!(a > 0.0);
        assert!(p > 0.0);
    }

    #[test]
    fn convex_hull_not_empty() {
        let c = Contour {
            points: vec![(0, 0), (2, 0), (1, 1), (2, 2), (0, 2), (1, 1)],
        };
        let h = convex_hull(&c);
        assert!(h.points.len() >= 3);
    }

    #[test]
    fn approx_poly_dp_reduces_points() {
        let c = Contour {
            points: vec![
                (0, 0),
                (1, 0),
                (2, 0),
                (3, 1),
                (3, 2),
                (3, 3),
                (2, 3),
                (1, 3),
                (0, 3),
                (0, 2),
                (0, 1),
            ],
        };
        let a = approx_poly_dp(&c, 0.8, true);
        assert!(a.points.len() < c.points.len());
        assert!(a.points.len() >= 4);
    }

    #[test]
    fn connected_components_stats_basic() {
        let mut img = GrayImage::new(16, 16);
        // Component 1: 3x3 block
        for y in 2..5 {
            for x in 3..6 {
                img.put_pixel(x, y, Luma([255]));
            }
        }
        // Component 2: 2x4 block
        for y in 10..14 {
            for x in 11..13 {
                img.put_pixel(x, y, Luma([255]));
            }
        }

        let (_labels, num_labels, stats) = connected_components_with_stats(&img, 8);
        assert_eq!(num_labels, 3); // background + 2 components
        assert_eq!(stats.len(), 2);
        let total_area: u32 = stats.iter().map(|s| s.area).sum();
        assert_eq!(total_area, 9 + 8);
    }
}
