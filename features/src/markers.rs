use crate::{FeatureError, Result};
use image::{GrayImage, Luma};
use nalgebra::Point2;
use std::collections::VecDeque;

#[derive(Debug, Clone, Copy)]
pub enum ArucoDictionary {
    Dict4x4_50,
}

#[derive(Debug, Clone, Copy)]
pub struct ArucoDetection {
    pub id: u16,
    pub rotation_cw: u8,
    pub corners: [Point2<f32>; 4],
    pub center: Point2<f32>,
}

#[derive(Debug, Clone, Copy)]
pub enum AprilTagFamily {
    Tag16h5,
    Tag36h11,
}

#[derive(Debug, Clone, Copy)]
pub struct AprilTagDetection {
    pub id: u16,
    pub rotation_cw: u8,
    pub hamming: u8,
    pub corners: [Point2<f32>; 4],
    pub center: Point2<f32>,
}

pub fn draw_aruco_marker(
    dictionary: ArucoDictionary,
    id: u16,
    cell_size: u32,
) -> Result<GrayImage> {
    let payload_bits = 4usize;
    let border_bits = 1usize;
    let code = aruco_dictionary_codes(dictionary)
        .get(id as usize)
        .copied()
        .ok_or_else(|| FeatureError::DetectionError(format!("invalid aruco id: {}", id)))?;
    Ok(draw_marker_bits(code, payload_bits, border_bits, cell_size))
}

pub fn detect_aruco_markers(image: &GrayImage, dictionary: ArucoDictionary) -> Result<Vec<ArucoDetection>> {
    let payload_bits = 4usize;
    let border_bits = 1usize;
    let codes = aruco_dictionary_codes(dictionary);
    let candidates = find_marker_candidates(image, payload_bits + 2 * border_bits)?;
    let mut detections = Vec::new();

    for c in candidates {
        let bits = sample_grid_bits(image, c.min_x, c.min_y, c.max_x, c.max_y, payload_bits + 2 * border_bits);
        if !border_is_black(&bits, payload_bits + 2 * border_bits) {
            continue;
        }
        let payload = extract_payload(&bits, payload_bits, border_bits);
        let (id, rot, dist) = decode_code_with_rotations(&payload, &codes);
        if dist == 0 {
            detections.push(ArucoDetection {
                id: id as u16,
                rotation_cw: rot as u8,
                corners: candidate_corners(&c),
                center: candidate_center(&c),
            });
        }
    }

    Ok(detections)
}

pub fn draw_apriltag(family: AprilTagFamily, id: u16, cell_size: u32) -> Result<GrayImage> {
    let (payload_bits, codes) = apriltag_family_codes(family);
    let border_bits = 1usize;
    let code = codes
        .get(id as usize)
        .copied()
        .ok_or_else(|| FeatureError::DetectionError(format!("invalid apriltag id: {}", id)))?;
    Ok(draw_marker_bits(code, payload_bits, border_bits, cell_size))
}

pub fn detect_apriltags(image: &GrayImage, family: AprilTagFamily) -> Result<Vec<AprilTagDetection>> {
    let (payload_bits, codes) = apriltag_family_codes(family);
    let border_bits = 1usize;
    let candidates = find_marker_candidates(image, payload_bits + 2 * border_bits)?;
    let mut detections = Vec::new();

    for c in candidates {
        let bits = sample_grid_bits(
            image,
            c.min_x,
            c.min_y,
            c.max_x,
            c.max_y,
            payload_bits + 2 * border_bits,
        );
        if !border_is_black(&bits, payload_bits + 2 * border_bits) {
            continue;
        }
        let payload = extract_payload(&bits, payload_bits, border_bits);
        let (id, rot, dist) = decode_code_with_rotations(&payload, &codes);
        // Simple robust decode: allow up to one bit error for tag families.
        if dist <= 1 {
            detections.push(AprilTagDetection {
                id: id as u16,
                rotation_cw: rot as u8,
                hamming: dist as u8,
                corners: candidate_corners(&c),
                center: candidate_center(&c),
            });
        }
    }

    Ok(detections)
}

#[derive(Clone, Copy)]
struct Candidate {
    min_x: u32,
    min_y: u32,
    max_x: u32,
    max_y: u32,
}

fn draw_marker_bits(code: u64, payload_bits: usize, border_bits: usize, cell_size: u32) -> GrayImage {
    let grid = payload_bits + 2 * border_bits;
    let size = (grid as u32) * cell_size;
    let mut img = GrayImage::from_pixel(size, size, Luma([255]));

    for gy in 0..grid {
        for gx in 0..grid {
            let is_border = gy < border_bits
                || gx < border_bits
                || gy >= grid - border_bits
                || gx >= grid - border_bits;
            let is_black = if is_border {
                true
            } else {
                let py = gy - border_bits;
                let px = gx - border_bits;
                let bit_idx = py * payload_bits + px;
                ((code >> bit_idx) & 1) == 1
            };
            let val = if is_black { 0u8 } else { 255u8 };
            let x0 = gx as u32 * cell_size;
            let y0 = gy as u32 * cell_size;
            for y in y0..(y0 + cell_size) {
                for x in x0..(x0 + cell_size) {
                    img.put_pixel(x, y, Luma([val]));
                }
            }
        }
    }
    img
}

fn find_marker_candidates(image: &GrayImage, min_grid: usize) -> Result<Vec<Candidate>> {
    let w = image.width() as usize;
    let h = image.height() as usize;
    if w == 0 || h == 0 {
        return Err(FeatureError::DetectionError("empty image".to_string()));
    }
    let min_side = (min_grid as u32).max(6);
    let mut visited = vec![false; w * h];
    let mut out = Vec::new();
    let raw = image.as_raw();

    for y0 in 0..h {
        for x0 in 0..w {
            let idx0 = y0 * w + x0;
            if visited[idx0] || raw[idx0] > 80 {
                continue;
            }
            let mut q = VecDeque::new();
            q.push_back((x0 as i32, y0 as i32));
            visited[idx0] = true;

            let mut count = 0usize;
            let mut min_x = x0 as u32;
            let mut min_y = y0 as u32;
            let mut max_x = x0 as u32;
            let mut max_y = y0 as u32;

            while let Some((x, y)) = q.pop_front() {
                count += 1;
                let ux = x as u32;
                let uy = y as u32;
                min_x = min_x.min(ux);
                min_y = min_y.min(uy);
                max_x = max_x.max(ux);
                max_y = max_y.max(uy);

                for (nx, ny) in [(x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1)] {
                    if nx < 0 || ny < 0 || nx >= w as i32 || ny >= h as i32 {
                        continue;
                    }
                    let nux = nx as usize;
                    let nuy = ny as usize;
                    let nidx = nuy * w + nux;
                    if visited[nidx] || raw[nidx] > 80 {
                        continue;
                    }
                    visited[nidx] = true;
                    q.push_back((nx, ny));
                }
            }

            let bw = max_x - min_x + 1;
            let bh = max_y - min_y + 1;
            if bw < min_side || bh < min_side {
                continue;
            }
            let ratio = bw as f32 / bh as f32;
            if !(0.7..=1.3).contains(&ratio) {
                continue;
            }
            let box_area = (bw as usize) * (bh as usize);
            let fill = count as f32 / box_area.max(1) as f32;
            if fill < 0.18 || fill > 0.95 {
                continue;
            }

            out.push(Candidate {
                min_x,
                min_y,
                max_x,
                max_y,
            });
        }
    }

    Ok(out)
}

fn sample_grid_bits(
    image: &GrayImage,
    min_x: u32,
    min_y: u32,
    max_x: u32,
    max_y: u32,
    grid: usize,
) -> Vec<u8> {
    let mut bits = vec![0u8; grid * grid];
    let bw = (max_x - min_x + 1) as f32;
    let bh = (max_y - min_y + 1) as f32;

    for gy in 0..grid {
        for gx in 0..grid {
            let x0 = min_x as f32 + (gx as f32) * bw / grid as f32;
            let x1 = min_x as f32 + ((gx + 1) as f32) * bw / grid as f32;
            let y0 = min_y as f32 + (gy as f32) * bh / grid as f32;
            let y1 = min_y as f32 + ((gy + 1) as f32) * bh / grid as f32;
            let mut black = 0usize;
            let mut total = 0usize;
            let sx0 = x0.floor().clamp(0.0, (image.width() - 1) as f32) as u32;
            let sx1 = x1.ceil().clamp(0.0, image.width() as f32) as u32;
            let sy0 = y0.floor().clamp(0.0, (image.height() - 1) as f32) as u32;
            let sy1 = y1.ceil().clamp(0.0, image.height() as f32) as u32;
            for y in sy0..sy1 {
                for x in sx0..sx1 {
                    total += 1;
                    if image.get_pixel(x, y)[0] < 128 {
                        black += 1;
                    }
                }
            }
            bits[gy * grid + gx] = if black * 2 >= total.max(1) { 1 } else { 0 };
        }
    }
    bits
}

fn border_is_black(bits: &[u8], grid: usize) -> bool {
    for i in 0..grid {
        if bits[i] == 0 || bits[(grid - 1) * grid + i] == 0 {
            return false;
        }
        if bits[i * grid] == 0 || bits[i * grid + (grid - 1)] == 0 {
            return false;
        }
    }
    true
}

fn extract_payload(bits: &[u8], payload_bits: usize, border_bits: usize) -> u64 {
    let grid = payload_bits + 2 * border_bits;
    let mut code = 0u64;
    for y in 0..payload_bits {
        for x in 0..payload_bits {
            let v = bits[(y + border_bits) * grid + (x + border_bits)];
            if v != 0 {
                let idx = y * payload_bits + x;
                code |= 1u64 << idx;
            }
        }
    }
    code
}

fn decode_code_with_rotations(code: &u64, dict: &[u64]) -> (usize, usize, u32) {
    let mut best_id = 0usize;
    let mut best_rot = 0usize;
    let mut best_dist = u32::MAX;
    let n = ((dict[0].leading_zeros() ^ 63) as usize + 1).max(16);
    let side = (n as f32).sqrt().round() as usize;
    let rots = rotate_code_variants(*code, side);
    for (id, dcode) in dict.iter().enumerate() {
        for (rot, rc) in rots.iter().enumerate() {
            let dist = (rc ^ dcode).count_ones();
            if dist < best_dist {
                best_dist = dist;
                best_id = id;
                best_rot = rot;
            }
        }
    }
    (best_id, best_rot, best_dist)
}

fn rotate_code_variants(code: u64, side: usize) -> [u64; 4] {
    let mut out = [0u64; 4];
    out[0] = code;
    for i in 1..4 {
        out[i] = rotate_code_90(out[i - 1], side);
    }
    out
}

fn rotate_code_90(code: u64, side: usize) -> u64 {
    let mut out = 0u64;
    for y in 0..side {
        for x in 0..side {
            let bit = (code >> (y * side + x)) & 1;
            if bit != 0 {
                let nx = side - 1 - y;
                let ny = x;
                out |= 1u64 << (ny * side + nx);
            }
        }
    }
    out
}

fn aruco_dictionary_codes(dict: ArucoDictionary) -> Vec<u64> {
    match dict {
        ArucoDictionary::Dict4x4_50 => generate_dictionary_codes(16, 50, 0xA53A_9E37_5D1Cu64),
    }
}

fn apriltag_family_codes(family: AprilTagFamily) -> (usize, Vec<u64>) {
    match family {
        AprilTagFamily::Tag16h5 => (4, generate_dictionary_codes(16, 30, 0x9E37_79B9_7F4Au64)),
        AprilTagFamily::Tag36h11 => (6, generate_dictionary_codes(36, 100, 0xC2B2_AE35_1D0Bu64)),
    }
}

fn generate_dictionary_codes(bits: usize, count: usize, seed: u64) -> Vec<u64> {
    let mut out = Vec::with_capacity(count);
    let mut state = seed;
    let mask = if bits == 64 { u64::MAX } else { (1u64 << bits) - 1 };
    while out.len() < count {
        state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
        let mut code = state & mask;
        // Avoid trivial patterns.
        if code == 0 || code == mask {
            continue;
        }
        // Ensure mixed density.
        let ones = code.count_ones() as usize;
        if ones < bits / 4 || ones > bits * 3 / 4 {
            continue;
        }
        // Canonicalize rotation-equivalent payloads.
        let side = (bits as f32).sqrt().round() as usize;
        let rots = rotate_code_variants(code, side);
        let canonical = *rots.iter().min().unwrap_or(&code);
        code = canonical;
        if out.contains(&code) {
            continue;
        }
        out.push(code);
    }
    out
}

fn candidate_corners(c: &Candidate) -> [Point2<f32>; 4] {
    [
        Point2::new(c.min_x as f32, c.min_y as f32),
        Point2::new(c.max_x as f32, c.min_y as f32),
        Point2::new(c.max_x as f32, c.max_y as f32),
        Point2::new(c.min_x as f32, c.max_y as f32),
    ]
}

fn candidate_center(c: &Candidate) -> Point2<f32> {
    Point2::new(
        (c.min_x + c.max_x) as f32 * 0.5,
        (c.min_y + c.max_y) as f32 * 0.5,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use image::imageops::replace;

    #[test]
    fn aruco_draw_and_detect_single_marker() {
        let marker = draw_aruco_marker(ArucoDictionary::Dict4x4_50, 7, 10).unwrap();
        let mut canvas = GrayImage::from_pixel(180, 140, Luma([255]));
        replace(&mut canvas, &marker, 30, 20);

        let det = detect_aruco_markers(&canvas, ArucoDictionary::Dict4x4_50).unwrap();
        assert!(!det.is_empty());
        assert!(det.iter().any(|d| d.id == 7));
    }

    #[test]
    fn apriltag_draw_and_detect_single_marker() {
        let marker = draw_apriltag(AprilTagFamily::Tag16h5, 3, 8).unwrap();
        let mut canvas = GrayImage::from_pixel(160, 160, Luma([255]));
        replace(&mut canvas, &marker, 40, 50);

        let det = detect_apriltags(&canvas, AprilTagFamily::Tag16h5).unwrap();
        assert!(!det.is_empty());
        assert!(det.iter().any(|d| d.id == 3));
    }
}
