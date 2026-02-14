//! Object tracking algorithms
//!
//! Various tracking methods for following objects in video sequences

use crate::Result;
use cv_core::KeyPoint;
use image::GrayImage;

/// Tracker interface
pub trait Tracker {
    fn init(&mut self, frame: &GrayImage, bbox: (u32, u32, u32, u32)) -> Result<()>;
    fn update(&mut self, frame: &GrayImage) -> Result<(u32, u32, u32, u32)>;
}

/// Simple template matching tracker
pub struct TemplateTracker {
    template: Option<GrayImage>,
    last_position: Option<(u32, u32)>,
    search_radius: u32,
}

impl TemplateTracker {
    pub fn new(search_radius: u32) -> Self {
        Self {
            template: None,
            last_position: None,
            search_radius,
        }
    }

    fn extract_template(&self, frame: &GrayImage, x: u32, y: u32, w: u32, h: u32) -> GrayImage {
        let mut template = GrayImage::new(w, h);

        for dy in 0..h {
            for dx in 0..w {
                let px = (x + dx).min(frame.width() - 1);
                let py = (y + dy).min(frame.height() - 1);
                let val = frame.get_pixel(px, py)[0];
                template.put_pixel(dx, dy, image::Luma([val]));
            }
        }

        template
    }

    fn find_best_match(&self, frame: &GrayImage) -> Option<(u32, u32)> {
        let template = self.template.as_ref()?;
        let last_pos = self.last_position?;

        let (tw, th) = (template.width(), template.height());
        let mut best_pos = last_pos;
        let mut best_score = f32::INFINITY;

        // Search in neighborhood
        let search_min_x = last_pos.0.saturating_sub(self.search_radius);
        let search_max_x = (last_pos.0 + self.search_radius).min(frame.width() - tw);
        let search_min_y = last_pos.1.saturating_sub(self.search_radius);
        let search_max_y = (last_pos.1 + self.search_radius).min(frame.height() - th);

        for y in search_min_y..=search_max_y {
            for x in search_min_x..=search_max_x {
                let score = self.compute_match_score(frame, template, x, y);
                if score < best_score {
                    best_score = score;
                    best_pos = (x, y);
                }
            }
        }

        Some(best_pos)
    }

    fn compute_match_score(&self, frame: &GrayImage, template: &GrayImage, x: u32, y: u32) -> f32 {
        let mut sum_squared_diff = 0.0f32;
        let mut count = 0;

        for ty in 0..template.height() {
            for tx in 0..template.width() {
                let fx = (x + tx).min(frame.width() - 1);
                let fy = (y + ty).min(frame.height() - 1);

                let frame_val = frame.get_pixel(fx, fy)[0] as f32;
                let template_val = template.get_pixel(tx, ty)[0] as f32;

                let diff = frame_val - template_val;
                sum_squared_diff += diff * diff;
                count += 1;
            }
        }

        if count > 0 {
            sum_squared_diff / count as f32
        } else {
            f32::INFINITY
        }
    }
}

impl Tracker for TemplateTracker {
    fn init(&mut self, frame: &GrayImage, bbox: (u32, u32, u32, u32)) -> Result<()> {
        let (x, y, w, h) = bbox;
        self.template = Some(self.extract_template(frame, x, y, w, h));
        self.last_position = Some((x, y));
        Ok(())
    }

    fn update(&mut self, frame: &GrayImage) -> Result<(u32, u32, u32, u32)> {
        if let Some(new_pos) = self.find_best_match(frame) {
            self.last_position = Some(new_pos);

            if let Some(ref template) = self.template {
                return Ok((new_pos.0, new_pos.1, template.width(), template.height()));
            }
        }

        Err(crate::VideoError::TrackingError(
            "Failed to track object".to_string(),
        ))
    }
}

/// Mean-shift tracker
pub struct MeanShiftTracker {
    target_model: Option<Vec<f32>>,
    last_position: Option<(f64, f64)>,
    window_size: (u32, u32),
    max_iterations: usize,
    epsilon: f64,
}

impl MeanShiftTracker {
    pub fn new(window_width: u32, window_height: u32) -> Self {
        Self {
            target_model: None,
            last_position: None,
            window_size: (window_width, window_height),
            max_iterations: 10,
            epsilon: 0.1,
        }
    }

    fn compute_color_histogram(&self, frame: &GrayImage, cx: f64, cy: f64) -> Vec<f32> {
        let mut histogram = vec![0.0f32; 256];
        let (half_w, half_h) = (
            self.window_size.0 as f64 / 2.0,
            self.window_size.1 as f64 / 2.0,
        );

        let min_x = (cx - half_w).max(0.0) as u32;
        let max_x = (cx + half_w).min(frame.width() as f64 - 1.0) as u32;
        let min_y = (cy - half_h).max(0.0) as u32;
        let max_y = (cy + half_h).min(frame.height() as f64 - 1.0) as u32;

        let mut total_weight = 0.0;

        for y in min_y..=max_y {
            for x in min_x..=max_x {
                // Epanechnikov kernel weight
                let dx = (x as f64 - cx) / half_w;
                let dy = (y as f64 - cy) / half_h;
                let dist_sq = dx * dx + dy * dy;

                if dist_sq < 1.0 {
                    let weight = 1.0 - dist_sq;
                    let intensity = frame.get_pixel(x, y)[0] as usize;
                    histogram[intensity] += weight as f32;
                    total_weight += weight;
                }
            }
        }

        // Normalize
        if total_weight > 0.0 {
            for val in &mut histogram {
                *val /= total_weight as f32;
            }
        }

        histogram
    }

    fn compute_mean_shift(&self, frame: &GrayImage, cx: f64, cy: f64) -> (f64, f64) {
        let target = self.target_model.as_ref().unwrap();
        let (half_w, half_h) = (
            self.window_size.0 as f64 / 2.0,
            self.window_size.1 as f64 / 2.0,
        );

        let mut numerator_x = 0.0;
        let mut numerator_y = 0.0;
        let mut denominator = 0.0;

        let min_x = (cx - half_w).max(0.0) as u32;
        let max_x = (cx + half_w).min(frame.width() as f64 - 1.0) as u32;
        let min_y = (cy - half_h).max(0.0) as u32;
        let max_y = (cy + half_h).min(frame.height() as f64 - 1.0) as u32;

        for y in min_y..=max_y {
            for x in min_x..=max_x {
                let dx = (x as f64 - cx) / half_w;
                let dy = (y as f64 - cy) / half_h;
                let dist_sq = dx * dx + dy * dy;

                if dist_sq < 1.0 {
                    let intensity = frame.get_pixel(x, y)[0] as usize;
                    let weight = (target[intensity] / (target[intensity] + 1e-6)).sqrt();

                    numerator_x += x as f64 * weight as f64;
                    numerator_y += y as f64 * weight as f64;
                    denominator += weight as f64;
                }
            }
        }

        if denominator > 0.0 {
            (numerator_x / denominator, numerator_y / denominator)
        } else {
            (cx, cy)
        }
    }
}

impl Tracker for MeanShiftTracker {
    fn init(&mut self, frame: &GrayImage, bbox: (u32, u32, u32, u32)) -> Result<()> {
        let (x, y, w, h) = bbox;
        self.window_size = (w, h);
        let cx = x as f64 + w as f64 / 2.0;
        let cy = y as f64 + h as f64 / 2.0;

        self.target_model = Some(self.compute_color_histogram(frame, cx, cy));
        self.last_position = Some((cx, cy));

        Ok(())
    }

    fn update(&mut self, frame: &GrayImage) -> Result<(u32, u32, u32, u32)> {
        let (mut cx, mut cy) = self.last_position.ok_or_else(|| {
            crate::VideoError::TrackingError("Tracker not initialized".to_string())
        })?;

        for _ in 0..self.max_iterations {
            let (new_cx, new_cy) = self.compute_mean_shift(frame, cx, cy);

            let dist = ((new_cx - cx).powi(2) + (new_cy - cy).powi(2)).sqrt();

            cx = new_cx;
            cy = new_cy;

            if dist < self.epsilon {
                break;
            }
        }

        self.last_position = Some((cx, cy));

        let x = (cx - self.window_size.0 as f64 / 2.0) as u32;
        let y = (cy - self.window_size.1 as f64 / 2.0) as u32;

        Ok((x, y, self.window_size.0, self.window_size.1))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use image::Luma;

    fn create_test_sequence() -> Vec<GrayImage> {
        let mut frames = Vec::new();
        let width = 100u32;
        let height = 100u32;

        for frame_idx in 0..5 {
            let mut frame = GrayImage::new(width, height);

            // Fill with black
            for y in 0..height {
                for x in 0..width {
                    frame.put_pixel(x, y, Luma([0]));
                }
            }

            // Draw moving square
            let square_x = 30 + frame_idx * 5;
            let square_y = 40;
            let square_size = 20;

            for y in 0..square_size {
                for x in 0..square_size {
                    let px = (square_x + x).min(width - 1);
                    let py = (square_y + y).min(height - 1);
                    frame.put_pixel(px, py, Luma([255]));
                }
            }

            frames.push(frame);
        }

        frames
    }

    #[test]
    fn test_template_tracker() {
        let frames = create_test_sequence();
        let mut tracker = TemplateTracker::new(15);

        // Initialize with first frame
        tracker.init(&frames[0], (30, 40, 20, 20)).unwrap();

        // Track through sequence
        for (i, frame) in frames.iter().enumerate().skip(1) {
            let bbox = tracker.update(frame).unwrap();
            println!("Frame {}: bbox at ({}, {})", i, bbox.0, bbox.1);

            // Square should move right
            assert!(bbox.0 >= 30 + (i as u32 - 1) * 5);
        }
    }

    #[test]
    fn test_mean_shift_tracker() {
        let frames = create_test_sequence();
        let mut tracker = MeanShiftTracker::new(20, 20);

        // Initialize with first frame
        tracker.init(&frames[0], (30, 40, 20, 20)).unwrap();

        // Track through sequence
        for (i, frame) in frames.iter().enumerate().skip(1) {
            let bbox = tracker.update(frame).unwrap();
            println!("Frame {}: MeanShift bbox at ({}, {})", i, bbox.0, bbox.1);
        }
    }
}
