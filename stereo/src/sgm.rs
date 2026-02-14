//! Semi-Global Matching (SGM) Stereo Algorithm
//!
//! More accurate than block matching by enforcing smoothness constraints
//! along multiple directions. Based on Hirschmuller's SGM algorithm.

use crate::{DisparityMap, Result, StereoError};
use image::GrayImage;

/// SGM stereo matcher
pub struct SgmMatcher {
    pub min_disparity: i32,
    pub max_disparity: i32,
    pub p1: u32,                // Penalty for disparity change of 1
    pub p2: u32,                // Penalty for larger disparity changes
    pub paths: Vec<(i32, i32)>, // Aggregation directions
}

impl Default for SgmMatcher {
    fn default() -> Self {
        // 8 directions: horizontal, vertical, and 4 diagonals
        let paths = vec![
            (1, 0),   // Right
            (-1, 0),  // Left
            (0, 1),   // Down
            (0, -1),  // Up
            (1, 1),   // Down-right
            (-1, 1),  // Down-left
            (1, -1),  // Up-right
            (-1, -1), // Up-left
        ];

        Self {
            min_disparity: 0,
            max_disparity: 64,
            p1: 10,
            p2: 120,
            paths,
        }
    }
}

impl SgmMatcher {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_disparity_range(mut self, min: i32, max: i32) -> Self {
        self.min_disparity = min;
        self.max_disparity = max;
        self
    }

    pub fn with_penalties(mut self, p1: u32, p2: u32) -> Self {
        self.p1 = p1;
        self.p2 = p2;
        self
    }

    pub fn compute(&self, left: &GrayImage, right: &GrayImage) -> Result<DisparityMap> {
        if left.width() != right.width() || left.height() != right.height() {
            return Err(StereoError::SizeMismatch(
                "Left and right images must have the same dimensions".to_string(),
            ));
        }

        let width = left.width() as usize;
        let height = left.height() as usize;
        let num_disparities = (self.max_disparity - self.min_disparity + 1) as usize;

        // Step 1: Compute pixel-wise matching costs (Census or SAD)
        let cost_volume = self.compute_matching_costs(left, right);

        // Step 2: Aggregate costs along multiple paths
        let aggregated_costs = self.aggregate_costs(&cost_volume, width, height, num_disparities);

        // Step 3: Winner-take-all to select disparities
        let mut disparity = DisparityMap::new(
            left.width(),
            left.height(),
            self.min_disparity,
            self.max_disparity,
        );

        for y in 0..height {
            for x in 0..width {
                let best_d =
                    self.find_best_disparity(&aggregated_costs, x, y, width, num_disparities);
                disparity.set(x as u32, y as u32, best_d as f32);
            }
        }

        Ok(disparity)
    }

    fn compute_matching_costs(&self, left: &GrayImage, right: &GrayImage) -> Vec<u32> {
        let width = left.width() as usize;
        let height = left.height() as usize;
        let num_disparities = (self.max_disparity - self.min_disparity + 1) as usize;

        let mut costs = vec![0u32; width * height * num_disparities];

        // Use simple SAD (Sum of Absolute Differences) over small window
        let window_size = 3;
        let half_window = window_size / 2;

        for y in half_window..height - half_window {
            for x in half_window..width - half_window {
                for (d_idx, d) in (self.min_disparity..=self.max_disparity).enumerate() {
                    let mut cost = 0u32;

                    for dy in -(half_window as i32)..=half_window as i32 {
                        for dx in -(half_window as i32)..=half_window as i32 {
                            let lx = (x as i32 + dx) as u32;
                            let ly = (y as i32 + dy) as u32;
                            let rx = (x as i32 + dx - d) as u32;
                            let ry = (y as i32 + dy) as u32;

                            if rx < right.width() && ry < right.height() {
                                let left_val = left.get_pixel(lx, ly)[0] as i32;
                                let right_val = right.get_pixel(rx, ry)[0] as i32;
                                cost += (left_val - right_val).abs() as u32;
                            }
                        }
                    }

                    let idx = (y * width + x) * num_disparities + d_idx;
                    costs[idx] = cost;
                }
            }
        }

        costs
    }

    fn aggregate_costs(
        &self,
        cost_volume: &[u32],
        width: usize,
        height: usize,
        num_disparities: usize,
    ) -> Vec<u32> {
        let mut aggregated = vec![0u32; width * height * num_disparities];

        // For each aggregation direction
        for &(dx, dy) in &self.paths {
            let path_costs =
                self.aggregate_along_path(cost_volume, width, height, num_disparities, dx, dy);

            // Sum costs from all directions
            for i in 0..aggregated.len() {
                aggregated[i] += path_costs[i];
            }
        }

        aggregated
    }

    fn aggregate_along_path(
        &self,
        cost_volume: &[u32],
        width: usize,
        height: usize,
        num_disparities: usize,
        dx: i32,
        dy: i32,
    ) -> Vec<u32> {
        let mut path_costs = vec![0u32; width * height * num_disparities];

        // Determine scan direction
        let (x_range, y_range): (Vec<usize>, Vec<usize>) = if dx >= 0 && dy >= 0 {
            ((0..width).collect(), (0..height).collect())
        } else if dx < 0 && dy >= 0 {
            (((0..width).rev()).collect(), (0..height).collect())
        } else if dx >= 0 && dy < 0 {
            ((0..width).collect(), ((0..height).rev()).collect())
        } else {
            (((0..width).rev()).collect(), ((0..height).rev()).collect())
        };

        for &y in &y_range {
            for &x in &x_range {
                let px = x as i32 - dx;
                let py = y as i32 - dy;

                for d in 0..num_disparities {
                    let idx = (y * width + x) * num_disparities + d;
                    let mut min_cost = cost_volume[idx];

                    if px >= 0 && px < width as i32 && py >= 0 && py < height as i32 {
                        let prev_idx = (py as usize * width + px as usize) * num_disparities;

                        // Find minimum previous cost
                        let mut min_prev = u32::MAX;
                        for pd in 0..num_disparities {
                            let pd_idx = prev_idx + pd;
                            if pd_idx < path_costs.len() {
                                let cost = path_costs[pd_idx];
                                let penalty = if d == pd {
                                    0
                                } else if (d as i32 - pd as i32).abs() == 1 {
                                    self.p1
                                } else {
                                    self.p2
                                };
                                min_prev = min_prev.min(cost + penalty);
                            }
                        }

                        if min_prev != u32::MAX {
                            min_cost = min_cost.saturating_add(min_prev);
                        }
                    }

                    path_costs[idx] = min_cost;
                }
            }
        }

        path_costs
    }

    fn find_best_disparity(
        &self,
        aggregated_costs: &[u32],
        x: usize,
        y: usize,
        width: usize,
        num_disparities: usize,
    ) -> i32 {
        let idx_base = (y * width + x) * num_disparities;

        let mut best_d = 0;
        let mut min_cost = u32::MAX;

        for d in 0..num_disparities {
            let cost = aggregated_costs[idx_base + d];
            if cost < min_cost {
                min_cost = cost;
                best_d = d;
            }
        }

        self.min_disparity + best_d as i32
    }
}

/// Compute stereo matching using Semi-Global Matching
pub fn stereo_sgm(left: &GrayImage, right: &GrayImage, max_disparity: i32) -> Result<DisparityMap> {
    let matcher = SgmMatcher::new().with_disparity_range(0, max_disparity);

    matcher.compute(left, right)
}

#[cfg(test)]
mod tests {
    use super::*;
    use image::Luma;

    fn create_test_stereo_pair() -> (GrayImage, GrayImage) {
        let width = 64u32;
        let height = 64u32;

        let mut left = GrayImage::new(width, height);
        let mut right = GrayImage::new(width, height);

        // Create vertical stripes with disparity
        for y in 0..height {
            for x in 0..width {
                let pattern = ((x / 8) % 2) * 200;
                left.put_pixel(x, y, Luma([pattern as u8]));

                // Shift pattern by 5 pixels for right image
                let shifted_x = if x >= 5 { x - 5 } else { 0 };
                let right_pattern = ((shifted_x / 8) % 2) * 200;
                right.put_pixel(x, y, Luma([right_pattern as u8]));
            }
        }

        (left, right)
    }

    #[test]
    fn test_sgm_matcher() {
        let (left, right) = create_test_stereo_pair();

        let matcher = SgmMatcher::new()
            .with_disparity_range(0, 16)
            .with_penalties(5, 60);

        let disparity = matcher.compute(&left, &right).unwrap();

        assert_eq!(disparity.width, left.width());
        assert_eq!(disparity.height, left.height());

        println!(
            "SGM computed disparity map: {}x{}",
            disparity.width, disparity.height
        );

        // Check that disparities are within expected range
        let valid_count = disparity
            .data
            .iter()
            .filter(|&&d| d >= 0.0 && d <= 16.0)
            .count();

        assert!(valid_count > 0, "Should have valid disparities");
    }
}
