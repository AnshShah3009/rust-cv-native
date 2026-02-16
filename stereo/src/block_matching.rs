//! Block Matching Stereo Algorithm
//!
//! Simple but fast stereo matching using Sum of Absolute Differences (SAD)
//! or Sum of Squared Differences (SSD) over a local window.

use crate::{DisparityMap, Result, StereoError};
use image::GrayImage;
use rayon::prelude::*;

/// Block matching stereo matcher
pub struct BlockMatcher {
    pub block_size: usize,
    pub min_disparity: i32,
    pub max_disparity: i32,
    pub metric: MatchingMetric,
    pub uniqueness_ratio: f32,
}

#[derive(Clone, Copy)]
pub enum MatchingMetric {
    SAD, // Sum of Absolute Differences
    SSD, // Sum of Squared Differences
    NCC, // Normalized Cross-Correlation
}

impl Default for BlockMatcher {
    fn default() -> Self {
        Self {
            block_size: 11,
            min_disparity: 0,
            max_disparity: 64,
            metric: MatchingMetric::SAD,
            uniqueness_ratio: 0.95,
        }
    }
}

impl BlockMatcher {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_block_size(mut self, size: usize) -> Self {
        self.block_size = size;
        self
    }

    pub fn with_disparity_range(mut self, min: i32, max: i32) -> Self {
        self.min_disparity = min;
        self.max_disparity = max;
        self
    }

    pub fn with_metric(mut self, metric: MatchingMetric) -> Self {
        self.metric = metric;
        self
    }

    pub fn compute(&self, left: &GrayImage, right: &GrayImage) -> Result<DisparityMap> {
        if left.width() != right.width() || left.height() != right.height() {
            return Err(StereoError::SizeMismatch(
                "Left and right images must have the same dimensions".to_string(),
            ));
        }

        let width = left.width() as i32;
        let height = left.height() as i32;
        let width_usize = left.width() as usize;
        let half_block = (self.block_size / 2) as i32;
        let left_data = left.as_raw();
        let right_data = right.as_raw();

        let mut disparity = DisparityMap::new(
            left.width(),
            left.height(),
            self.min_disparity,
            self.max_disparity,
        );

        // Compute disparity map row-wise in parallel.
        disparity
            .data
            .par_chunks_mut(width_usize)
            .enumerate()
            .for_each(|(y_usize, row)| {
                let y = y_usize as i32;
                if y < half_block || y >= height - half_block {
                    return;
                }
                for x in half_block..width - half_block {
                    let best_disparity =
                        self.find_best_disparity(left_data, right_data, width_usize, x, y, half_block);
                    row[x as usize] = best_disparity as f32;
                }
            });

        Ok(disparity)
    }

    fn find_best_disparity(
        &self,
        left_data: &[u8],
        right_data: &[u8],
        width: usize,
        x: i32,
        y: i32,
        half_block: i32,
    ) -> i32 {
        // Clamp disparity search so all right-image accesses stay in-bounds.
        let min_valid = (x + half_block - (width as i32 - 1)).max(self.min_disparity);
        let max_valid = (x - half_block).min(self.max_disparity);
        if min_valid > max_valid {
            return -1;
        }

        let mut best_disparity = self.min_disparity;
        let mut best_cost = f32::INFINITY;
        let mut second_best_cost = f32::INFINITY;

        for d in min_valid..=max_valid {
            let cost = self.compute_matching_cost(left_data, right_data, width, x, y, d, half_block);

            if cost < best_cost {
                second_best_cost = best_cost;
                best_cost = cost;
                best_disparity = d;
            } else if cost < second_best_cost {
                second_best_cost = cost;
            }
        }

        // Uniqueness check
        if second_best_cost < best_cost * self.uniqueness_ratio {
            // Ambiguous match
            -1
        } else {
            best_disparity
        }
    }

    fn compute_matching_cost(
        &self,
        left_data: &[u8],
        right_data: &[u8],
        width: usize,
        x: i32,
        y: i32,
        disparity: i32,
        half_block: i32,
    ) -> f32 {
        let mut cost = 0.0f32;
        let mut count = 0usize;

        for dy in -half_block..=half_block {
            let ly = (y + dy) as usize;
            for dx in -half_block..=half_block {
                let lx = (x + dx) as usize;
                let rx = (x + dx - disparity) as usize;
                let left_val = left_data[ly * width + lx] as f32;
                let right_val = right_data[ly * width + rx] as f32;

                match self.metric {
                    MatchingMetric::SAD => {
                        cost += (left_val - right_val).abs();
                    }
                    MatchingMetric::SSD => {
                        let diff = left_val - right_val;
                        cost += diff * diff;
                    }
                    MatchingMetric::NCC => {
                        // Simplified NCC - would need mean subtraction in practice
                        cost += left_val * right_val;
                    }
                }
                count += 1;
            }
        }

        cost / count as f32
    }
}

/// Compute stereo matching using block matching
pub fn stereo_block_match(
    left: &GrayImage,
    right: &GrayImage,
    block_size: usize,
    max_disparity: i32,
) -> Result<DisparityMap> {
    let matcher = BlockMatcher::new()
        .with_block_size(block_size)
        .with_disparity_range(0, max_disparity);

    matcher.compute(left, right)
}

#[cfg(test)]
mod tests {
    use super::*;
    use image::Luma;

    fn create_test_stereo_pair() -> (GrayImage, GrayImage) {
        let width = 100u32;
        let height = 100u32;

        // Create left image with a pattern
        let mut left = GrayImage::new(width, height);
        for y in 0..height {
            for x in 0..width {
                let val = ((x / 10) * 25) as u8;
                left.put_pixel(x, y, Luma([val]));
            }
        }

        // Create right image with horizontal shift (disparity)
        let mut right = GrayImage::new(width, height);
        let disparity = 10i32;

        for y in 0..height {
            for x in 0..width {
                let src_x = if x >= disparity as u32 {
                    x - disparity as u32
                } else {
                    0
                };
                let val = ((src_x / 10) * 25) as u8;
                right.put_pixel(x, y, Luma([val]));
            }
        }

        (left, right)
    }

    #[test]
    fn test_block_matcher() {
        let (left, right) = create_test_stereo_pair();

        let matcher = BlockMatcher::new()
            .with_block_size(5)
            .with_disparity_range(0, 20);

        let disparity = matcher.compute(&left, &right).unwrap();

        assert_eq!(disparity.width, left.width());
        assert_eq!(disparity.height, left.height());

        // Check that we have some valid disparities
        let valid_count = disparity.data.iter().filter(|&&d| d >= 0.0).count();

        assert!(valid_count > 0, "Should have some valid disparities");
        println!("Found {} valid disparities", valid_count);
    }
}
