//! Depth estimation from disparity maps
//!
//! Convert disparity values to 3D depth and point clouds.

use crate::{DisparityMap, StereoParams};
use nalgebra::{Point2, Point3};

/// 3D point cloud from stereo depth
#[derive(Debug, Clone)]
pub struct PointCloud {
    pub points: Vec<Point3<f64>>,
    pub colors: Option<Vec<(u8, u8, u8)>>,
}

impl PointCloud {
    pub fn new() -> Self {
        Self {
            points: Vec::new(),
            colors: None,
        }
    }

    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            points: Vec::with_capacity(capacity),
            colors: Some(Vec::with_capacity(capacity)),
        }
    }

    pub fn push(&mut self, point: Point3<f64>) {
        self.points.push(point);
    }

    pub fn push_with_color(&mut self, point: Point3<f64>, color: (u8, u8, u8)) {
        self.points.push(point);
        if let Some(ref mut colors) = self.colors {
            colors.push(color);
        }
    }

    pub fn len(&self) -> usize {
        self.points.len()
    }

    pub fn is_empty(&self) -> bool {
        self.points.is_empty()
    }
}

impl Default for PointCloud {
    fn default() -> Self {
        Self::new()
    }
}

/// Compute depth map from disparity map
pub fn disparity_to_depth(disparity: &DisparityMap, params: &StereoParams) -> Vec<Option<f64>> {
    let size = (disparity.width * disparity.height) as usize;
    let mut depths = Vec::with_capacity(size);

    for y in 0..disparity.height {
        for x in 0..disparity.width {
            let d = disparity.get(x, y);

            // Filter invalid disparities
            let depth = if d < 0.5 || d >= (disparity.max_disparity as f32) {
                None
            } else {
                params.disparity_to_depth(d as f64)
            };

            depths.push(depth);
        }
    }

    depths
}

/// Compute 3D point cloud from disparity map
pub fn disparity_to_pointcloud(
    disparity: &DisparityMap,
    left_image: &image::GrayImage,
    params: &StereoParams,
) -> PointCloud {
    let mut pointcloud = PointCloud::with_capacity((disparity.width * disparity.height) as usize);

    for y in 0..disparity.height {
        for x in 0..disparity.width {
            let d = disparity.get(x, y);

            // Skip invalid disparities
            if d < 0.5 || d >= (disparity.max_disparity as f32) {
                continue;
            }

            if let Some(depth) = params.disparity_to_depth(d as f64) {
                // Compute 3D coordinates using pinhole camera model
                // X = (x - cx) * Z / f
                // Y = (y - cy) * Z / f
                // Z = depth

                let x_normalized = (x as f64 - params.cx) / params.focal_length;
                let y_normalized = (y as f64 - params.cy) / params.focal_length;

                let x_3d = x_normalized * depth;
                let y_3d = y_normalized * depth;
                let z_3d = depth;

                let point = Point3::new(x_3d, y_3d, z_3d);

                // Get intensity as grayscale color
                let intensity = left_image.get_pixel(x, y)[0];
                pointcloud.push_with_color(point, (intensity, intensity, intensity));
            }
        }
    }

    pointcloud
}

/// Reproject 3D point to image coordinates
pub fn project_point_to_image(point: &Point3<f64>, params: &StereoParams) -> Option<Point2<f64>> {
    if point.z.abs() < 1e-6 {
        return None;
    }

    let x = point.x * params.focal_length / point.z + params.cx;
    let y = point.y * params.focal_length / point.z + params.cy;

    Some(Point2::new(x, y))
}

/// Compute 3D coordinates from disparity
pub fn disparity_to_3d(
    x: u32,
    y: u32,
    disparity: f64,
    params: &StereoParams,
) -> Option<Point3<f64>> {
    if disparity.abs() < 1e-6 {
        return None;
    }

    let depth = (params.focal_length * params.baseline) / disparity;

    let x_normalized = (x as f64 - params.cx) / params.focal_length;
    let y_normalized = (y as f64 - params.cy) / params.focal_length;

    let x_3d = x_normalized * depth;
    let y_3d = y_normalized * depth;
    let z_3d = depth;

    Some(Point3::new(x_3d, y_3d, z_3d))
}

/// Filter point cloud by depth range
pub fn filter_pointcloud_by_depth(
    pointcloud: &PointCloud,
    min_depth: f64,
    max_depth: f64,
) -> PointCloud {
    let mut filtered = PointCloud::new();

    for (i, point) in pointcloud.points.iter().enumerate() {
        let depth = point.z;
        if depth >= min_depth && depth <= max_depth {
            if let Some(ref colors) = pointcloud.colors {
                filtered.push_with_color(*point, colors[i]);
            } else {
                filtered.push(*point);
            }
        }
    }

    filtered
}

/// Compute depth statistics
pub fn compute_depth_stats(depths: &[Option<f64>]) -> Option<(f64, f64, f64)> {
    let valid_depths: Vec<f64> = depths.iter().filter_map(|&d| d).collect();

    if valid_depths.is_empty() {
        return None;
    }

    let min = valid_depths.iter().copied().fold(f64::INFINITY, f64::min);
    let max = valid_depths
        .iter()
        .copied()
        .fold(f64::NEG_INFINITY, f64::max);
    let mean = valid_depths.iter().sum::<f64>() / valid_depths.len() as f64;

    Some((min, max, mean))
}

/// Export point cloud to PLY format
pub fn export_to_ply(pointcloud: &PointCloud, filename: &str) -> std::io::Result<()> {
    use std::fs::File;
    use std::io::Write;

    let mut file = File::create(filename)?;

    // Write PLY header
    writeln!(file, "ply")?;
    writeln!(file, "format ascii 1.0")?;
    writeln!(file, "element vertex {}", pointcloud.len())?;
    writeln!(file, "property float x")?;
    writeln!(file, "property float y")?;
    writeln!(file, "property float z")?;

    if pointcloud.colors.is_some() {
        writeln!(file, "property uchar red")?;
        writeln!(file, "property uchar green")?;
        writeln!(file, "property uchar blue")?;
    }

    writeln!(file, "end_header")?;

    // Write points
    for (i, point) in pointcloud.points.iter().enumerate() {
        if let Some(ref colors) = pointcloud.colors {
            let (r, g, b) = colors[i];
            writeln!(
                file,
                "{} {} {} {} {} {}",
                point.x, point.y, point.z, r, g, b
            )?;
        } else {
            writeln!(file, "{} {} {}", point.x, point.y, point.z)?;
        }
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::DisparityMap;

    #[test]
    fn test_disparity_to_depth() {
        let params = StereoParams::new(500.0, 0.1, 320.0, 240.0);

        // Create a simple disparity map
        let mut disparity = DisparityMap::new(10, 10, 0, 64);
        disparity.set(5, 5, 50.0);

        let depths = disparity_to_depth(&disparity, &params);

        // At position (5, 5), depth should be (500 * 0.1) / 50 = 1.0
        let idx = 5 * 10 + 5;
        assert_eq!(depths[idx], Some(1.0));
    }

    #[test]
    fn test_disparity_to_3d() {
        let params = StereoParams::new(500.0, 0.1, 320.0, 240.0);

        // At center point with disparity 50
        let point = disparity_to_3d(320, 240, 50.0, &params);

        assert!(point.is_some());
        let p = point.unwrap();

        // At center (cx, cy), x_normalized and y_normalized are 0
        // So x_3d and y_3d should be 0
        assert!((p.x).abs() < 1e-6);
        assert!((p.y).abs() < 1e-6);
        assert!((p.z - 1.0).abs() < 1e-6); // depth = 1.0
    }

    #[test]
    fn test_pointcloud_filtering() {
        let mut pc = PointCloud::new();

        pc.push(Point3::new(1.0, 2.0, 0.5)); // Too close
        pc.push(Point3::new(2.0, 3.0, 2.0)); // Valid
        pc.push(Point3::new(3.0, 4.0, 5.0)); // Too far

        let filtered = filter_pointcloud_by_depth(&pc, 1.0, 3.0);

        assert_eq!(filtered.len(), 1);
        assert!((filtered.points[0].z - 2.0).abs() < 1e-6);
    }
}
