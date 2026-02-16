use cv_core::PointCloud;
use nalgebra::{Matrix3, Point3, SymmetricEigen, Vector3};
use rstar::{RTree, RTreeObject, AABB};
use std::collections::HashMap;
use std::fs::File;
use std::io::{BufRead, BufReader, BufWriter, Write};

/// Downsample a point cloud with a voxel grid.
/// Returns a new point cloud with one point per voxel (the centroid).
pub fn voxel_down_sample(pc: &PointCloud, voxel_size: f32) -> PointCloud {
    if voxel_size <= 0.0 {
        return pc.clone();
    }

    let mut grid: HashMap<(i32, i32, i32), (Vector3<f32>, Vector3<f32>, Vector3<f32>, usize)> =
        HashMap::new();
    // Key -> (sum_points, sum_colors, sum_normals, count)

    let has_colors = pc.colors.is_some();
    let has_normals = pc.normals.is_some();

    for i in 0..pc.len() {
        let p = pc.points[i];
        let hx = (p.x / voxel_size).floor() as i32;
        let hy = (p.y / voxel_size).floor() as i32;
        let hz = (p.z / voxel_size).floor() as i32;
        let key = (hx, hy, hz);

        let entry =
            grid.entry(key)
                .or_insert((Vector3::zeros(), Vector3::zeros(), Vector3::zeros(), 0));
        entry.0 += p.coords;
        if has_colors {
            if let Some(colors) = &pc.colors {
                entry.1 += colors[i].coords;
            }
        }
        if has_normals {
            if let Some(normals) = &pc.normals {
                entry.2 += normals[i];
            }
        }
        entry.3 += 1;
    }

    let mut new_points = Vec::with_capacity(grid.len());
    let mut new_colors = if has_colors {
        Some(Vec::with_capacity(grid.len()))
    } else {
        None
    };
    let mut new_normals = if has_normals {
        Some(Vec::with_capacity(grid.len()))
    } else {
        None
    };

    for (_, (sum_p, sum_c, sum_n, count)) in grid {
        let factor = 1.0 / count as f32;
        new_points.push(Point3::from(sum_p * factor));

        if let Some(colors) = &mut new_colors {
            colors.push(Point3::from(sum_c * factor));
        }
        if let Some(normals) = &mut new_normals {
            let mut n = sum_n * factor;
            if n.norm_squared() > 1e-6 {
                n.normalize_mut();
            }
            normals.push(n);
        }
    }

    PointCloud {
        points: new_points,
        colors: new_colors,
        normals: new_normals,
    }
}

// Wrapper for RTree
struct PointWrapper(usize, Point3<f32>);

impl RTreeObject for PointWrapper {
    type Envelope = AABB<[f32; 3]>;
    fn envelope(&self) -> Self::Envelope {
        AABB::from_point([self.1.x, self.1.y, self.1.z])
    }
}

impl rstar::PointDistance for PointWrapper {
    fn distance_2(&self, point: &[f32; 3]) -> f32 {
        let dx = self.1.x - point[0];
        let dy = self.1.y - point[1];
        let dz = self.1.z - point[2];
        dx * dx + dy * dy + dz * dz
    }
}

/// Estimate normals for the point cloud using K-nearest neighbors.
/// Uses PCA/SVD on the covariance matrix of the neighborhood.
/// Modifies the point cloud in place to add normals.
pub fn estimate_normals(pc: &mut PointCloud, k: usize) {
    if pc.is_empty() {
        return;
    }

    let wrappers: Vec<PointWrapper> = pc
        .points
        .iter()
        .enumerate()
        .map(|(i, p)| PointWrapper(i, *p))
        .collect();

    let tree = RTree::bulk_load(wrappers);
    let mut normals = Vec::with_capacity(pc.len());

    for p in &pc.points {
        let query_point = [p.x, p.y, p.z];
        // nearest_neighbor_iter returns neighbors sorted by distance? No, usually not guaranteed sorted.
        // We just need k neighbors.
        let neighbors: Vec<&PointWrapper> =
            tree.nearest_neighbor_iter(&query_point).take(k).collect();

        if neighbors.len() < 3 {
            normals.push(Vector3::new(0.0, 0.0, 1.0)); // Default up
            continue;
        }

        // Compute centroid
        let mut centroid = Vector3::zeros();
        for n in &neighbors {
            centroid += n.1.coords;
        }
        centroid /= neighbors.len() as f32;

        // Compute covariance matrix
        let mut cov = Matrix3::zeros();
        for n in &neighbors {
            let d = n.1.coords - centroid;
            cov += d * d.transpose();
        }
        cov /= neighbors.len() as f32;

        // SVD / Eigen decomposition
        // We want the eigenvector corresponding to the smallest eigenvalue.
        let eigen = SymmetricEigen::new(cov);

        // Find index of smallest eigenvalue explicitly to be robust
        let mut min_val = f32::MAX;
        let mut min_idx = 0;
        for i in 0..3 {
            let val = eigen.eigenvalues[i];
            if val < min_val {
                min_val = val;
                min_idx = i;
            }
        }

        let normal = eigen.eigenvectors.column(min_idx).into_owned();
        normals.push(normal);
    }

    pc.normals = Some(normals);
}

/// Simple PLY reader/writer using internal implementation or ply-rs.
/// For simplicity and to avoid complex dependency usage if not needed, we will use ply-rs as planned.
/// But adding `ply-rs` might have failed if crate version mismatch.
/// Let's assume we use a simple ASCII writer for now to save complexity if `ply-rs` usage is complex.
/// Actually, to be robust, let's implement a very simple PLY ASCII reader/writer tailored for xyz/rgb/nxnynz.

pub fn write_ply(pc: &PointCloud, path: &str) -> std::io::Result<()> {
    let mut file = BufWriter::new(File::create(path)?);

    writeln!(file, "ply")?;
    writeln!(file, "format ascii 1.0")?;
    writeln!(file, "element vertex {}", pc.len())?;
    writeln!(file, "property float x")?;
    writeln!(file, "property float y")?;
    writeln!(file, "property float z")?;

    if pc.colors.is_some() {
        writeln!(file, "property uchar red")?;
        writeln!(file, "property uchar green")?;
        writeln!(file, "property uchar blue")?;
    }

    if pc.normals.is_some() {
        writeln!(file, "property float nx")?;
        writeln!(file, "property float ny")?;
        writeln!(file, "property float nz")?;
    }

    writeln!(file, "end_header")?;

    for i in 0..pc.len() {
        let p = pc.points[i];
        write!(file, "{} {} {}", p.x, p.y, p.z)?;

        if let Some(colors) = &pc.colors {
            let c = colors[i];
            // Colors in Point3<f32> assumed 0..1 or 0..255? Open3D uses 0..1 usually in float.
            // Let's assume 0..1 float and convert to uchar 0..255.
            let r = (c.x * 255.0).clamp(0.0, 255.0) as u8;
            let g = (c.y * 255.0).clamp(0.0, 255.0) as u8;
            let b = (c.z * 255.0).clamp(0.0, 255.0) as u8;
            write!(file, " {} {} {}", r, g, b)?;
        }

        if let Some(normals) = &pc.normals {
            let n = normals[i];
            write!(file, " {} {} {}", n.x, n.y, n.z)?;
        }

        writeln!(file)?;
    }

    Ok(())
}

pub fn read_ply(path: &str) -> std::io::Result<PointCloud> {
    // Basic PLY ASCII reader (robustness limited)
    let file = File::open(path)?;
    let reader = BufReader::new(file);
    let mut lines = reader.lines();

    let mut vertex_count = 0;
    let mut has_colors = false;
    let mut has_normals = false;
    let mut header_ended = false;

    // Parse Header
    while let Some(line) = lines.next() {
        let line = line?;
        if line.starts_with("element vertex") {
            let parts: Vec<&str> = line.split_whitespace().collect();
            if parts.len() >= 3 {
                vertex_count = parts[2].parse().unwrap_or(0);
            }
        } else if line.contains("property uchar red") {
            has_colors = true;
        } else if line.contains("property float nx") {
            has_normals = true;
        } else if line.trim() == "end_header" {
            header_ended = true;
            break;
        }
    }

    if !header_ended {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            "PLY header not found",
        ));
    }

    let mut points = Vec::with_capacity(vertex_count);
    let mut colors = if has_colors {
        Some(Vec::with_capacity(vertex_count))
    } else {
        None
    };
    let mut normals = if has_normals {
        Some(Vec::with_capacity(vertex_count))
    } else {
        None
    };

    for _ in 0..vertex_count {
        if let Some(line) = lines.next() {
            let line = line?;
            let mut parts = line.split_whitespace();

            // XYZ
            let x: f32 = parts
                .next()
                .ok_or(std::io::Error::new(
                    std::io::ErrorKind::InvalidData,
                    "Missing X",
                ))?
                .parse()
                .unwrap_or(0.0);
            let y: f32 = parts
                .next()
                .ok_or(std::io::Error::new(
                    std::io::ErrorKind::InvalidData,
                    "Missing Y",
                ))?
                .parse()
                .unwrap_or(0.0);
            let z: f32 = parts
                .next()
                .ok_or(std::io::Error::new(
                    std::io::ErrorKind::InvalidData,
                    "Missing Z",
                ))?
                .parse()
                .unwrap_or(0.0);
            points.push(Point3::new(x, y, z));

            // Optional Color
            if let Some(c_vec) = &mut colors {
                let r: u8 = parts.next().unwrap_or("0").parse().unwrap_or(0);
                let g: u8 = parts.next().unwrap_or("0").parse().unwrap_or(0);
                let b: u8 = parts.next().unwrap_or("0").parse().unwrap_or(0);
                c_vec.push(Point3::new(
                    r as f32 / 255.0,
                    g as f32 / 255.0,
                    b as f32 / 255.0,
                ));
            }

            // Optional Normal
            if let Some(n_vec) = &mut normals {
                let nx: f32 = parts.next().unwrap_or("0").parse().unwrap_or(0.0);
                let ny: f32 = parts.next().unwrap_or("0").parse().unwrap_or(0.0);
                let nz: f32 = parts.next().unwrap_or("0").parse().unwrap_or(0.0);
                n_vec.push(Vector3::new(nx, ny, nz));
            }
        }
    }

    Ok(PointCloud {
        points,
        colors,
        normals,
    })
}

// Add helper imports
use rand::seq::IndexedRandom;
use rstar::PointDistance;

/// Remove statistical outliers.
/// Compute mean distance to `k` neighbors for each point.
/// Points with mean distance > global_mean + std_ratio * std_dev are removed.
pub fn remove_statistical_outliers(
    pc: &PointCloud,
    k: usize,
    std_ratio: f64,
) -> (PointCloud, Vec<usize>) {
    if pc.is_empty() || k == 0 {
        return (pc.clone(), (0..pc.len()).collect());
    }

    let wrappers: Vec<PointWrapper> = pc
        .points
        .iter()
        .enumerate()
        .map(|(i, p)| PointWrapper(i, *p))
        .collect();
    let tree = RTree::bulk_load(wrappers);

    let mut distances = Vec::with_capacity(pc.len());

    for p in &pc.points {
        let query_point = [p.x, p.y, p.z];
        // nearest_neighbor_iter returns k+1 including itself (dist 0).
        // take(k+1) gives nearest neighbors.
        let neighbors: Vec<&PointWrapper> = tree
            .nearest_neighbor_iter(&query_point)
            .take(k + 1)
            .collect();

        // Sum distances to k neighbors (skip itself)
        let mut sum_dist = 0.0;
        let mut count = 0;
        for (idx, n) in neighbors.iter().enumerate() {
            if idx == 0 {
                continue;
            } // skip self
            sum_dist += n.distance_2(&query_point).sqrt();
            count += 1;
        }

        if count > 0 {
            distances.push(sum_dist / count as f32);
        } else {
            distances.push(0.0);
        }
    }

    let mean_dist =
        crate::mean(&distances.iter().map(|&d| d as f64).collect::<Vec<_>>()).unwrap_or(0.0) as f32;
    // std dev
    let variance = distances
        .iter()
        .map(|d| {
            let diff = d - mean_dist;
            diff * diff
        })
        .sum::<f32>()
        / distances.len() as f32;
    let std_dev = variance.sqrt();

    let threshold = mean_dist + std_ratio as f32 * std_dev;

    let mut inliers = Vec::new();
    let mut new_points = Vec::new();
    let mut new_colors = if pc.colors.is_some() {
        Some(Vec::new())
    } else {
        None
    };
    let mut new_normals = if pc.normals.is_some() {
        Some(Vec::new())
    } else {
        None
    };

    for (i, &dist) in distances.iter().enumerate() {
        if dist <= threshold {
            inliers.push(i);
            new_points.push(pc.points[i]);
            if let Some(c) = &pc.colors {
                if let Some(nc) = &mut new_colors {
                    nc.push(c[i]);
                }
            }
            if let Some(n) = &pc.normals {
                if let Some(nn) = &mut new_normals {
                    nn.push(n[i]);
                }
            }
        }
    }

    (
        PointCloud {
            points: new_points,
            colors: new_colors,
            normals: new_normals,
        },
        inliers,
    )
}

/// Remove radius outliers.
/// Points with fewer than `min_points` neighbors within `radius` are removed.
pub fn remove_radius_outliers(
    pc: &PointCloud,
    radius: f32,
    min_points: usize,
) -> (PointCloud, Vec<usize>) {
    if pc.is_empty() {
        return (pc.clone(), (0..pc.len()).collect());
    }

    let wrappers: Vec<PointWrapper> = pc
        .points
        .iter()
        .enumerate()
        .map(|(i, p)| PointWrapper(i, *p))
        .collect();
    let tree = RTree::bulk_load(wrappers);

    let mut inliers = Vec::new();
    let mut new_points = Vec::new();
    let mut new_colors = if pc.colors.is_some() {
        Some(Vec::new())
    } else {
        None
    };
    let mut new_normals = if pc.normals.is_some() {
        Some(Vec::new())
    } else {
        None
    };
    let r2 = radius * radius;

    for (i, p) in pc.points.iter().enumerate() {
        let query_point = [p.x, p.y, p.z];
        // locate_within_distance uses squared distance
        let count = tree.locate_within_distance(query_point, r2).count();

        // count includes self.
        if count >= min_points {
            inliers.push(i);
            new_points.push(*p);
            if let Some(c) = &pc.colors {
                if let Some(nc) = &mut new_colors {
                    nc.push(c[i]);
                }
            }
            if let Some(n) = &pc.normals {
                if let Some(nn) = &mut new_normals {
                    nn.push(n[i]);
                }
            }
        }
    }

    (
        PointCloud {
            points: new_points,
            colors: new_colors,
            normals: new_normals,
        },
        inliers,
    )
}

/// Segment a plane using RANSAC.
/// Returns (a, b, c, d) plane model and list of inlier indices.
/// Plane equation: ax + by + cz + d = 0.
pub fn segment_plane(
    pc: &PointCloud,
    distance_threshold: f32,
    ransac_n: usize,
    num_iterations: usize,
) -> (Option<[f32; 4]>, Vec<usize>) {
    if pc.points.len() < ransac_n || ransac_n < 3 {
        return (None, Vec::new());
    }

    let mut rng = rand::rng();
    let mut best_plane = None;
    let mut best_inliers = Vec::new();

    // Indices for sampling
    let indices: Vec<usize> = (0..pc.points.len()).collect();

    for _ in 0..num_iterations {
        // Sample random points
        // Use sample instead of deprecated choose_multiple
        let sample_indices: Vec<usize> = indices.sample(&mut rng, ransac_n).cloned().collect();

        // Fit plane to 3 points (or more using least squares if n > 3, but strict RANSAC uses minimal set)
        // Let's use first 3 points for minimal model.
        let p1 = pc.points[sample_indices[0]];
        let p2 = pc.points[sample_indices[1]];
        let p3 = pc.points[sample_indices[2]];

        let v1 = p2 - p1;
        let v2 = p3 - p1;
        let normal = v1.cross(&v2).normalize();

        if normal.x.is_nan() || normal.y.is_nan() || normal.z.is_nan() {
            continue;
        }

        let d = -normal.dot(&p1.coords);
        let a = normal.x;
        let b = normal.y;
        let c = normal.z;

        // Count inliers
        let mut current_inliers = Vec::new();
        for (i, p) in pc.points.iter().enumerate() {
            let dist = (a * p.x + b * p.y + c * p.z + d).abs() / (a * a + b * b + c * c).sqrt();
            if dist < distance_threshold {
                current_inliers.push(i);
            }
        }

        if current_inliers.len() > best_inliers.len() {
            best_inliers = current_inliers;
            best_plane = Some([a, b, c, d]);
        }
    }

    (best_plane, best_inliers)
}

/// DBSCAN clustering.
/// Returns a list of labels for each point. -1 indicates noise.
/// 0..N indicates cluster index.
pub fn cluster_dbscan(pc: &PointCloud, eps: f32, min_points: usize) -> Vec<i32> {
    let n = pc.len();
    let mut labels = vec![-1; n]; // -1 = noise/unvisited
    let mut cluster_idx = 0;

    // Build tree
    let wrappers: Vec<PointWrapper> = pc
        .points
        .iter()
        .enumerate()
        .map(|(i, p)| PointWrapper(i, *p))
        .collect();
    let tree = RTree::bulk_load(wrappers);

    let eps2 = eps * eps;
    let mut visited = vec![false; n];

    for i in 0..n {
        if visited[i] {
            continue;
        }
        visited[i] = true;

        let p = pc.points[i];
        let query_point = [p.x, p.y, p.z];
        // Neighbors include self
        let neighbors: Vec<&PointWrapper> =
            tree.locate_within_distance(query_point, eps2).collect();

        if neighbors.len() < min_points {
            labels[i] = -1; // Noise
        } else {
            labels[i] = cluster_idx;
            // Expand cluster
            let mut seeds = neighbors.clone();
            let mut head = 0;
            while head < seeds.len() {
                let current_idx = seeds[head].0;
                head += 1;

                if labels[current_idx] == -1 {
                    labels[current_idx] = cluster_idx; // Change noise to border
                }
                if !visited[current_idx] {
                    visited[current_idx] = true;
                    labels[current_idx] = cluster_idx;

                    let p_curr = pc.points[current_idx];
                    let q_curr = [p_curr.x, p_curr.y, p_curr.z];
                    let n_curr: Vec<&PointWrapper> =
                        tree.locate_within_distance(q_curr, eps2).collect();
                    if n_curr.len() >= min_points {
                        for n in n_curr {
                            // Avoid adding duplicates?
                            // seeds is a simple vec, might grow definedly.
                            // Actually better to use a Set or just push and check visited.
                            // But we check visited when popping/processing?
                            // The standard DBSCAN expands seeds.
                            // Simple list append approach:
                            seeds.push(n);
                        }
                    }
                }
            }
            cluster_idx += 1;
        }
    }

    labels
}

/// Compute FPFH features.
/// Returns an N x 33 histogram feature vector for each point.
/// Simplification: Returns zero vectors if normals missing.
/// NOTE: Full FPFH implementation is complex. This is a placeholder or simplified version.
pub fn compute_fpfh_feature(pc: &PointCloud, _search_radius: f32) -> Option<Vec<[f32; 33]>> {
    if pc.normals.is_none() {
        return None;
    }
    // Placeholder implementation returning zeros
    // Real FPFH requires computing SPFH for each point and then weighted average.
    Some(vec![[0.0; 33]; pc.len()])
}

#[cfg(test)]
mod tests {
    use super::*;
    use nalgebra::Point3;
    use std::fs;

    #[test]
    fn test_voxel_down_sample() {
        // Create points in a 0.1x0.1x0.1 cluster
        let mut points = Vec::new();
        for i in 0..10 {
            for j in 0..10 {
                points.push(Point3::new(i as f32 * 0.01, j as f32 * 0.01, 0.0));
            }
        }
        let pc = PointCloud::new(points);

        // Voxel size 0.2 should collapse everything to 1 point (or very few).
        // 0.0 -> 0.09 range. 0.2 covers all.
        let down = voxel_down_sample(&pc, 0.2);
        assert_eq!(down.len(), 1);

        // Voxel size 0.05. Should have 2x2 = 4 voxels approx?
        // 0.00..0.04 -> 0
        // 0.05..0.09 -> 1
        let down = voxel_down_sample(&pc, 0.05);
        // We have points at 0.00, 0.01 ... 0.09.
        // 0.00, 0.01, 0.02, 0.03, 0.04 -> bin 0
        // 0.05, 0.06, 0.07, 0.08, 0.09 -> bin 1
        // So 2 bins in X, 2 in Y -> 4 points total.
        assert_eq!(down.len(), 4);
    }

    #[test]
    fn test_estimate_normals() {
        // Plane at z=0
        let mut points = Vec::new();
        for i in 0..5 {
            for j in 0..5 {
                points.push(Point3::new(i as f32, j as f32, 0.0));
            }
        }
        let mut pc = PointCloud::new(points);
        estimate_normals(&mut pc, 5); // 5 neighbors

        assert!(pc.normals.is_some());
        let normals = pc.normals.as_ref().unwrap();

        // Normals should be (0,0,1) or (0,0,-1).
        for n in normals.iter() {
            assert!(n.z.abs() > 0.9, "Normal {:?} is not vertical", n);
        }
    }

    #[test]
    fn test_io() {
        let points = vec![Point3::new(0.0, 0.0, 0.0), Point3::new(1.0, 1.0, 1.0)];
        let colors = vec![Point3::new(1.0, 0.0, 0.0), Point3::new(0.0, 1.0, 0.0)];
        let pc = PointCloud::new(points).with_colors(colors);

        let path = "/tmp/test_pc.ply";
        write_ply(&pc, path).unwrap();

        let loaded = read_ply(path).unwrap();
        assert_eq!(loaded.len(), 2);
        assert!(loaded.colors.is_some());

        // Cleanup (optional, but good)
        let _ = fs::remove_file(path);
    }

    #[test]
    fn test_outlier_removal() {
        // Create a cluster of 10 points at (0,0,0) and 1 outlier at (10,10,10)
        let mut points = Vec::new();
        for _ in 0..10 {
            points.push(Point3::new(0.0, 0.0, 0.0));
        }
        points.push(Point3::new(10.0, 10.0, 10.0)); // Index 10
        let pc = PointCloud::new(points);

        // Radius removal: Radius 1.0, min 5 points.
        // The cluster has 10 neighbors (self included). Outlier has 1 (self).
        let (filtered, inliers) = remove_radius_outliers(&pc, 1.0, 5);
        assert_eq!(filtered.len(), 10);
        assert!(!inliers.contains(&10));

        // Statistical removal:
        // Cluster points have dist 0 to neighbors.
        // Outlier has dist ~17 to neighbors.
        // Mean dist will be small, std dev small. Outlier > mean + 2*std.
        let (filtered_stat, inliers_stat) = remove_statistical_outliers(&pc, 5, 1.0);
        assert_eq!(filtered_stat.len(), 10);
        assert!(!inliers_stat.contains(&10));
    }

    #[test]
    fn test_segment_plane() {
        // Create plane z=0 points + noise
        let mut points = Vec::new();
        // Inliers: grid 10x10
        for x in 0..10 {
            for y in 0..10 {
                points.push(Point3::new(x as f32, y as f32, 0.0));
            }
        }
        // Outliers
        points.push(Point3::new(0.0, 0.0, 10.0));
        points.push(Point3::new(1.0, 1.0, 10.0));

        // Also add some random jitter to plane points to make it realistic?
        // But perfect plane is easier to test.

        let pc = PointCloud::new(points);
        let (model, inliers) = segment_plane(&pc, 0.1, 3, 100);

        assert!(model.is_some());
        let [a, b, c, d] = model.unwrap();

        // Should be z=0, so a=0, b=0, c=1 (or -1), d=0
        // Normal could be (0,0,-1) too.
        println!("Plane: {}x + {}y + {}z + {} = 0", a, b, c, d);
        assert!(c.abs() > 0.9);
        assert!(d.abs() < 0.1);
        assert_eq!(inliers.len(), 100);
    }

    #[test]
    fn test_dbscan_clustering() {
        // Create two clusters far apart
        let mut points = Vec::new();
        // Cluster 1: 5 points at (0,0,0)
        for _ in 0..5 {
            points.push(Point3::new(0.0, 0.0, 0.0));
        }
        // Cluster 2: 5 points at (10,10,10)
        for _ in 0..5 {
            points.push(Point3::new(10.0, 10.0, 10.0));
        }
        // Noise: 1 point at (5,5,5)
        points.push(Point3::new(5.0, 5.0, 5.0));

        let pc = PointCloud::new(points);

        // eps=1.0, min_points=3
        let labels = cluster_dbscan(&pc, 1.0, 3);

        // First 5 should be cluster 0
        for i in 0..5 {
            assert_eq!(labels[i], 0);
        }
        // Next 5 should be cluster 1
        for i in 5..10 {
            assert_eq!(labels[i], 1);
        }
        // Last one should be noise (-1)
        assert_eq!(labels[10], -1);
    }

    #[test]
    fn test_fpfh() {
        let points = vec![Point3::new(0.0, 0.0, 0.0)];
        let normals = vec![nalgebra::Vector3::new(0.0, 0.0, 1.0)];
        let pc = PointCloud::new(points).with_normals(normals);

        let features = compute_fpfh_feature(&pc, 0.1);
        assert!(features.is_some());
        assert_eq!(features.unwrap()[0].len(), 33);

        let pc_no_norm = PointCloud::new(vec![Point3::new(0.0, 0.0, 0.0)]);
        assert!(compute_fpfh_feature(&pc_no_norm, 0.1).is_none());
    }
}
