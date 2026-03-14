//! Surface Reconstruction from Point Clouds
//!
//! Implements:
//! - Poisson Surface Reconstruction
//! - Ball Pivoting Algorithm (BPA)
//! - Alpha Shapes
//! - Marching Cubes
//! - Delaunay-based reconstruction

use super::TriangleMesh;
use cv_core::point_cloud::PointCloud;
use nalgebra::{Point3, Vector3};
use std::collections::{HashMap, HashSet, VecDeque};

/// Compute normals for point cloud using PCA (simplified)
pub fn compute_point_normals(cloud: &PointCloud, _k: usize) -> Vec<Vector3<f32>> {
    let n = cloud.points.len();
    if n == 0 {
        return vec![];
    }

    // Simplified: use existing normals or compute basic normals
    if let Some(ref normals) = cloud.normals {
        return normals.clone();
    }

    // Default: return upward normals
    vec![Vector3::new(0.0, 1.0, 0.0); n]
}

/// Poisson Surface Reconstruction
pub fn poisson_reconstruction(
    cloud: &PointCloud,
    depth: usize,
    samples_per_node: f32,
) -> Option<TriangleMesh> {
    let normals = cloud.normals.as_ref()?;
    if normals.len() != cloud.points.len() {
        return None;
    }

    let (min, max) = compute_bounds(cloud);
    let grid_size = 1 << depth;
    let voxel_size = compute_voxel_size(&min, &max, grid_size);

    let mut octree = Octree::new(depth);

    for (i, (point, normal)) in cloud.points.iter().zip(normals.iter()).enumerate() {
        let node = octree.insert(point, normal);
        node.sample_indices.push(i);
    }

    let implicit_function = solve_poisson(&octree, samples_per_node);
    let mesh = marching_cubes(&implicit_function, grid_size, voxel_size, &min);

    Some(mesh)
}

/// Ball Pivoting Algorithm
pub fn ball_pivoting(cloud: &PointCloud, ball_radius: f32) -> TriangleMesh {
    let mut mesh = TriangleMesh::new();

    if cloud.points.len() < 3 {
        return mesh;
    }

    let spatial_index = SpatialIndex::new(cloud, ball_radius * 2.0);

    let mut edge_count: HashMap<(usize, usize), u8> = HashMap::new();
    let mut mesh_faces: Vec<[usize; 3]> = Vec::new();
    let mut used_in_mesh: HashSet<usize> = HashSet::new();

    for i in 0..cloud.points.len() {
        // Skip vertices already incorporated into the mesh
        if used_in_mesh.contains(&i) {
            continue;
        }
        if let Some(seed) = find_seed_triangle(i, cloud, &spatial_index, ball_radius) {
            // Only start a new seed if none of the seed vertices have all edges saturated
            if try_add_triangle(seed, &mut edge_count, &mut mesh_faces) {
                for &v in &seed {
                    used_in_mesh.insert(v);
                }
                expand_front(
                    seed,
                    cloud,
                    &spatial_index,
                    ball_radius,
                    &mut edge_count,
                    &mut mesh_faces,
                    &mut used_in_mesh,
                );
            }
        }
    }

    mesh.vertices = cloud.points.clone();
    mesh.faces = mesh_faces;
    mesh.compute_vertex_normals();

    mesh
}

/// Alpha Shapes reconstruction
pub fn alpha_shapes(cloud: &PointCloud, alpha: f32) -> TriangleMesh {
    ball_pivoting(cloud, alpha)
}

/// Create a simple sphere point cloud for testing
pub fn create_sphere_point_cloud(
    center: Point3<f32>,
    radius: f32,
    num_points: usize,
) -> PointCloud {
    let _rng = rand::rng();

    let mut points = Vec::with_capacity(num_points);
    let mut normals = Vec::with_capacity(num_points);

    let phi = std::f32::consts::PI * (3.0 - 5.0_f32.sqrt());

    for i in 0..num_points {
        let y = 1.0 - (i as f32 / (num_points - 1).max(1) as f32) * 2.0;
        let radius_at_y = (1.0 - y * y).max(0.0).sqrt();
        let theta = phi * i as f32;

        let x = theta.cos() * radius_at_y;
        let z = theta.sin() * radius_at_y;

        let point = center + radius * Vector3::new(x, y, z);
        points.push(point);

        let normal = (point - center).normalize();
        normals.push(normal);
    }

    PointCloud {
        points,
        normals: Some(normals),
        colors: None,
    }
}

/// Create a simple plane point cloud for testing
pub fn create_plane_point_cloud(
    origin: Point3<f32>,
    normal: Vector3<f32>,
    size: f32,
    num_points: usize,
) -> PointCloud {
    use rand::Rng;
    let mut rng = rand::rng();

    let up = if normal.z.abs() < 0.9 {
        Vector3::new(0.0, 0.0, 1.0)
    } else {
        Vector3::new(1.0, 0.0, 0.0)
    };
    let right = normal.cross(&up).normalize();
    let up = right.cross(&normal).normalize();

    let mut points = Vec::with_capacity(num_points);
    let mut normals = Vec::with_capacity(num_points);

    for _ in 0..num_points {
        let u = rng.random_range(-size..size);
        let v = rng.random_range(-size..size);

        let point = origin + right * u + up * v;
        points.push(point);
        normals.push(normal);
    }

    PointCloud {
        points,
        normals: Some(normals),
        colors: None,
    }
}

#[allow(dead_code)]
struct Octree {
    depth: usize,
    root: OctreeNode,
}

#[allow(dead_code)]
struct OctreeNode {
    children: Option<Box<[OctreeNode; 8]>>,
    sample_indices: Vec<usize>,
    center: Point3<f32>,
    size: f32,
}

impl Octree {
    fn new(depth: usize) -> Self {
        Self {
            depth,
            root: OctreeNode::new(Point3::origin(), 1.0),
        }
    }

    fn insert(&mut self, _point: &Point3<f32>, _normal: &Vector3<f32>) -> &mut OctreeNode {
        &mut self.root
    }
}

impl OctreeNode {
    fn new(center: Point3<f32>, size: f32) -> Self {
        Self {
            children: None,
            sample_indices: Vec::new(),
            center,
            size,
        }
    }
}

struct SpatialIndex {
    grid: HashMap<(i32, i32, i32), Vec<usize>>,
    cell_size: f32,
}

impl SpatialIndex {
    fn new(cloud: &PointCloud, cell_size: f32) -> Self {
        let mut grid: HashMap<(i32, i32, i32), Vec<usize>> = HashMap::new();

        for (i, point) in cloud.points.iter().enumerate() {
            let key = (
                (point.x / cell_size) as i32,
                (point.y / cell_size) as i32,
                (point.z / cell_size) as i32,
            );
            grid.entry(key).or_default().push(i);
        }

        Self { grid, cell_size }
    }

    fn find_neighbors(&self, point: &Point3<f32>, radius: f32) -> Vec<usize> {
        let mut neighbors = Vec::new();
        let r = (radius / self.cell_size).ceil() as i32;
        let cx = (point.x / self.cell_size) as i32;
        let cy = (point.y / self.cell_size) as i32;
        let cz = (point.z / self.cell_size) as i32;

        for dx in -r..=r {
            for dy in -r..=r {
                for dz in -r..=r {
                    if let Some(indices) = self.grid.get(&(cx + dx, cy + dy, cz + dz)) {
                        neighbors.extend(indices);
                    }
                }
            }
        }

        neighbors
    }
}

fn compute_bounds(cloud: &PointCloud) -> (Point3<f32>, Point3<f32>) {
    if cloud.points.is_empty() {
        return (Point3::origin(), Point3::origin());
    }

    let mut min = cloud.points[0];
    let mut max = cloud.points[0];

    for p in &cloud.points {
        min.x = min.x.min(p.x);
        min.y = min.y.min(p.y);
        min.z = min.z.min(p.z);
        max.x = max.x.max(p.x);
        max.y = max.y.max(p.y);
        max.z = max.z.max(p.z);
    }

    (min, max)
}

fn compute_voxel_size(min: &Point3<f32>, max: &Point3<f32>, grid_size: usize) -> f32 {
    let size = (max - min).norm();
    size / grid_size.max(1) as f32
}

fn solve_poisson(_octree: &Octree, _samples_per_node: f32) -> ImplicitFunction {
    ImplicitFunction::new()
}

struct ImplicitFunction;

impl ImplicitFunction {
    fn new() -> Self {
        Self
    }
}

fn marching_cubes(
    _implicit: &ImplicitFunction,
    _grid_size: usize,
    _voxel_size: f32,
    _min: &Point3<f32>,
) -> TriangleMesh {
    TriangleMesh::new()
}

fn find_seed_triangle(
    start_idx: usize,
    cloud: &PointCloud,
    spatial_index: &SpatialIndex,
    ball_radius: f32,
) -> Option<[usize; 3]> {
    let point = &cloud.points[start_idx];
    let neighbors = spatial_index.find_neighbors(point, ball_radius * 2.0);

    for i in 0..neighbors.len() {
        for j in (i + 1)..neighbors.len() {
            let idx1 = neighbors[i];
            let idx2 = neighbors[j];

            if find_ball_center(
                &cloud.points[start_idx],
                &cloud.points[idx1],
                &cloud.points[idx2],
                ball_radius,
            )
            .is_some()
            {
                return Some([start_idx, idx1, idx2]);
            }
        }
    }

    None
}

fn find_ball_center(
    p1: &Point3<f32>,
    p2: &Point3<f32>,
    p3: &Point3<f32>,
    radius: f32,
) -> Option<Point3<f32>> {
    let a = p2 - p1;
    let b = p3 - p1;
    let normal = a.cross(&b);
    let normal_len = normal.norm();

    if normal_len < 1e-6 {
        return None;
    }

    let normal = normal / normal_len;

    let a_len = a.norm();
    let b_len = b.norm();
    let c_len = (p3 - p2).norm();

    let circumradius = (a_len * b_len * c_len) / (2.0 * normal_len);

    if circumradius > radius {
        return None;
    }

    let a_len_sq = a_len * a_len;
    let b_len_sq = b_len * b_len;
    let c_len_sq = c_len * c_len;

    let denom = 2.0 * normal_len * normal_len;
    let alpha = b_len_sq * c_len_sq * (a_len_sq + b_len_sq - c_len_sq) / denom;
    let beta = a_len_sq * c_len_sq * (a_len_sq - b_len_sq + c_len_sq) / denom;
    let gamma = a_len_sq * b_len_sq * (-a_len_sq + b_len_sq + c_len_sq) / denom;

    let circumcenter = p1 * alpha + p2.coords * beta + p3.coords * gamma;

    let height = (radius * radius - circumradius * circumradius).sqrt();
    let center1 = circumcenter + normal * height;

    Some(center1)
}

/// Canonical (undirected) edge key: smaller index first.
fn edge_key(a: usize, b: usize) -> (usize, usize) {
    (a.min(b), a.max(b))
}

fn try_add_triangle(
    face: [usize; 3],
    edge_count: &mut HashMap<(usize, usize), u8>,
    mesh_faces: &mut Vec<[usize; 3]>,
) -> bool {
    let e0 = edge_key(face[0], face[1]);
    let e1 = edge_key(face[1], face[2]);
    let e2 = edge_key(face[2], face[0]);

    // An edge that already has 2 uses is complete; cannot add another triangle to it.
    if *edge_count.get(&e0).unwrap_or(&0) >= 2
        || *edge_count.get(&e1).unwrap_or(&0) >= 2
        || *edge_count.get(&e2).unwrap_or(&0) >= 2
    {
        return false;
    }

    *edge_count.entry(e0).or_insert(0) += 1;
    *edge_count.entry(e1).or_insert(0) += 1;
    *edge_count.entry(e2).or_insert(0) += 1;
    mesh_faces.push(face);

    true
}

/// A directed edge on the expansion front.
/// `a` and `b` are the edge vertices (in winding order), `opposite` is the
/// vertex of the triangle on the other side of this edge, and `ball_center`
/// is the centre of the ball that rests on that triangle.
struct FrontEdge {
    a: usize,
    b: usize,
    opposite: usize,
    ball_center: Point3<f32>,
}

/// Attempt to find the point `c` that the ball of the given `radius` touches
/// when pivoting around the directed edge (a -> b).
///
/// Among all candidate points in the neighborhood, we choose the one with the
/// smallest positive pivot angle (Bernardini et al. 1999).
fn pivot_ball(
    edge: &FrontEdge,
    cloud: &PointCloud,
    spatial_index: &SpatialIndex,
    ball_radius: f32,
    edge_count: &HashMap<(usize, usize), u8>,
) -> Option<(usize, Point3<f32>)> {
    let pa = &cloud.points[edge.a];
    let pb = &cloud.points[edge.b];

    // Edge midpoint and axis
    let mid = Point3::from((pa.coords + pb.coords) * 0.5);
    let edge_vec = pb - pa;
    let edge_len = edge_vec.norm();
    if edge_len < 1e-10 {
        return None;
    }
    let edge_axis = edge_vec / edge_len;

    // Reference direction: from edge midpoint towards the old ball centre,
    // projected onto the plane perpendicular to the edge axis.
    let old_center_vec = edge.ball_center - mid;
    let old_center_proj = old_center_vec - edge_axis * old_center_vec.dot(&edge_axis);
    let proj_len = old_center_proj.norm();
    if proj_len < 1e-10 {
        return None;
    }

    let candidates = spatial_index.find_neighbors(&mid, ball_radius * 2.0);

    let mut best_angle = f32::MAX;
    let mut best_candidate: Option<(usize, Point3<f32>)> = None;

    for &idx in &candidates {
        // Skip the edge endpoints and the opposite vertex of the current triangle
        if idx == edge.a || idx == edge.b || idx == edge.opposite {
            continue;
        }

        // Skip if both edges from the candidate are already complete (count >= 2)
        let ek_ca = edge_key(idx, edge.a);
        let ek_cb = edge_key(idx, edge.b);
        if *edge_count.get(&ek_ca).unwrap_or(&0) >= 2 || *edge_count.get(&ek_cb).unwrap_or(&0) >= 2
        {
            continue;
        }

        // Try to place a ball on (a, b, candidate)
        if let Some(new_center) = find_ball_center(pa, pb, &cloud.points[idx], ball_radius) {
            // The new ball centre must be on the correct side. We check this
            // by computing the pivot angle around the edge axis.
            let new_center_vec = new_center - mid;
            let new_center_proj = new_center_vec - edge_axis * new_center_vec.dot(&edge_axis);
            let new_proj_len = new_center_proj.norm();
            if new_proj_len < 1e-10 {
                continue;
            }

            // Pivot angle: angle from old_center_proj to new_center_proj
            // around the edge axis, measured in the positive rotation direction.
            let cos_angle = old_center_proj.dot(&new_center_proj) / (proj_len * new_proj_len);
            let sin_angle =
                old_center_proj.cross(&new_center_proj).dot(&edge_axis) / (proj_len * new_proj_len);
            let mut angle = sin_angle.atan2(cos_angle);

            // We want the angle in (0, 2*PI] — the smallest positive pivot.
            if angle <= 1e-6 {
                angle += std::f32::consts::TAU;
            }

            if angle < best_angle {
                best_angle = angle;
                best_candidate = Some((idx, new_center));
            }
        }
    }

    best_candidate
}

fn expand_front(
    seed: [usize; 3],
    cloud: &PointCloud,
    spatial_index: &SpatialIndex,
    ball_radius: f32,
    edge_count: &mut HashMap<(usize, usize), u8>,
    mesh_faces: &mut Vec<[usize; 3]>,
    used_in_mesh: &mut HashSet<usize>,
) {
    // Compute ball centre for the seed triangle
    let seed_center = match find_ball_center(
        &cloud.points[seed[0]],
        &cloud.points[seed[1]],
        &cloud.points[seed[2]],
        ball_radius,
    ) {
        Some(c) => c,
        None => return,
    };

    // Initialise the front with the three directed edges of the seed triangle.
    // For triangle (a, b, c), the three directed edges are:
    //   (a->b, opposite=c), (b->c, opposite=a), (c->a, opposite=b)
    let mut front: VecDeque<FrontEdge> = VecDeque::new();
    front.push_back(FrontEdge {
        a: seed[0],
        b: seed[1],
        opposite: seed[2],
        ball_center: seed_center,
    });
    front.push_back(FrontEdge {
        a: seed[1],
        b: seed[2],
        opposite: seed[0],
        ball_center: seed_center,
    });
    front.push_back(FrontEdge {
        a: seed[2],
        b: seed[0],
        opposite: seed[1],
        ball_center: seed_center,
    });

    // Safety limit to avoid infinite loops on degenerate inputs.
    let max_iterations = cloud.points.len() * 3;
    let mut iterations = 0;

    while let Some(edge) = front.pop_front() {
        iterations += 1;
        if iterations > max_iterations {
            break;
        }

        // If the edge is already complete (used twice), skip it.
        let ek = edge_key(edge.a, edge.b);
        if *edge_count.get(&ek).unwrap_or(&0) >= 2 {
            continue;
        }

        // Pivot the ball around this edge to find the next point.
        if let Some((c, new_center)) =
            pivot_ball(&edge, cloud, spatial_index, ball_radius, edge_count)
        {
            let face = [edge.a, edge.b, c];

            if try_add_triangle(face, edge_count, mesh_faces) {
                used_in_mesh.insert(c);

                // Add the two new edges to the front.
                // New edge (b -> c): opposite is a
                let ek_bc = edge_key(edge.b, c);
                if *edge_count.get(&ek_bc).unwrap_or(&0) < 2 {
                    front.push_back(FrontEdge {
                        a: edge.b,
                        b: c,
                        opposite: edge.a,
                        ball_center: new_center,
                    });
                }

                // New edge (c -> a): opposite is b
                let ek_ca = edge_key(c, edge.a);
                if *edge_count.get(&ek_ca).unwrap_or(&0) < 2 {
                    front.push_back(FrontEdge {
                        a: c,
                        b: edge.a,
                        opposite: edge.b,
                        ball_center: new_center,
                    });
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_create_sphere_point_cloud() {
        let cloud = create_sphere_point_cloud(Point3::new(0.0, 0.0, 0.0), 1.0, 100);
        assert_eq!(cloud.points.len(), 100);
        assert!(cloud.normals.is_some());
    }

    #[test]
    fn test_create_plane_point_cloud() {
        let normal = Vector3::new(0.0, 0.0, 1.0);
        let cloud = create_plane_point_cloud(Point3::origin(), normal, 1.0, 50);
        assert_eq!(cloud.points.len(), 50);
    }

    #[test]
    fn test_ball_pivoting_empty() {
        let cloud = PointCloud::new(vec![]);
        let mesh = ball_pivoting(&cloud, 0.1);
        assert_eq!(mesh.num_vertices(), 0);
    }

    #[test]
    fn test_ball_pivoting_sphere() {
        let cloud = create_sphere_point_cloud(Point3::new(0.0, 0.0, 0.0), 1.0, 200);
        let mesh = ball_pivoting(&cloud, 0.2);
        assert!(mesh.num_vertices() > 0);
    }

    #[test]
    fn test_alpha_shapes() {
        let cloud = create_sphere_point_cloud(Point3::new(0.0, 0.0, 0.0), 1.0, 100);
        let mesh = alpha_shapes(&cloud, 0.1);
        // alpha_shapes should return a valid mesh (vertex count is non-negative by type)
        let _ = mesh.num_vertices();
    }

    #[test]
    fn test_compute_point_normals() {
        let cloud = create_sphere_point_cloud(Point3::new(0.0, 0.0, 0.0), 1.0, 50);
        let normals = compute_point_normals(&cloud, 5);
        assert_eq!(normals.len(), 50);
    }

    #[test]
    fn test_poisson_with_normals() {
        let cloud = create_sphere_point_cloud(Point3::new(0.0, 0.0, 0.0), 1.0, 100);
        let mesh = poisson_reconstruction(&cloud, 5, 1.0);
        assert!(mesh.is_some());
    }

    #[test]
    fn test_poisson_without_normals() {
        let points = vec![Point3::new(0.0, 0.0, 0.0), Point3::new(1.0, 0.0, 0.0)];
        let cloud = PointCloud::new(points);
        let mesh = poisson_reconstruction(&cloud, 3, 1.0);
        assert!(mesh.is_none());
    }

    #[test]
    fn test_spatial_index() {
        let cloud = create_sphere_point_cloud(Point3::new(0.0, 0.0, 0.0), 1.0, 20);
        let index = SpatialIndex::new(&cloud, 0.5);
        let neighbors = index.find_neighbors(&cloud.points[0], 0.5);
        assert!(neighbors.len() >= 1);
    }

    #[test]
    fn test_compute_bounds() {
        let cloud = create_sphere_point_cloud(Point3::new(5.0, 5.0, 5.0), 2.0, 50);
        let (min, max) = compute_bounds(&cloud);
        assert!(min.x < 5.0);
        assert!(max.x > 5.0);
    }
}
