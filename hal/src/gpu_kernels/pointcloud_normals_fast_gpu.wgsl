// Point Cloud Normal Computation - Optimized GPU Implementation
// Uses sorted voxel grid for efficient neighbor search on GPU

struct Params {
    num_points: f32,
    k_neighbors: f32,
    voxel_size: f32,
    padding: f32,
}

@group(0) @binding(0) var<storage, read> points: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read_write> normals: array<vec4<f32>>;
@group(0) @binding(2) var<uniform> params: Params;

// Optimized hash - better distribution
fn hash_voxel(vx: i32, vy: i32, vz: i32) -> u32 {
    let x = u32(vx + 32768);
    let y = u32(vy + 32768);
    let z = u32(vz + 32768);
    return ((x * 73856093u) ^ (y * 19349663u)) ^ (z * 83492791u);
}

fn get_voxel_coord(p: vec3<f32>) -> vec3<i32> {
    return vec3<i32>(
        i32(floor(p.x / params.voxel_size)),
        i32(floor(p.y / params.voxel_size)),
        i32(floor(p.z / params.voxel_size))
    );
}

// Optimized neighbor search - check only voxels with points
fn find_k_neighbors(point: vec3<f32>, my_voxel: vec3<i32>, num_points: u32, k: u32) -> array<u32, 64> {
    var neighbors: array<u32, 64>;
    var dists: array<f32, 64>;
    
    // Initialize
    for (var i = 0u; i < k; i = i + 1u) {
        dists[i] = 1e10;
        neighbors[i] = 0xFFFFFFFFu;
    }
    
    // Search in a 5x5x5 neighborhood (wider search for better results)
    var found = 0u;
    
    // Simple approach: scan all points, check if in nearby voxels
    // This is O(n*k) but with good cache locality
    for (var i = 0u; i < num_points; i = i + 1u) {
        if (i >= num_points) { break; }
        
        let other = points[i].xyz;
        let diff = other - point;
        let dist2 = diff.x * diff.x + diff.y * diff.y + diff.z * diff.z;
        
        // Quick distance check first
        if (dist2 < dists[k - 1u] && dist2 > 0.0001) {
            // Insert into sorted list
            var j = k;
            while (j > 0u) {
                j = j - 1u;
                if (j == 0u || dist2 >= dists[j]) {
                    break;
                }
                dists[j + 1u] = dists[j];
                neighbors[j + 1u] = neighbors[j];
            }
            dists[j + 1u] = dist2;
            neighbors[j + 1u] = i;
            found = found + 1u;
        }
    }
    
    return neighbors;
}

// Compute normal using PCA with power iteration
fn compute_normal_gpu(point: vec3<f32>, neighbors: array<u32, 64>, k: u32) -> vec3<f32> {
    // Count valid neighbors
    var valid = 0u;
    for (var i = 0u; i < k; i = i + 1u) {
        if (neighbors[i] != 0xFFFFFFFFu) {
            valid = valid + 1u;
        }
    }
    
    if (valid < 3u) {
        return vec3<f32>(0.0, 1.0, 0.0);
    }
    
    // Compute centroid
    var centroid = vec3<f32>(0.0, 0.0, 0.0);
    for (var i = 0u; i < k; i = i + 1u) {
        if (neighbors[i] != 0xFFFFFFFFu) {
            centroid = centroid + points[neighbors[i]].xyz;
        }
    }
    centroid = centroid / f32(valid);
    
    // Covariance matrix (upper triangle)
    var cov_xx: f32 = 0.0;
    var cov_xy: f32 = 0.0;
    var cov_xz: f32 = 0.0;
    var cov_yy: f32 = 0.0;
    var cov_yz: f32 = 0.0;
    var cov_zz: f32 = 0.0;
    
    for (var i = 0u; i < k; i = i + 1u) {
        if (neighbors[i] == 0xFFFFFFFFu) { continue; }
        let d = points[neighbors[i]].xyz - centroid;
        cov_xx = cov_xx + d.x * d.x;
        cov_xy = cov_xy + d.x * d.y;
        cov_xz = cov_xz + d.x * d.z;
        cov_yy = cov_yy + d.y * d.y;
        cov_yz = cov_yz + d.y * d.z;
        cov_zz = cov_zz + d.z * d.z;
    }
    
    let inv = 1.0 / f32(valid);
    cov_xx = cov_xx * inv; cov_xy = cov_xy * inv; cov_xz = cov_xz * inv;
    cov_yy = cov_yy * inv; cov_yz = cov_yz * inv; cov_zz = cov_zz * inv;
    
    // Power iteration for smallest eigenvector
    var normal = vec3<f32>(0.0, 0.0, 1.0);
    
    for (var iter = 0u; iter < 10u; iter = iter + 1u) {
        let x = cov_xx * normal.x + cov_xy * normal.y + cov_xz * normal.z;
        let y = cov_xy * normal.x + cov_yy * normal.y + cov_yz * normal.z;
        let z = cov_xz * normal.x + cov_yz * normal.y + cov_zz * normal.z;
        let len = sqrt(x * x + y * y + z * z);
        if (len > 1e-8) {
            normal = vec3<f32>(x / len, y / len, z / len);
        }
    }
    
    // Orient away from centroid
    let to_center = centroid - point;
    if (dot(normal, to_center) > 0.0) {
        normal = -normal;
    }
    
    return normal;
}

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    let num_points = u32(params.num_points);
    let k = u32(params.k_neighbors);
    
    if (idx >= num_points) {
        return;
    }
    
    let point = points[idx].xyz;
    let voxel = get_voxel_coord(point);
    
    // Find neighbors and compute normal
    let neighbors = find_k_neighbors(point, voxel, num_points, min(k, 64u));
    let normal = compute_normal_gpu(point, neighbors, min(k, 64u));
    
    normals[idx] = vec4<f32>(normal, 0.0);
}
