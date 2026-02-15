// TSDF Integration Shader
// Integrates depth frame into TSDF volume

@group(0) @binding(0) var<storage, read> depth_image: array<f32>;
@group(0) @binding(1) var<storage, read> color_image: array<u32>; // Packed RGBA
@group(0) @binding(2) var<storage, read_write> tsdf_values: array<f32>;
@group(0) @binding(3) var<storage, read_write> weights: array<f32>;
@group(0) @binding(4) var<storage, read_write> colors: array<u32>;

@group(1) @binding(0) var<uniform> camera_pose: mat4x4<f32>;
@group(1) @binding(1) var<uniform> intrinsics: vec4<f32>; // fx, fy, cx, cy
@group(1) @binding(2) var<uniform> image_size: vec2<u32>; // width, height
@group(1) @binding(3) var<uniform> voxel_size: f32;
@group(1) @binding(4) var<uniform> truncation_distance: f32;
@group(1) @binding(5) var<uniform> volume_size: vec3<u32>; // vx, vy, vz

// Backproject pixel to 3D point in camera space
fn unproject(u: u32, v: u32, depth: f32) -> vec3<f32> {
    let fx = intrinsics.x;
    let fy = intrinsics.y;
    let cx = intrinsics.z;
    let cy = intrinsics.w;
    
    let x = (f32(u) - cx) * depth / fx;
    let y = (f32(v) - cy) * depth / fy;
    let z = depth;
    
    return vec3<f32>(x, y, z);
}

// Get voxel index from 3D coordinates
fn voxel_index(vx: u32, vy: u32, vz: u32) -> u32 {
    return vz * volume_size.x * volume_size.y + vy * volume_size.x + vx;
}

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let u = global_id.x;
    let v = global_id.y;
    
    if (u >= image_size.x || v >= image_size.y) {
        return;
    }
    
    let pixel_idx = v * image_size.x + u;
    let depth = depth_image[pixel_idx];
    
    // Skip invalid depth
    if (depth <= 0.0 || depth > 10.0) {
        return;
    }
    
    // Backproject to camera space
    let point_camera = unproject(u, v, depth);
    
    // Transform to world space
    let p_h = vec4<f32>(point_camera, 1.0);
    let point_world = (camera_pose * p_h).xyz;
    
    // Ray direction (from camera center to point)
    let camera_pos = camera_pose[3].xyz;
    let ray_dir = normalize(point_world - camera_pos);
    
    // March along ray and update voxels
    let start_dist = max(0.0, depth - truncation_distance);
    let end_dist = depth + truncation_distance;
    let steps = u32((end_dist - start_dist) / voxel_size);
    
    for (var i = 0u; i <= steps; i = i + 1u) {
        let t = start_dist + f32(i) * voxel_size;
        let voxel_pos = camera_pos + ray_dir * t;
        
        // Convert to voxel coordinates
        let vx = u32(voxel_pos.x / voxel_size);
        let vy = u32(voxel_pos.y / voxel_size);
        let vz = u32(voxel_pos.z / voxel_size);
        
        // Check bounds
        if (vx >= volume_size.x || vy >= volume_size.y || vz >= volume_size.z) {
            continue;
        }
        
        // Compute TSDF value
        let dist = length(voxel_pos - point_world);
        let sdf = select(-dist, dist, t > depth); // Negative if behind surface
        let tsdf = clamp(sdf / truncation_distance, -1.0, 1.0);
        
        // Running average update
        let idx = voxel_index(vx, vy, vz);
        let old_tsdf = tsdf_values[idx];
        let old_weight = weights[idx];
        
        let new_weight = min(old_weight + 1.0, 100.0); // Max weight = 100
        let new_tsdf = (old_tsdf * old_weight + tsdf) / new_weight;
        
        tsdf_values[idx] = new_tsdf;
        weights[idx] = new_weight;
        
        // Update color if close to surface
        if (abs(tsdf) < 0.1) {
            colors[idx] = color_image[pixel_idx];
        }
    }
}
