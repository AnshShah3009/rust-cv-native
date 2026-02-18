// TSDF Integration Shader (Voxel-centric)
// Each thread processes one voxel, projecting it into the depth map.

@group(0) @binding(0) var<storage, read> depth_image: array<f32>;
@group(0) @binding(1) var<storage, read> color_image: array<u32>; // Packed RGBA
@group(0) @binding(2) var<storage, read_write> tsdf_values: array<f32>;
@group(0) @binding(3) var<storage, read_write> weights: array<f32>;
@group(0) @binding(4) var<storage, read_write> colors: array<u32>;

@group(1) @binding(0) var<uniform> world_to_camera: mat4x4<f32>;
@group(1) @binding(1) var<uniform> intrinsics: vec4<f32>; // fx, fy, cx, cy
@group(1) @binding(2) var<uniform> image_size: vec2<u32>; // width, height
@group(1) @binding(3) var<uniform> voxel_size: f32;
@group(1) @binding(4) var<uniform> truncation_distance: f32;
@group(1) @binding(5) var<uniform> volume_size: vec3<u32>; // vx, vy, vz

// Project 3D point in camera space to 2D pixel coordinates
fn project(p: vec3<f32>) -> vec2<f32> {
    let fx = intrinsics.x;
    let fy = intrinsics.y;
    let cx = intrinsics.z;
    let cy = intrinsics.w;
    
    let u = (p.x * fx / p.z) + cx;
    let v = (p.y * fy / p.z) + cy;
    
    return vec2<f32>(u, v);
}

@compute @workgroup_size(8, 8, 8)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let vx = global_id.x;
    let vy = global_id.y;
    let vz = global_id.z;
    
    if (vx >= volume_size.x || vy >= volume_size.y || vz >= volume_size.z) {
        return;
    }
    
    // Voxel position in world space (assuming origin at 0,0,0)
    let p_world = vec3<f32>(
        (f32(vx) + 0.5) * voxel_size,
        (f32(vy) + 0.5) * voxel_size,
        (f32(vz) + 0.5) * voxel_size
    );
    
    // Transform to camera space
    let p_camera = (world_to_camera * vec4<f32>(p_world, 1.0)).xyz;
    
    // Check if point is in front of camera
    if (p_camera.z <= 0.0) {
        return;
    }
    
    // Project to image
    let uv = project(p_camera);
    let u = u32(round(uv.x));
    let v = u32(round(uv.y));
    
    if (u >= image_size.x || v >= image_size.y) {
        return;
    }
    
    let pixel_idx = v * image_size.x + u;
    let depth = depth_image[pixel_idx];
    
    if (depth <= 0.0 || depth > 10.0) {
        return;
    }
    
    // Signed distance
    let dist = depth - p_camera.z;
    
    // If voxel is behind surface further than truncation, skip
    if (dist < -truncation_distance) {
        return;
    }
    
    // Compute TSDF
    let tsdf = clamp(dist / truncation_distance, -1.0, 1.0);
    
    // Weighted update
    let idx = vz * volume_size.x * volume_size.y + vy * volume_size.x + vx;
    let old_tsdf = tsdf_values[idx];
    let old_weight = weights[idx];
    
    // Standard TSDF weight: 1.0 per frame, capped at 50
    let weight = 1.0;
    let new_weight = min(old_weight + weight, 50.0);
    let new_tsdf = (old_tsdf * old_weight + tsdf * weight) / new_weight;
    
    tsdf_values[idx] = new_tsdf;
    weights[idx] = new_weight;
    
    // Color update if close to surface
    if (dist > -truncation_distance && dist < truncation_distance * 0.5) {
        colors[idx] = color_image[pixel_idx];
    }
}
