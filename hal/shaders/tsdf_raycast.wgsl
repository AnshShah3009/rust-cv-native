// TSDF Raycasting Shader
// Casts rays into the volume to render the zero-level set.

struct Params {
    width: u32,
    height: u32,
    voxel_size: f32,
    truncation: f32,
    step_factor: f32, // Step size multiplier (usually < 1.0 * truncation)
    min_depth: f32,
    max_depth: f32,
    vol_x: u32,
    vol_y: u32,
    vol_z: u32,
}

@group(0) @binding(0) var<storage, read> tsdf_volume: array<f32>;
@group(0) @binding(1) var<storage, read_write> output_depth: array<f32>;
@group(0) @binding(2) var<storage, read_write> output_normals: array<f32>; // packed vec4 (x, y, z, 0)
@group(0) @binding(3) var<uniform> params: Params;
@group(0) @binding(4) var<uniform> camera_to_world: mat4x4<f32>;
@group(0) @binding(5) var<uniform> intrinsics_inv: vec4<f32>; // 1/fx, 1/fy, cx, cy

fn get_tsdf(p: vec3<f32>) -> f32 {
    let x = i32(floor(p.x / params.voxel_size));
    let y = i32(floor(p.y / params.voxel_size));
    let z = i32(floor(p.z / params.voxel_size));

    if (x < 0 || x >= i32(params.vol_x) || y < 0 || y >= i32(params.vol_y) || z < 0 || z >= i32(params.vol_z)) {
        return 0.0; // Empty/Unknown
    }

    let idx = u32(z) * params.vol_x * params.vol_y + u32(y) * params.vol_x + u32(x);
    return tsdf_volume[idx];
}

fn get_tsdf_interp(p: vec3<f32>) -> f32 {
    // Trilinear interpolation could go here, but NN is often sufficient for raycasting
    // if voxel size is small. For high quality, we want trilinear.
    // Simplifying to NN for initial implementation speed.
    return get_tsdf(p);
}

fn compute_normal(p: vec3<f32>) -> vec3<f32> {
    let d = params.voxel_size;
    let dx = get_tsdf_interp(p + vec3<f32>(d, 0.0, 0.0)) - get_tsdf_interp(p - vec3<f32>(d, 0.0, 0.0));
    let dy = get_tsdf_interp(p + vec3<f32>(0.0, d, 0.0)) - get_tsdf_interp(p - vec3<f32>(0.0, d, 0.0));
    let dz = get_tsdf_interp(p + vec3<f32>(0.0, 0.0, d)) - get_tsdf_interp(p - vec3<f32>(0.0, 0.0, d));
    
    let n = vec3<f32>(dx, dy, dz);
    if (length(n) > 0.0) {
        return normalize(n);
    }
    return vec3<f32>(0.0, 0.0, 0.0);
}

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let x = global_id.x;
    let y = global_id.y;

    if (x >= params.width || y >= params.height) {
        return;
    }

    // Unproject pixel to ray direction in camera space
    let u = f32(x);
    let v = f32(y);
    
    let mx = (u - params.intrinsics_inv.z) * params.intrinsics_inv.x;
    let my = (v - params.intrinsics_inv.w) * params.intrinsics_inv.y;
    
    let dir_cam = normalize(vec3<f32>(mx, my, 1.0));
    let origin_cam = vec3<f32>(0.0, 0.0, 0.0);

    // Transform to world space
    let dir_world = (camera_to_world * vec4<f32>(dir_cam, 0.0)).xyz;
    let origin_world = (camera_to_world * vec4<f32>(origin_cam, 1.0)).xyz;

    var t = params.min_depth;
    var found = false;
    var t_step = params.voxel_size * params.step_factor;

    // Ray marching
    // We look for a zero crossing from positive to negative
    var prev_tsdf = get_tsdf(origin_world + dir_world * t);
    
    // Optimization: Skip empty space if we had a hierarchy, but linear search for now.
    loop {
        if (t >= params.max_depth) { break; }
        
        let p = origin_world + dir_world * t;
        let tsdf = get_tsdf(p);
        
        if (prev_tsdf > 0.0 && tsdf < 0.0 && tsdf > -0.8) {
            // Zero crossing found!
            // Refine t with linear interpolation
            let t_surf = t - t_step * (tsdf / (tsdf - prev_tsdf));
            let p_surf = origin_world + dir_world * t_surf;
            
            output_depth[y * params.width + x] = t_surf;
            
            let normal = compute_normal(p_surf);
            let n_idx = (y * params.width + x) * 4u;
            output_normals[n_idx + 0u] = normal.x;
            output_normals[n_idx + 1u] = normal.y;
            output_normals[n_idx + 2u] = normal.z;
            output_normals[n_idx + 3u] = 0.0;
            
            found = true;
            break;
        }
        
        prev_tsdf = tsdf;
        t += t_step;
    }

    if (!found) {
        output_depth[y * params.width + x] = 0.0;
        let n_idx = (y * params.width + x) * 4u;
        output_normals[n_idx] = 0.0;
        output_normals[n_idx + 1u] = 0.0;
        output_normals[n_idx + 2u] = 0.0;
        output_normals[n_idx + 3u] = 0.0;
    }
}
