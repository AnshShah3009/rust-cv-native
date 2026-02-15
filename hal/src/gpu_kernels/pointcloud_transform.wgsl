// Point Cloud Transform Shader
// Transforms 3D points using a 4x4 transformation matrix

@group(0) @binding(0) var<storage, read> points_in: array<vec3<f32>>;
@group(0) @binding(1) var<storage, read_write> points_out: array<vec3<f32>>;
@group(0) @binding(2) var<uniform> transform: mat4x4<f32>;
@group(0) @binding(3) var<uniform> num_points: u32;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    
    if (idx >= num_points) {
        return;
    }
    
    let p = points_in[idx];
    let p_h = vec4<f32>(p.x, p.y, p.z, 1.0);
    let transformed = transform * p_h;
    
    points_out[idx] = transformed.xyz;
}
