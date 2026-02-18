struct Params {
    src_w: u32,
    src_h: u32,
    dst_w: u32,
    dst_h: u32,
    channels: u32,
}

@group(0) @binding(0) var<storage, read> input_data: array<u32>;
@group(0) @binding(1) var<storage, read_write> output_data: array<u32>;
@group(0) @binding(2) var<uniform> params: Params;

fn get_u8(idx: u32) -> u32 {
    let u32_idx = idx / 4u;
    let shift = (idx % 4u) * 8u;
    return (input_data[u32_idx] >> shift) & 0xFFu;
}

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let x_u32 = global_id.x; // Thread processes 4 horizontal pixels
    let y = global_id.y;
    
    if (x_u32 * 4u >= params.dst_w || y >= params.dst_h) {
        return;
    }

    let scale_x = f32(params.src_w) / f32(params.dst_w);
    let scale_y = f32(params.src_h) / f32(params.dst_h);

    var res_combined = 0u;
    for (var i = 0u; i < 4u; i++) {
        let x = x_u32 * 4u + i;
        if (x >= params.dst_w) { break; }

        let src_x = u32(f32(x) * scale_x);
        let src_y = u32(f32(y) * scale_y);
        
        let val = get_u8(src_y * params.src_w + src_x);
        res_combined = res_combined | ((val & 0xFFu) << (i * 8u));
    }

    output_data[y * ((params.dst_w + 3u) / 4u) + x_u32] = res_combined;
}
