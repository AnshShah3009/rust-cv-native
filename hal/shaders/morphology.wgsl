struct Params {
    width: u32,
    height: u32,
    kw: u32,
    kh: u32,
    typ: u32, // 0: Erode, 1: Dilate
    iterations: u32,
}

@group(0) @binding(0) var<storage, read> input_data: array<u32>;
@group(0) @binding(1) var<storage, read> kernel_mask: array<u32>; // 1 bit per element or just array of u32? Array of u32 (0 or 1) is simple.
@group(0) @binding(2) var<storage, read_write> output_data: array<u32>;
@group(0) @binding(3) var<uniform> params: Params;

fn get_pixel(x: i32, y: i32) -> u32 {
    let w = i32(params.width);
    let h = i32(params.height);
    let ix = clamp(x, 0, w - 1);
    let iy = clamp(y, 0, h - 1);
    let idx = u32(iy * w + ix);
    let combined = input_data[idx / 4u];
    return (combined >> ((idx % 4u) * 8u)) & 0xFFu;
}

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let u32_x = global_id.x;
    let y = i32(global_id.y);

    if (u32_x * 4u >= params.width || u32(y) >= params.height) {
        return;
    }

    var res_combined = 0u;
    let cx = i32(params.kw / 2u);
    let cy = i32(params.kh / 2u);

    for (var i = 0u; i < 4u; i++) {
        let x = i32(u32_x * 4u + i);
        if (u32(x) >= params.width) { break; }

        var val: u32;
        if (params.typ == 0u) { val = 255u; } else { val = 0u; }

        for (var ky = 0u; ky < params.kh; ky++) {
            for (var kx = 0u; kx < params.kw; kx++) {
                // If kernel has 1 at this position
                if (kernel_mask[ky * params.kw + kx] != 0u) {
                    let px = x + i32(kx) - cx;
                    let py = y + i32(ky) - cy;
                    let p = get_pixel(px, py);
                    
                    if (params.typ == 0u) { // Erode
                        val = min(val, p);
                    } else { // Dilate
                        val = max(val, p);
                    }
                }
            }
        }
        res_combined = res_combined | ((val & 0xFFu) << (i * 8u));
    }

    output_data[u32_x + u32(y) * ((params.width + 3u) / 4u)] = res_combined;
}
