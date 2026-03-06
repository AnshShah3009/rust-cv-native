//! CubeCL feature detection kernels — Tier 3.
//!
//! Multi-pass algorithms ported from WGSL:
//!   `canny`   — 3-pass: gradients → NMS → hysteresis

use cubecl::calculate_cube_count_elemwise;
use cubecl::prelude::*;
use cubecl_wgpu::WgpuRuntime;

use crate::cubecl::WgpuClient;

// ---------------------------------------------------------------------------
// Shared helper: read one u8 from packed u32 array
// ---------------------------------------------------------------------------

/// Same as image::read_u8_packed — duplicated here to avoid cross-module
/// `#[cube]` fn visibility issues (CubeCL 0.9 does not re-export `#[cube]` fns).
#[cube]
fn read_byte(buf: &Array<u32>, byte_idx: usize) -> u32 {
    let word = byte_idx / 4;
    let off = ((byte_idx % 4) as u32) * 8u32;
    (buf[word] >> off) & 0xFFu32
}

// ---------------------------------------------------------------------------
// Canny edge detector — Pass 1: Sobel gradients → magnitude + direction
// ---------------------------------------------------------------------------

#[cube(launch)]
fn canny_gradients_kernel(
    input: &Array<u32>,
    mag: &mut Array<f32>,
    dir: &mut Array<u32>,
    #[comptime] img_w: usize,
    #[comptime] img_h: usize,
    #[comptime] n_pixels: usize,
) {
    let pos = ABSOLUTE_POS;
    if pos < n_pixels {
        let px = pos % img_w;
        let py = pos / img_w;

        let xm = select(px > 0usize, px - 1, 0usize);
        let xp = select(px + 1 < img_w, px + 1, img_w - 1);
        let ym = select(py > 0usize, py - 1, 0usize);
        let yp = select(py + 1 < img_h, py + 1, img_h - 1);

        let p00 = f32::cast_from(read_byte(input, ym * img_w + xm));
        let p01 = f32::cast_from(read_byte(input, ym * img_w + px));
        let p02 = f32::cast_from(read_byte(input, ym * img_w + xp));
        let p10 = f32::cast_from(read_byte(input, py * img_w + xm));
        let p12 = f32::cast_from(read_byte(input, py * img_w + xp));
        let p20 = f32::cast_from(read_byte(input, yp * img_w + xm));
        let p21 = f32::cast_from(read_byte(input, yp * img_w + px));
        let p22 = f32::cast_from(read_byte(input, yp * img_w + xp));

        let gx = (p02 + 2.0f32 * p12 + p22) - (p00 + 2.0f32 * p10 + p20);
        let gy = (p20 + 2.0f32 * p21 + p22) - (p00 + 2.0f32 * p01 + p02);

        mag[pos] = f32::sqrt(gx * gx + gy * gy);

        let abs_gx = f32::abs(gx);
        let abs_gy = f32::abs(gy);
        let tan22 = 0.414_213_56_f32;
        dir[pos] = select(
            abs_gy <= abs_gx * tan22,
            0u32,
            select(
                abs_gx <= abs_gy * tan22,
                2u32,
                select(gx * gy > 0.0f32, 1u32, 3u32),
            ),
        );
    }
}

// ---------------------------------------------------------------------------
// Pass 2: Non-maximum suppression
// ---------------------------------------------------------------------------

#[cube(launch)]
fn canny_nms_kernel(
    mag: &Array<f32>,
    dir: &Array<u32>,
    output: &mut Array<f32>,
    #[comptime] img_w: usize,
    #[comptime] img_h: usize,
    #[comptime] n_pixels: usize,
) {
    let pos = ABSOLUTE_POS;
    if pos < n_pixels {
        let px = pos % img_w;
        let py = pos / img_w;
        let d = dir[pos];
        let m = mag[pos];

        let xm = select(px > 0usize, px - 1, 0usize);
        let xp = select(px + 1 < img_w, px + 1, img_w - 1);
        let ym = select(py > 0usize, py - 1, 0usize);
        let yp = select(py + 1 < img_h, py + 1, img_h - 1);

        // Read the two neighbours in the gradient direction
        let n1 = RuntimeCell::<f32>::new(0.0f32);
        let n2 = RuntimeCell::<f32>::new(0.0f32);

        if d == 0u32 {
            n1.store(mag[py * img_w + xm]);
            n2.store(mag[py * img_w + xp]);
        } else if d == 2u32 {
            n1.store(mag[ym * img_w + px]);
            n2.store(mag[yp * img_w + px]);
        } else if d == 1u32 {
            n1.store(mag[ym * img_w + xp]);
            n2.store(mag[yp * img_w + xm]);
        } else {
            n1.store(mag[ym * img_w + xm]);
            n2.store(mag[yp * img_w + xp]);
        }

        let is_max = m >= n1.read() && m >= n2.read();
        output[pos] = select(is_max, m, 0.0f32);
    }
}

// ---------------------------------------------------------------------------
// Pass 3: Double threshold
// ---------------------------------------------------------------------------

#[cube(launch)]
fn canny_threshold_kernel(
    nms: &Array<f32>,
    edges: &mut Array<u32>, // 0=no, 1=weak, 2=strong
    #[comptime] n_pixels: usize,
    #[comptime] low_u: u32,  // low * 1000
    #[comptime] high_u: u32, // high * 1000
) {
    let pos = ABSOLUTE_POS;
    if pos < n_pixels {
        let v = nms[pos];
        let low = low_u as f32 / 1000.0f32;
        let high = high_u as f32 / 1000.0f32;
        edges[pos] = select(v >= high, 2u32, select(v >= low, 1u32, 0u32));
    }
}

// ---------------------------------------------------------------------------
// Public Canny API
// ---------------------------------------------------------------------------

/// Canny edge detection.  Returns `Vec<u8>` (255=edge, 0=no edge).
pub fn canny(
    client: &WgpuClient,
    gray: &[u8],
    img_w: usize,
    img_h: usize,
    low_threshold: f32,
    high_threshold: f32,
) -> Vec<u8> {
    let n_pixels = img_w * img_h;
    let in_padded = gray.len().div_ceil(4) * 4;
    let mut padded = gray.to_vec();
    padded.resize(in_padded, 0u8);

    let in_h = client.create_from_slice(&padded);
    let mag_h = client.empty(n_pixels * 4);
    let dir_h = client.empty(n_pixels * 4);
    let nms_h = client.empty(n_pixels * 4);
    let edges_h = client.empty(n_pixels * 4);

    let cube_dim = CubeDim::new_1d(256);
    let mk_count = || calculate_cube_count_elemwise::<WgpuRuntime>(client, n_pixels, cube_dim);

    // Pass 1
    unsafe {
        canny_gradients_kernel::launch::<WgpuRuntime>(
            client,
            mk_count(),
            cube_dim,
            ArrayArg::from_raw_parts::<u32>(&in_h, in_padded / 4, 1),
            ArrayArg::from_raw_parts::<f32>(&mag_h, n_pixels, 1),
            ArrayArg::from_raw_parts::<u32>(&dir_h, n_pixels, 1),
            img_w,
            img_h,
            n_pixels,
        )
    }
    .unwrap();

    // Pass 2
    unsafe {
        canny_nms_kernel::launch::<WgpuRuntime>(
            client,
            mk_count(),
            cube_dim,
            ArrayArg::from_raw_parts::<f32>(&mag_h, n_pixels, 1),
            ArrayArg::from_raw_parts::<u32>(&dir_h, n_pixels, 1),
            ArrayArg::from_raw_parts::<f32>(&nms_h, n_pixels, 1),
            img_w,
            img_h,
            n_pixels,
        )
    }
    .unwrap();

    // Pass 3
    let low_u = (low_threshold * 1000.0) as u32;
    let high_u = (high_threshold * 1000.0) as u32;
    unsafe {
        canny_threshold_kernel::launch::<WgpuRuntime>(
            client,
            mk_count(),
            cube_dim,
            ArrayArg::from_raw_parts::<f32>(&nms_h, n_pixels, 1),
            ArrayArg::from_raw_parts::<u32>(&edges_h, n_pixels, 1),
            n_pixels,
            low_u,
            high_u,
        )
    }
    .unwrap();

    let edge_bytes = client.read_one(edges_h);
    let edge_flags = u32::from_bytes(&edge_bytes);

    // CPU: mark strong edges; hysteresis for weak edges
    let mut result = vec![0u8; n_pixels];
    for i in 0..n_pixels {
        if edge_flags[i] == 2 {
            result[i] = 255;
        }
    }
    let mut changed = true;
    let mut iterations = 0;
    while changed && iterations < 10 {
        changed = false;
        iterations += 1;
        for i in 0..n_pixels {
            if edge_flags[i] == 1 && result[i] == 0 {
                let px = i % img_w;
                let py = i / img_w;
                let near_strong = [
                    (
                        py > 0 && px > 0,
                        py.saturating_sub(1) * img_w + px.saturating_sub(1),
                    ),
                    (py > 0, py.saturating_sub(1) * img_w + px),
                    (
                        py > 0 && px + 1 < img_w,
                        py.saturating_sub(1) * img_w + (px + 1).min(img_w - 1),
                    ),
                    (px > 0, py * img_w + px.saturating_sub(1)),
                    (px + 1 < img_w, py * img_w + (px + 1).min(img_w - 1)),
                    (
                        py + 1 < img_h && px > 0,
                        (py + 1).min(img_h - 1) * img_w + px.saturating_sub(1),
                    ),
                    (py + 1 < img_h, (py + 1).min(img_h - 1) * img_w + px),
                    (
                        py + 1 < img_h && px + 1 < img_w,
                        (py + 1).min(img_h - 1) * img_w + (px + 1).min(img_w - 1),
                    ),
                ]
                .iter()
                .any(|&(valid, idx)| valid && result[idx] == 255);
                if near_strong {
                    result[i] = 255;
                    changed = true;
                }
            }
        }
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cubecl::get_client;
    use serial_test::serial;

    #[test]
    #[serial]
    fn test_canny_uniform_no_edges() {
        let client = get_client();
        let img = vec![128u8; 100]; // 10×10 uniform
        let edges = canny(&client, &img, 10, 10, 50.0, 100.0);
        assert_eq!(edges.len(), 100);
        for &e in &edges {
            assert_eq!(e, 0, "uniform image should have no edges");
        }
    }
}
