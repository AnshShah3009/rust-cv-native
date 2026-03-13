//! CubeCL feature matching and descriptor kernels.
//!
//! Tier 1:
//!   `hamming_match`   — brute-force nearest-neighbour on binary descriptors
//!   `brief`           — rotated BRIEF binary descriptor computation
//!   `nms`             — non-maximum suppression over score maps

use cubecl::calculate_cube_count_elemwise;
use cubecl::prelude::*;
use cubecl_wgpu::WgpuRuntime;

use crate::cubecl::WgpuClient;

// ---------------------------------------------------------------------------
// Hamming distance brute-force match
// ---------------------------------------------------------------------------

/// Brute-force nearest-neighbour match on binary descriptors.
///
/// `desc_u32s`: words per descriptor (e.g. 8 for 256-bit ORB).
#[cube(launch)]
fn hamming_match_kernel(
    query_desc: &Array<u32>,
    train_desc: &Array<u32>,
    matches: &mut Array<u32>,
    distances: &mut Array<u32>,
    #[comptime] n_query: usize,
    #[comptime] n_train: usize,
    #[comptime] desc_u32s: usize,
) {
    let qi = ABSOLUTE_POS;
    if qi < n_query {
        let q_base = qi * desc_u32s;

        // Use RuntimeCell to hold mutable state across loop iterations
        let best_dist = RuntimeCell::<u32>::new(0xFFFF_FFFFu32);
        let best_idx = RuntimeCell::<u32>::new(0u32);

        for ti in 0usize..n_train {
            let t_base = ti * desc_u32s;
            let mut dist = 0u32;
            for w in 0usize..desc_u32s {
                let xored = query_desc[q_base + w] ^ train_desc[t_base + w];
                dist += xored.count_ones();
            }
            if dist < best_dist.read() {
                best_dist.store(dist);
                best_idx.store(u32::cast_from(ti));
            }
        }

        matches[qi] = best_idx.read();
        distances[qi] = best_dist.read();
    }
}

/// Brute-force Hamming nearest-neighbour match.
/// Returns `(match_indices, hamming_distances)`.
pub fn hamming_match(
    client: &WgpuClient,
    query: &[u32],
    train: &[u32],
    desc_u32s: usize,
) -> (Vec<u32>, Vec<u32>) {
    let n_query = query.len() / desc_u32s;
    let n_train = train.len() / desc_u32s;

    let q_handle = client.create_from_slice(u32::as_bytes(query));
    let t_handle = client.create_from_slice(u32::as_bytes(train));
    let m_handle = client.empty(n_query * 4);
    let d_handle = client.empty(n_query * 4);

    let cube_dim = CubeDim::new_1d(64);
    let cube_count = calculate_cube_count_elemwise::<WgpuRuntime>(client, n_query, cube_dim);

    macro_rules! launch_dw {
        ($dw:expr) => {
            unsafe {
                hamming_match_kernel::launch::<WgpuRuntime>(
                    client,
                    cube_count,
                    cube_dim,
                    ArrayArg::from_raw_parts::<u32>(&q_handle, query.len(), 1),
                    ArrayArg::from_raw_parts::<u32>(&t_handle, train.len(), 1),
                    ArrayArg::from_raw_parts::<u32>(&m_handle, n_query, 1),
                    ArrayArg::from_raw_parts::<u32>(&d_handle, n_query, 1),
                    n_query,
                    n_train,
                    $dw,
                )
            }
            .unwrap()
        };
    }
    match desc_u32s {
        4 => launch_dw!(4),
        8 => launch_dw!(8),
        16 => launch_dw!(16),
        _ => launch_dw!(8),
    }

    let mb = client.read_one(m_handle);
    let db = client.read_one(d_handle);
    (u32::from_bytes(&mb).to_vec(), u32::from_bytes(&db).to_vec())
}

// ---------------------------------------------------------------------------
// Rotated BRIEF descriptor computation
// ---------------------------------------------------------------------------

#[cube]
fn read_img_u8(image: &Array<u32>, bx: usize, by: usize, img_w: usize) -> f32 {
    let idx = by * img_w + bx;
    let word = idx / 4;
    let off = ((idx % 4) as u32) * 8u32;
    f32::cast_from((image[word] >> off) & 0xFFu32)
}

#[cube]
fn bilinear_u8(image: &Array<u32>, xf: f32, yf: f32, img_w: usize, img_h: usize) -> f32 {
    let x0 = f32::floor(xf) as usize;
    let y0 = f32::floor(yf) as usize;
    let x1 = (x0 + 1).min(img_w - 1);
    let y1 = (y0 + 1).min(img_h - 1);
    let dx = xf - f32::floor(xf);
    let dy = yf - f32::floor(yf);

    read_img_u8(image, x0, y0, img_w) * (1.0f32 - dx) * (1.0f32 - dy)
        + read_img_u8(image, x1, y0, img_w) * dx * (1.0f32 - dy)
        + read_img_u8(image, x0, y1, img_w) * (1.0f32 - dx) * dy
        + read_img_u8(image, x1, y1, img_w) * dx * dy
}

/// Rotated BRIEF — compute 256-bit descriptors for `n_kp` keypoints.
///
/// `kp_data`  — 8 × f32 per keypoint: [x, y, size, angle, response, octave, class_id, pad]
/// `patterns` — 4 × f32 per BRIEF pair: [x1, y1, x2, y2]; 256 pairs = 1 024 floats
/// `desc_out` — 8 × u32 per keypoint (256 bits)
#[cube(launch)]
fn brief_kernel(
    image: &Array<u32>,
    kp_data: &Array<f32>,
    patterns: &Array<f32>,
    desc_out: &mut Array<u32>,
    #[comptime] img_w: usize,
    #[comptime] img_h: usize,
    #[comptime] n_kp: usize,
) {
    let ki = ABSOLUTE_POS;
    if ki < n_kp {
        let kp_base = ki * 8;
        let kp_x = kp_data[kp_base];
        let kp_y = kp_data[kp_base + 1];
        let angle = kp_data[kp_base + 3];
        let cos_a = f32::cos(angle);
        let sin_a = f32::sin(angle);

        let desc_base = ki * 8;

        for chunk in 0usize..8usize {
            let mut word = 0u32;
            for bit in 0usize..32usize {
                let pair_idx = chunk * 32 + bit;
                let p_base = pair_idx * 4;
                let px1 = patterns[p_base];
                let py1 = patterns[p_base + 1];
                let px2 = patterns[p_base + 2];
                let py2 = patterns[p_base + 3];

                let x1r = (px1 * cos_a - py1 * sin_a + kp_x).clamp(0.0f32, (img_w - 1) as f32);
                let y1r = (px1 * sin_a + py1 * cos_a + kp_y).clamp(0.0f32, (img_h - 1) as f32);
                let x2r = (px2 * cos_a - py2 * sin_a + kp_x).clamp(0.0f32, (img_w - 1) as f32);
                let y2r = (px2 * sin_a + py2 * cos_a + kp_y).clamp(0.0f32, (img_h - 1) as f32);

                let v1 = bilinear_u8(image, x1r, y1r, img_w, img_h);
                let v2 = bilinear_u8(image, x2r, y2r, img_w, img_h);

                if v1 < v2 {
                    word |= 1u32 << u32::cast_from(bit);
                }
            }
            desc_out[desc_base + chunk] = word;
        }
    }
}

/// Compute rotated BRIEF descriptors.
/// Returns `Vec<u32>` of length `8 * n_keypoints` (256 bits per keypoint).
pub fn brief(
    client: &WgpuClient,
    image: &[u8],
    img_w: usize,
    img_h: usize,
    keypoints: &[f32],
    patterns: &[f32],
) -> Vec<u32> {
    let n_kp = keypoints.len() / 8;
    let img_bytes = image.len();
    let img_padded = img_bytes.div_ceil(4) * 4;
    let mut padded_img = image.to_vec();
    padded_img.resize(img_padded, 0u8);

    let img_handle = client.create_from_slice(&padded_img);
    let kp_handle = client.create_from_slice(f32::as_bytes(keypoints));
    let pat_handle = client.create_from_slice(f32::as_bytes(patterns));
    let out_handle = client.empty(n_kp * 8 * 4);

    let cube_dim = CubeDim::new_1d(256);
    let cube_count = calculate_cube_count_elemwise::<WgpuRuntime>(client, n_kp, cube_dim);

    unsafe {
        brief_kernel::launch::<WgpuRuntime>(
            client,
            cube_count,
            cube_dim,
            ArrayArg::from_raw_parts::<u32>(&img_handle, img_padded / 4, 1),
            ArrayArg::from_raw_parts::<f32>(&kp_handle, keypoints.len(), 1),
            ArrayArg::from_raw_parts::<f32>(&pat_handle, patterns.len(), 1),
            ArrayArg::from_raw_parts::<u32>(&out_handle, n_kp * 8, 1),
            img_w,
            img_h,
            n_kp,
        )
    }
    .unwrap();

    let result = client.read_one(out_handle);
    u32::from_bytes(&result).to_vec()
}

// ---------------------------------------------------------------------------
// Non-maximum suppression over a score map
// ---------------------------------------------------------------------------

#[cube(launch)]
fn nms_kernel(
    scores: &Array<f32>,
    out: &mut Array<u32>,
    #[comptime] w: usize,
    #[comptime] h: usize,
    #[comptime] radius: usize,
) {
    let pos = ABSOLUTE_POS;
    if pos < w * h {
        let px = pos % w;
        let py = pos / w;
        let center = scores[pos];

        let x0 = select(px >= radius, px - radius, 0usize);
        let y0 = select(py >= radius, py - radius, 0usize);
        let x1 = (px + radius + 1).min(w);
        let y1 = (py + radius + 1).min(h);

        let mut is_max = 1u32;
        for ny in y0..y1 {
            for nx in x0..x1 {
                if (ny != py || nx != px) && scores[ny * w + nx] > center {
                    is_max = 0u32;
                }
            }
        }
        out[pos] = is_max;
    }
}

/// Suppress non-maxima in a score map.
/// Returns a `Vec<u32>` (1=keep, 0=suppress).
pub fn nms(client: &WgpuClient, scores: &[f32], w: usize, h: usize, radius: usize) -> Vec<u32> {
    let n = w * h;
    let s_handle = client.create_from_slice(f32::as_bytes(scores));
    let out_handle = client.empty(n * 4);

    let cube_dim = CubeDim::new_1d(256);
    let cube_count = calculate_cube_count_elemwise::<WgpuRuntime>(client, n, cube_dim);

    macro_rules! launch_r {
        ($r:expr) => {
            unsafe {
                nms_kernel::launch::<WgpuRuntime>(
                    client,
                    cube_count,
                    cube_dim,
                    ArrayArg::from_raw_parts::<f32>(&s_handle, n, 1),
                    ArrayArg::from_raw_parts::<u32>(&out_handle, n, 1),
                    w,
                    h,
                    $r,
                )
            }
            .unwrap()
        };
    }
    match radius {
        1 => launch_r!(1),
        2 => launch_r!(2),
        3 => launch_r!(3),
        _ => launch_r!(5),
    }

    let result = client.read_one(out_handle);
    u32::from_bytes(&result).to_vec()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cubecl::get_client;
    use serial_test::serial;

    #[test]
    #[serial]
    fn test_hamming_self_match() {
        let Some(client) = get_client() else {
            eprintln!("GPU unavailable, skipping test_hamming_self_match");
            return;
        };
        let descs: Vec<u32> = vec![
            0xDEAD_BEEF,
            0x1234_5678,
            0,
            0,
            0,
            0,
            0,
            0, // desc 0
            0xCAFE_BABE,
            0xABCD_EF01,
            0,
            0,
            0,
            0,
            0,
            0, // desc 1
        ];
        let (matches, dists) = hamming_match(&client, &descs, &descs, 8);
        assert_eq!(dists[0], 0);
        assert_eq!(dists[1], 0);
        assert_eq!(matches[0], 0);
        assert_eq!(matches[1], 1);
    }

    #[test]
    #[serial]
    fn test_nms_single_peak() {
        let Some(client) = get_client() else {
            eprintln!("GPU unavailable, skipping test_nms_single_peak");
            return;
        };
        let scores = vec![1.0f32, 1.0, 1.0, 1.0, 5.0, 1.0, 1.0, 1.0, 1.0];
        let result = nms(&client, &scores, 3, 3, 1);
        assert_eq!(result[4], 1, "centre should be kept");
        for i in [0, 1, 2, 3, 5, 6, 7, 8] {
            assert_eq!(result[i], 0, "index {i} should be suppressed");
        }
    }
}
