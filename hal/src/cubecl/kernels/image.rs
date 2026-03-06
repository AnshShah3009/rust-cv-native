//! CubeCL image processing kernels.
//!
//! CubeCL DSL constraints applied throughout:
//! - No closures inside `#[cube]` fns.
//! - No `return` — use `terminate!()` for early exit.
//! - No `continue` — restructure loops.
//! - Output bytes as `f32` or `u32` per-value to avoid data races on
//!   packed-u32 writes; the host wrapper converts back to `Vec<u8>`.

use cubecl::calculate_cube_count_elemwise;
use cubecl::prelude::*;
use cubecl_wgpu::WgpuRuntime;

use crate::cubecl::WgpuClient;

// ---------------------------------------------------------------------------
// Helper: read one packed u8 byte from a u32 array (all `#[cube]` callers)
// ---------------------------------------------------------------------------
#[cube]
fn read_u8_packed(buf: &Array<u32>, byte_idx: usize) -> u32 {
    let word = byte_idx / 4;
    let off = ((byte_idx % 4) as u32) * 8u32;
    (buf[word] >> off) & 0xFFu32
}

// ---------------------------------------------------------------------------
// Threshold — handles complete u32 words per thread (4 pixels each)
// ---------------------------------------------------------------------------

/// mode: 0=Binary, 1=BinaryInv, 2=Trunc, 3=ToZero, 4=ToZeroInv
#[cube(launch)]
fn threshold_kernel(
    input: &Array<u32>,
    output: &mut Array<u32>,
    #[comptime] thresh: u32,
    #[comptime] max_val: u32,
    #[comptime] mode: u32,
    #[comptime] n_u32s: usize,
    #[comptime] n_bytes: usize,
) {
    let pos = ABSOLUTE_POS;
    if pos < n_u32s {
        let packed = input[pos];
        let mut out = 0u32;

        for lane in 0usize..4usize {
            let byte_idx = pos * 4 + lane;
            if byte_idx < n_bytes {
                let shift = u32::cast_from(lane) * 8u32;
                let b = (packed >> shift) & 0xFFu32;
                let result: u32 = if mode == 0u32 {
                    select(b > thresh, max_val, 0u32)
                } else if mode == 1u32 {
                    select(b > thresh, 0u32, max_val)
                } else if mode == 2u32 {
                    select(b > thresh, thresh, b)
                } else if mode == 3u32 {
                    select(b > thresh, b, 0u32)
                } else {
                    select(b > thresh, 0u32, b)
                };
                out |= result << shift;
            }
        }
        output[pos] = out;
    }
}

/// Apply a threshold to a u8 image byte slice.
/// `mode`: 0=Binary, 1=BinaryInv, 2=Trunc, 3=ToZero, 4=ToZeroInv
pub fn threshold(
    client: &WgpuClient,
    input: &[u8],
    thresh: u8,
    max_val: u8,
    mode: u32,
) -> Vec<u8> {
    let len = input.len();
    let padded_len = len.div_ceil(4) * 4;
    let mut padded = input.to_vec();
    padded.resize(padded_len, 0u8);
    let n_u32s = padded_len / 4;

    let in_handle = client.create_from_slice(&padded);
    let out_handle = client.empty(padded_len);

    let cube_dim = CubeDim::new_1d(256);
    let cube_count = calculate_cube_count_elemwise::<WgpuRuntime>(client, n_u32s, cube_dim);

    macro_rules! launch_mode {
        ($m:expr) => {
            unsafe {
                threshold_kernel::launch::<WgpuRuntime>(
                    client,
                    cube_count,
                    cube_dim,
                    ArrayArg::from_raw_parts::<u32>(&in_handle, n_u32s, 1),
                    ArrayArg::from_raw_parts::<u32>(&out_handle, n_u32s, 1),
                    thresh as u32,
                    max_val as u32,
                    $m,
                    n_u32s,
                    len,
                )
            }
            .unwrap()
        };
    }
    match mode {
        0 => launch_mode!(0),
        1 => launch_mode!(1),
        2 => launch_mode!(2),
        3 => launch_mode!(3),
        _ => launch_mode!(4),
    }

    let result = client.read_one(out_handle);
    result[..len].to_vec()
}

// ---------------------------------------------------------------------------
// Color conversion — grayscale ↔ RGB
// ---------------------------------------------------------------------------

/// Convert packed RGB bytes to grayscale (ITU-R 601 luminance).
/// Output: one f32 per pixel (in range 0-255).
#[cube(launch)]
fn rgb_to_gray_kernel(
    input: &Array<u32>,
    output: &mut Array<f32>,
    #[comptime] n_pixels: usize,
) {
    let pos = ABSOLUTE_POS;
    if pos < n_pixels {
        let base = pos * 3;
        let r = read_u8_packed(input, base);
        let g = read_u8_packed(input, base + 1);
        let b = read_u8_packed(input, base + 2);
        output[pos] = (r * 299u32 + g * 587u32 + b * 114u32) as f32 / 1000.0f32;
    }
}

/// Convert packed grayscale bytes to RGB.
/// Output: three f32 per pixel [R, G, B] (in range 0-255).
#[cube(launch)]
fn gray_to_rgb_kernel(
    input: &Array<u32>,
    output: &mut Array<f32>,
    #[comptime] n_pixels: usize,
) {
    let pos = ABSOLUTE_POS;
    if pos < n_pixels {
        let gray = read_u8_packed(input, pos) as f32;
        let base = pos * 3;
        output[base] = gray;
        output[base + 1] = gray;
        output[base + 2] = gray;
    }
}

/// Convert RGB bytes to grayscale. Returns `Vec<u8>` (one byte per pixel).
pub fn rgb_to_grayscale(client: &WgpuClient, rgb: &[u8], n_pixels: usize) -> Vec<u8> {
    let in_padded = rgb.len().div_ceil(4) * 4;
    let mut padded_in = rgb.to_vec();
    padded_in.resize(in_padded, 0u8);

    let in_handle = client.create_from_slice(&padded_in);
    let out_handle = client.empty(n_pixels * 4); // f32 per pixel

    let cube_dim = CubeDim::new_1d(256);
    let cube_count = calculate_cube_count_elemwise::<WgpuRuntime>(client, n_pixels, cube_dim);
    unsafe {
        rgb_to_gray_kernel::launch::<WgpuRuntime>(
            client,
            cube_count,
            cube_dim,
            ArrayArg::from_raw_parts::<u32>(&in_handle, in_padded / 4, 1),
            ArrayArg::from_raw_parts::<f32>(&out_handle, n_pixels, 1),
            n_pixels,
        )
    }
    .unwrap();

    let result = client.read_one(out_handle);
    let floats = f32::from_bytes(&result);
    floats.iter().map(|&v| v.round() as u8).collect()
}

/// Convert grayscale bytes to RGB. Returns `Vec<u8>` (3 bytes per pixel).
pub fn grayscale_to_rgb(client: &WgpuClient, gray: &[u8], n_pixels: usize) -> Vec<u8> {
    let in_padded = gray.len().div_ceil(4) * 4;
    let mut padded_in = gray.to_vec();
    padded_in.resize(in_padded, 0u8);

    let in_handle = client.create_from_slice(&padded_in);
    let out_handle = client.empty(n_pixels * 3 * 4); // 3 f32 per pixel

    let cube_dim = CubeDim::new_1d(256);
    let cube_count = calculate_cube_count_elemwise::<WgpuRuntime>(client, n_pixels, cube_dim);
    unsafe {
        gray_to_rgb_kernel::launch::<WgpuRuntime>(
            client,
            cube_count,
            cube_dim,
            ArrayArg::from_raw_parts::<u32>(&in_handle, in_padded / 4, 1),
            ArrayArg::from_raw_parts::<f32>(&out_handle, n_pixels * 3, 1),
            n_pixels,
        )
    }
    .unwrap();

    let result = client.read_one(out_handle);
    let floats = f32::from_bytes(&result);
    floats.iter().map(|&v| v.round() as u8).collect()
}

// ---------------------------------------------------------------------------
// Resize — bilinear interpolation, per-pixel, all channel counts
// Output: f32 per channel per pixel (avoids packed-u32 race conditions)
// ---------------------------------------------------------------------------

#[cube(launch)]
fn resize_bilinear_kernel(
    input: &Array<u32>,
    output: &mut Array<f32>,
    #[comptime] src_w: usize,
    #[comptime] src_h: usize,
    #[comptime] dst_w: usize,
    #[comptime] dst_h: usize,
    #[comptime] channels: usize,
) {
    let pos = ABSOLUTE_POS;
    if pos < dst_w * dst_h {
        let dst_x = pos % dst_w;
        let dst_y = pos / dst_w;

        let scale_x = src_w as f32 / dst_w as f32;
        let scale_y = src_h as f32 / dst_h as f32;
        let src_xf = (dst_x as f32 + 0.5f32) * scale_x - 0.5f32;
        let src_yf = (dst_y as f32 + 0.5f32) * scale_y - 0.5f32;

        let x0 = f32::max(src_xf, 0.0f32) as usize;
        let y0 = f32::max(src_yf, 0.0f32) as usize;
        let x1 = (x0 + 1).min(src_w - 1);
        let y1 = (y0 + 1).min(src_h - 1);
        let fx = src_xf - f32::floor(src_xf);
        let fy = src_yf - f32::floor(src_yf);

        for ch in 0usize..channels {
            let b00 = read_u8_packed(input, (y0 * src_w + x0) * channels + ch) as f32;
            let b10 = read_u8_packed(input, (y0 * src_w + x1) * channels + ch) as f32;
            let b01 = read_u8_packed(input, (y1 * src_w + x0) * channels + ch) as f32;
            let b11 = read_u8_packed(input, (y1 * src_w + x1) * channels + ch) as f32;

            let v = b00 * (1.0f32 - fx) * (1.0f32 - fy)
                + b10 * fx * (1.0f32 - fy)
                + b01 * (1.0f32 - fx) * fy
                + b11 * fx * fy;
            output[pos * channels + ch] = v;
        }
    }
}

/// Resize image using bilinear interpolation (any channel count).
pub fn resize_bilinear(
    client: &WgpuClient,
    src: &[u8],
    src_w: usize,
    src_h: usize,
    dst_w: usize,
    dst_h: usize,
    channels: usize,
) -> Vec<u8> {
    let src_bytes = src_w * src_h * channels;
    let dst_pixels = dst_w * dst_h;
    let dst_vals = dst_pixels * channels;
    let in_padded = src_bytes.div_ceil(4) * 4;
    let mut padded_in = src.to_vec();
    padded_in.resize(in_padded, 0u8);

    let in_handle = client.create_from_slice(&padded_in);
    let out_handle = client.empty(dst_vals * 4); // f32 per channel per pixel

    let cube_dim = CubeDim::new_1d(256);
    let cube_count = calculate_cube_count_elemwise::<WgpuRuntime>(client, dst_pixels, cube_dim);

    macro_rules! launch_ch {
        ($ch:expr) => {
            unsafe {
                resize_bilinear_kernel::launch::<WgpuRuntime>(
                    client,
                    cube_count,
                    cube_dim,
                    ArrayArg::from_raw_parts::<u32>(&in_handle, in_padded / 4, 1),
                    ArrayArg::from_raw_parts::<f32>(&out_handle, dst_vals, 1),
                    src_w,
                    src_h,
                    dst_w,
                    dst_h,
                    $ch,
                )
            }
            .unwrap()
        };
    }
    match channels {
        1 => launch_ch!(1),
        2 => launch_ch!(2),
        3 => launch_ch!(3),
        _ => launch_ch!(4),
    }

    let result = client.read_one(out_handle);
    let floats = f32::from_bytes(&result);
    floats.iter().map(|&v| v.round().clamp(0.0, 255.0) as u8).collect()
}

// ---------------------------------------------------------------------------
// Sobel edge detection — all channel counts
// Output: f32 per channel per pixel
// ---------------------------------------------------------------------------

#[cube(launch)]
fn sobel_kernel(
    input: &Array<u32>,
    output: &mut Array<f32>,
    #[comptime] w: usize,
    #[comptime] h: usize,
    #[comptime] channels: usize,
) {
    let pos = ABSOLUTE_POS;
    if pos < w * h {
        let px = pos % w;
        let py = pos / w;

        let xm = select(px > 0usize, px - 1usize, 0usize);
        let xp = select(px + 1 < w, px + 1, w - 1usize);
        let ym = select(py > 0usize, py - 1usize, 0usize);
        let yp = select(py + 1 < h, py + 1, h - 1usize);

        for ch in 0usize..channels {
            let p_mm = i32::cast_from(read_u8_packed(input, (ym * w + xm) * channels + ch));
            let p_0m = i32::cast_from(read_u8_packed(input, (ym * w + px) * channels + ch));
            let p_pm = i32::cast_from(read_u8_packed(input, (ym * w + xp) * channels + ch));
            let p_m0 = i32::cast_from(read_u8_packed(input, (py * w + xm) * channels + ch));
            let p_p0 = i32::cast_from(read_u8_packed(input, (py * w + xp) * channels + ch));
            let p_mp = i32::cast_from(read_u8_packed(input, (yp * w + xm) * channels + ch));
            let p_0p = i32::cast_from(read_u8_packed(input, (yp * w + px) * channels + ch));
            let p_pp = i32::cast_from(read_u8_packed(input, (yp * w + xp) * channels + ch));

            let gx = -p_mm + p_pm - 2i32 * p_m0 + 2i32 * p_p0 - p_mp + p_pp;
            let gy = -p_mm - 2i32 * p_0m - p_pm + p_mp + 2i32 * p_0p + p_pp;
            let mag = f32::sqrt(f32::cast_from(gx * gx + gy * gy));
            output[pos * channels + ch] = mag.min(255.0f32);
        }
    }
}

/// Compute Sobel edges (any channel count). Returns `Vec<u8>`.
pub fn sobel(
    client: &WgpuClient,
    src: &[u8],
    width: usize,
    height: usize,
    channels: usize,
) -> Vec<u8> {
    let src_bytes = width * height * channels;
    let n_pixels = width * height;
    let n_vals = n_pixels * channels;
    let in_padded = src_bytes.div_ceil(4) * 4;
    let mut padded_in = src.to_vec();
    padded_in.resize(in_padded, 0u8);

    let in_handle = client.create_from_slice(&padded_in);
    let out_handle = client.empty(n_vals * 4); // f32 per val

    let cube_dim = CubeDim::new_1d(256);
    let cube_count = calculate_cube_count_elemwise::<WgpuRuntime>(client, n_pixels, cube_dim);

    macro_rules! launch_ch {
        ($ch:expr) => {
            unsafe {
                sobel_kernel::launch::<WgpuRuntime>(
                    client,
                    cube_count,
                    cube_dim,
                    ArrayArg::from_raw_parts::<u32>(&in_handle, in_padded / 4, 1),
                    ArrayArg::from_raw_parts::<f32>(&out_handle, n_vals, 1),
                    width,
                    height,
                    $ch,
                )
            }
            .unwrap()
        };
    }
    match channels {
        1 => launch_ch!(1),
        2 => launch_ch!(2),
        3 => launch_ch!(3),
        _ => launch_ch!(4),
    }

    let result = client.read_one(out_handle);
    let floats = f32::from_bytes(&result);
    floats.iter().map(|&v| v.round().clamp(0.0, 255.0) as u8).collect()
}

// ---------------------------------------------------------------------------
// Affine warp — inverse mapping, bilinear, all channel counts
// ---------------------------------------------------------------------------

/// `matrix`: [a, b, tx, c, d, ty] — maps destination pixel → source pixel.
#[cube(launch)]
fn warp_affine_kernel(
    input: &Array<u32>,
    output: &mut Array<f32>,
    matrix: &Array<f32>,
    #[comptime] src_w: usize,
    #[comptime] src_h: usize,
    #[comptime] dst_w: usize,
    #[comptime] dst_h: usize,
    #[comptime] channels: usize,
) {
    let pos = ABSOLUTE_POS;
    if pos < dst_w * dst_h {
        let dst_x = (pos % dst_w) as f32;
        let dst_y = (pos / dst_w) as f32;

        let a = matrix[0];
        let b = matrix[1];
        let tx = matrix[2];
        let c = matrix[3];
        let d = matrix[4];
        let ty = matrix[5];

        let src_xf = a * dst_x + b * dst_y + tx;
        let src_yf = c * dst_x + d * dst_y + ty;

        let x0f = f32::floor(src_xf);
        let y0f = f32::floor(src_yf);
        let fx = src_xf - x0f;
        let fy = src_yf - y0f;

        // Clamp source coordinates to valid range
        let cx0f = f32::max(x0f, 0.0f32).min((src_w - 1) as f32);
        let cx1f = f32::max(x0f + 1.0f32, 0.0f32).min((src_w - 1) as f32);
        let cy0f = f32::max(y0f, 0.0f32).min((src_h - 1) as f32);
        let cy1f = f32::max(y0f + 1.0f32, 0.0f32).min((src_h - 1) as f32);
        let cx0 = cx0f as usize;
        let cx1 = cx1f as usize;
        let cy0 = cy0f as usize;
        let cy1 = cy1f as usize;

        for ch in 0usize..channels {
            let b00 = read_u8_packed(input, (cy0 * src_w + cx0) * channels + ch) as f32;
            let b10 = read_u8_packed(input, (cy0 * src_w + cx1) * channels + ch) as f32;
            let b01 = read_u8_packed(input, (cy1 * src_w + cx0) * channels + ch) as f32;
            let b11 = read_u8_packed(input, (cy1 * src_w + cx1) * channels + ch) as f32;

            let v = b00 * (1.0f32 - fx) * (1.0f32 - fy)
                + b10 * fx * (1.0f32 - fy)
                + b01 * (1.0f32 - fx) * fy
                + b11 * fx * fy;
            output[pos * channels + ch] = v;
        }
    }
}

/// Apply a 2×3 inverse affine warp (any channel count). Returns `Vec<u8>`.
pub fn warp_affine(
    client: &WgpuClient,
    src: &[u8],
    src_w: usize,
    src_h: usize,
    dst_w: usize,
    dst_h: usize,
    channels: usize,
    matrix: &[f32; 6],
) -> Vec<u8> {
    let src_bytes = src_w * src_h * channels;
    let dst_pixels = dst_w * dst_h;
    let dst_vals = dst_pixels * channels;
    let in_padded = src_bytes.div_ceil(4) * 4;
    let mut padded_in = src.to_vec();
    padded_in.resize(in_padded, 0u8);

    let in_handle = client.create_from_slice(&padded_in);
    let out_handle = client.empty(dst_vals * 4);
    let mat_handle = client.create_from_slice(f32::as_bytes(matrix.as_slice()));

    let cube_dim = CubeDim::new_1d(256);
    let cube_count = calculate_cube_count_elemwise::<WgpuRuntime>(client, dst_pixels, cube_dim);

    macro_rules! launch_ch {
        ($ch:expr) => {
            unsafe {
                warp_affine_kernel::launch::<WgpuRuntime>(
                    client,
                    cube_count,
                    cube_dim,
                    ArrayArg::from_raw_parts::<u32>(&in_handle, in_padded / 4, 1),
                    ArrayArg::from_raw_parts::<f32>(&out_handle, dst_vals, 1),
                    ArrayArg::from_raw_parts::<f32>(&mat_handle, 6, 1),
                    src_w,
                    src_h,
                    dst_w,
                    dst_h,
                    $ch,
                )
            }
            .unwrap()
        };
    }
    match channels {
        1 => launch_ch!(1),
        2 => launch_ch!(2),
        3 => launch_ch!(3),
        _ => launch_ch!(4),
    }

    let result = client.read_one(out_handle);
    let floats = f32::from_bytes(&result);
    floats.iter().map(|&v| v.round().clamp(0.0, 255.0) as u8).collect()
}

// ---------------------------------------------------------------------------
// Template matching — SSD score map
// ---------------------------------------------------------------------------

#[cube(launch)]
fn template_match_ssd_kernel(
    image: &Array<u32>,
    tmpl: &Array<u32>,
    scores: &mut Array<f32>,
    #[comptime] img_w: usize,
    #[comptime] tmpl_w: usize,
    #[comptime] tmpl_h: usize,
    #[comptime] channels: usize,
    #[comptime] result_w: usize,
    #[comptime] n_positions: usize,
) {
    let pos = ABSOLUTE_POS;
    if pos < n_positions {
        let ox = pos % result_w;
        let oy = pos / result_w;

        let mut ssd = 0.0f32;
        for ty in 0usize..tmpl_h {
            for tx in 0usize..tmpl_w {
                for ch in 0usize..channels {
                    let img_b = read_u8_packed(image, ((oy + ty) * img_w + (ox + tx)) * channels + ch) as f32;
                    let tpl_b = read_u8_packed(tmpl, (ty * tmpl_w + tx) * channels + ch) as f32;
                    let diff = img_b - tpl_b;
                    ssd += diff * diff;
                }
            }
        }
        scores[pos] = ssd;
    }
}

/// Compute SSD template match scores for every valid position.
pub fn template_match_ssd(
    client: &WgpuClient,
    image: &[u8],
    img_w: usize,
    img_h: usize,
    tmpl: &[u8],
    tmpl_w: usize,
    tmpl_h: usize,
    channels: usize,
) -> Vec<f32> {
    let img_bytes = img_w * img_h * channels;
    let tmpl_bytes = tmpl_w * tmpl_h * channels;
    let result_w = img_w - tmpl_w + 1;
    let result_h = img_h - tmpl_h + 1;
    let n_positions = result_w * result_h;

    let in_padded = img_bytes.div_ceil(4) * 4;
    let tmpl_padded = tmpl_bytes.div_ceil(4) * 4;
    let mut padded_img = image.to_vec();
    padded_img.resize(in_padded, 0u8);
    let mut padded_tmpl = tmpl.to_vec();
    padded_tmpl.resize(tmpl_padded, 0u8);

    let img_handle = client.create_from_slice(&padded_img);
    let tmpl_handle = client.create_from_slice(&padded_tmpl);
    let scores_handle = client.empty(n_positions * 4);

    let cube_dim = CubeDim::new_1d(256);
    let cube_count = calculate_cube_count_elemwise::<WgpuRuntime>(client, n_positions, cube_dim);

    macro_rules! launch_ch {
        ($ch:expr) => {
            unsafe {
                template_match_ssd_kernel::launch::<WgpuRuntime>(
                    client,
                    cube_count,
                    cube_dim,
                    ArrayArg::from_raw_parts::<u32>(&img_handle, in_padded / 4, 1),
                    ArrayArg::from_raw_parts::<u32>(&tmpl_handle, tmpl_padded / 4, 1),
                    ArrayArg::from_raw_parts::<f32>(&scores_handle, n_positions, 1),
                    img_w,
                    tmpl_w,
                    tmpl_h,
                    $ch,
                    result_w,
                    n_positions,
                )
            }
            .unwrap()
        };
    }
    match channels {
        1 => launch_ch!(1),
        2 => launch_ch!(2),
        3 => launch_ch!(3),
        _ => launch_ch!(4),
    }

    let result = client.read_one(scores_handle);
    f32::from_bytes(&result).to_vec()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cubecl::get_client;
    use serial_test::serial;

    #[test]
    #[serial]
    fn test_threshold_binary() {
        let client = get_client();
        let input = vec![0u8, 100, 200, 255];
        let result = threshold(&client, &input, 128, 255, 0);
        assert_eq!(result[0], 0, "0 <= 128 → 0");
        assert_eq!(result[1], 0, "100 <= 128 → 0");
        assert_eq!(result[2], 255, "200 > 128 → 255");
        assert_eq!(result[3], 255, "255 > 128 → 255");
    }

    #[test]
    #[serial]
    fn test_rgb_to_grayscale() {
        let client = get_client();
        // Pure red pixel (R=255, G=0, B=0) → luminance = 255*0.299 ≈ 76
        let rgb = vec![255u8, 0, 0];
        let gray = rgb_to_grayscale(&client, &rgb, 1);
        assert_eq!(gray[0], 76);
    }

    #[test]
    #[serial]
    fn test_grayscale_to_rgb_round_trip() {
        let client = get_client();
        let gray = vec![42u8, 128, 200];
        let rgb = grayscale_to_rgb(&client, &gray, 3);
        assert_eq!(rgb.len(), 9);
        for i in 0..3 {
            assert_eq!(rgb[i * 3], gray[i]);
            assert_eq!(rgb[i * 3 + 1], gray[i]);
            assert_eq!(rgb[i * 3 + 2], gray[i]);
        }
    }
}
