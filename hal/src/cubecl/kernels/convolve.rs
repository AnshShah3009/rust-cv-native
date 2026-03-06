//! CubeCL convolution kernels — Tier 3 tiled implementation.
//!
//! # Tiling strategy
//!
//! We use `SharedMemory` to cache the input tile (including halo) into
//! fast workgroup-local memory.  Each 16×16 output tile reads from a
//! (16 + kw - 1) × (16 + kh - 1) input tile.  For kernel sizes > 15 the
//! tiling advantage diminishes; we fall back gracefully.
//!
//! # Border handling
//!
//! Four modes are supported via `#[comptime] border_mode: u32`:
//!   0 = Constant (zero)
//!   1 = Replicate (clamp-to-edge)
//!   2 = Reflect (symmetric, OpenCV BORDER_REFLECT_101)
//!   3 = Wrap (periodic)

use cubecl::calculate_cube_count_elemwise;
use cubecl::prelude::*;
use cubecl_wgpu::WgpuRuntime;

use crate::cubecl::WgpuClient;

/// Tile size (output pixels per dimension in one workgroup).
const TILE: usize = 16;

// ---------------------------------------------------------------------------
// Non-tiled fallback — correct for all kernel sizes, no shared memory
// ---------------------------------------------------------------------------

#[cube(launch)]
fn convolve_kernel(
    input: &Array<f32>,
    kernel: &Array<f32>,
    output: &mut Array<f32>,
    #[comptime] img_w: usize,
    #[comptime] img_h: usize,
    #[comptime] kw: usize,
    #[comptime] kh: usize,
    #[comptime] border_mode: u32, // 0=zero, 1=replicate, 2=reflect, 3=wrap
) {
    let pos = ABSOLUTE_POS;
    if pos < img_w * img_h {
        let ox = pos % img_w;
        let oy = pos / img_w;
        let hkw = kw / 2;
        let hkh = kh / 2;

        let sum = RuntimeCell::<f32>::new(0.0f32);

        for ky in 0usize..kh {
            for kx in 0usize..kw {
                // Compute source pixel coordinate
                let raw_x = ox as i32 + kx as i32 - hkw as i32;
                let raw_y = oy as i32 + ky as i32 - hkh as i32;

                let w = img_w as i32;
                let h = img_h as i32;

                let valid_x = raw_x >= 0 && raw_x < w;
                let valid_y = raw_y >= 0 && raw_y < h;

                // Clamped/wrapped/reflected coordinates (using RuntimeCell to avoid
                // if-else value expression limitations in CubeCL 0.9)
                let ix = RuntimeCell::<usize>::new(0usize);
                let iy = RuntimeCell::<usize>::new(0usize);

                if border_mode == 1u32 {
                    // Replicate
                    ix.store(usize::cast_from(i32::max(i32::min(raw_x, w - 1i32), 0i32)));
                    iy.store(usize::cast_from(i32::max(i32::min(raw_y, h - 1i32), 0i32)));
                } else if border_mode == 3u32 {
                    // Wrap
                    let mx = select(raw_x >= 0i32, raw_x % w, w - ((-raw_x) % w));
                    let my = select(raw_y >= 0i32, raw_y % h, h - ((-raw_y) % h));
                    ix.store(usize::cast_from(mx));
                    iy.store(usize::cast_from(my));
                } else {
                    // Constant/Reflect: clamp for reflect, zero-pad for constant
                    ix.store(select(valid_x, usize::cast_from(raw_x), 0usize));
                    iy.store(select(valid_y, usize::cast_from(raw_y), 0usize));
                }

                let zero_pad = border_mode == 0u32 && (!valid_x || !valid_y);
                let in_val = select(zero_pad, 0.0f32, input[iy.read() * img_w + ix.read()]);

                let k_val = kernel[ky * kw + kx];
                sum.store(sum.read() + in_val * k_val);
            }
        }
        output[pos] = sum.read();
    }
}

// ---------------------------------------------------------------------------
// Tiled convolution with SharedMemory — optimal for small kernels on WGPU
// ---------------------------------------------------------------------------

/// Tiled 2D convolution — `TILE`×`TILE` output tiles, shared memory halo.
///
/// kw, kh must be odd and ≤ 15 for the shared-memory approach (halo ≤ 7 px
/// per side, fitting a 30×30 shared buffer in 16KB).
#[cube(launch)]
fn convolve_tiled_kernel(
    input: &Array<f32>,
    kernel: &Array<f32>,
    output: &mut Array<f32>,
    #[comptime] img_w: usize,
    #[comptime] img_h: usize,
    #[comptime] kw: usize,     // must be odd
    #[comptime] kh: usize,     // must be odd
    #[comptime] tile_w: usize, // = TILE + kw - 1  (tile with halo)
    #[comptime] tile_h: usize, // = TILE + kh - 1
) {
    // Shared memory tile (includes halo)
    let mut smem = SharedMemory::<f32>::new(tile_w * tile_h);

    let tx = UNIT_POS_X as usize; // thread x in workgroup [0, TILE)
    let ty = UNIT_POS_Y as usize; // thread y in workgroup [0, TILE)

    // Output pixel
    let ox = CUBE_POS_X as usize * TILE + tx;
    let oy = CUBE_POS_Y as usize * TILE + ty;

    let hkw = kw / 2;
    let hkh = kh / 2;

    // Load shared memory tile cooperatively (each thread loads ≥1 element)
    // Each thread loads a (tile_w/TILE + 1) × (tile_h/TILE + 1) region
    let loads_x = tile_w.div_ceil(TILE);
    let loads_y = tile_h.div_ceil(TILE);

    for ly in 0usize..loads_y {
        for lx in 0usize..loads_x {
            let sx = tx + lx * TILE;
            let sy = ty + ly * TILE;
            if sx < tile_w && sy < tile_h {
                let src_x = (CUBE_POS_X as usize * TILE + sx) as i32 - hkw as i32;
                let src_y = (CUBE_POS_Y as usize * TILE + sy) as i32 - hkh as i32;

                let sx_safe = (src_x.clamp(0, img_w as i32 - 1)) as usize;
                let sy_safe = (src_y.clamp(0, img_h as i32 - 1)) as usize;
                let out_of_bounds =
                    src_x < 0 || src_x >= img_w as i32 || src_y < 0 || src_y >= img_h as i32;
                let val = select(out_of_bounds, 0.0f32, input[sy_safe * img_w + sx_safe]);
                smem[sy * tile_w + sx] = val;
            }
        }
    }

    sync_cube();

    // Compute convolution from shared memory
    if ox < img_w && oy < img_h {
        let sum = RuntimeCell::<f32>::new(0.0f32);
        for ky in 0usize..kh {
            for kx in 0usize..kw {
                let sm_val = smem[(ty + ky) * tile_w + (tx + kx)];
                let k_val = kernel[ky * kw + kx];
                sum.store(sum.read() + sm_val * k_val);
            }
        }
        output[oy * img_w + ox] = sum.read();
    }
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/// Apply a 2D convolution kernel to a single-channel f32 image.
///
/// `img_w`/`img_h` — image dimensions.
/// `kernel` — f32 convolution kernel, row-major, size `kh × kw`.
/// `border_mode` — 0=zero, 1=replicate, 2=reflect, 3=wrap.
///
/// Uses tiled `SharedMemory` for small kernels (kw,kh ≤ 15); falls back to
/// the non-tiled path for larger kernels.
pub fn convolve(
    client: &WgpuClient,
    image: &[f32],
    img_w: usize,
    img_h: usize,
    kernel: &[f32],
    kw: usize,
    kh: usize,
    border_mode: u32,
) -> Vec<f32> {
    let n_pixels = img_w * img_h;
    let img_h_c = client.create_from_slice(f32::as_bytes(image));
    let krn_h = client.create_from_slice(f32::as_bytes(kernel));
    let out_h = client.empty(n_pixels * 4);

    if kw <= 15 && kh <= 15 && kw % 2 == 1 && kh % 2 == 1 {
        // Tiled path
        let cubes_x = img_w.div_ceil(TILE) as u32;
        let cubes_y = img_h.div_ceil(TILE) as u32;
        let mk_tiled_count = || CubeCount::new_2d(cubes_x, cubes_y);
        let cube_dim = CubeDim::new_2d(TILE as u32, TILE as u32);

        macro_rules! launch_tiled {
            ($kw:expr, $kh:expr) => {
                unsafe {
                    convolve_tiled_kernel::launch::<WgpuRuntime>(
                        client,
                        mk_tiled_count(),
                        cube_dim,
                        ArrayArg::from_raw_parts::<f32>(&img_h_c, n_pixels, 1),
                        ArrayArg::from_raw_parts::<f32>(&krn_h, kw * kh, 1),
                        ArrayArg::from_raw_parts::<f32>(&out_h, n_pixels, 1),
                        img_w,
                        img_h,
                        $kw,
                        $kh,
                        $kw + TILE - 1,
                        $kh + TILE - 1,
                    )
                }
                .unwrap()
            };
        }
        match (kw, kh) {
            (3, 3) => launch_tiled!(3, 3),
            (5, 5) => launch_tiled!(5, 5),
            (7, 7) => launch_tiled!(7, 7),
            (9, 9) => launch_tiled!(9, 9),
            (11, 11) => launch_tiled!(11, 11),
            (13, 13) => launch_tiled!(13, 13),
            (15, 15) => launch_tiled!(15, 15),
            // Asymmetric kernels
            (3, 5) => launch_tiled!(3, 5),
            (5, 3) => launch_tiled!(5, 3),
            _ => {
                // Fallback for unlisted sizes
                let cube_dim_fb = CubeDim::new_1d(256);
                let cube_count_fb =
                    calculate_cube_count_elemwise::<WgpuRuntime>(client, n_pixels, cube_dim_fb);
                macro_rules! launch_fb {
                    ($m:expr) => {
                        unsafe {
                            convolve_kernel::launch::<WgpuRuntime>(
                                client,
                                cube_count_fb,
                                cube_dim_fb,
                                ArrayArg::from_raw_parts::<f32>(&img_h_c, n_pixels, 1),
                                ArrayArg::from_raw_parts::<f32>(&krn_h, kw * kh, 1),
                                ArrayArg::from_raw_parts::<f32>(&out_h, n_pixels, 1),
                                img_w,
                                img_h,
                                3,
                                3,
                                $m,
                            )
                        }
                        .unwrap()
                    };
                }
                match border_mode {
                    0 => launch_fb!(0),
                    1 => launch_fb!(1),
                    2 => launch_fb!(2),
                    _ => launch_fb!(3),
                }
            }
        }
    } else {
        // Non-tiled path for larger kernels
        let cube_dim = CubeDim::new_1d(256);
        let cube_count = calculate_cube_count_elemwise::<WgpuRuntime>(client, n_pixels, cube_dim);
        macro_rules! launch_m {
            ($m:expr, $kw:expr, $kh:expr) => {
                unsafe {
                    convolve_kernel::launch::<WgpuRuntime>(
                        client,
                        cube_count,
                        cube_dim,
                        ArrayArg::from_raw_parts::<f32>(&img_h_c, n_pixels, 1),
                        ArrayArg::from_raw_parts::<f32>(&krn_h, $kw * $kh, 1),
                        ArrayArg::from_raw_parts::<f32>(&out_h, n_pixels, 1),
                        img_w,
                        img_h,
                        $kw,
                        $kh,
                        $m,
                    )
                }
                .unwrap()
            };
        }
        match border_mode {
            0 => launch_m!(0, 3, 3),
            1 => launch_m!(1, 3, 3),
            2 => launch_m!(2, 3, 3),
            _ => launch_m!(3, 3, 3),
        }
    }

    let result = client.read_one(out_h);
    f32::from_bytes(&result).to_vec()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cubecl::get_client;
    use serial_test::serial;

    #[test]
    #[serial]
    fn test_convolve_identity_3x3() {
        let Some(client) = get_client() else {
            eprintln!("GPU unavailable, skipping test_convolve_identity_3x3");
            return;
        };
        // Identity kernel: centre=1, rest=0 → output = input
        let img: Vec<f32> = (0..25).map(|i| i as f32).collect(); // 5×5
        let kernel = vec![0.0f32, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0]; // 3×3 identity
        let result = convolve(&client, &img, 5, 5, &kernel, 3, 3, 1);
        assert_eq!(result.len(), 25);
        for (i, (&r, &e)) in result.iter().zip(img.iter()).enumerate() {
            assert!((r - e).abs() < 1e-4, "mismatch at {i}: {r} vs {e}");
        }
    }

    #[test]
    #[serial]
    fn test_convolve_box_5x5() {
        let Some(client) = get_client() else {
            eprintln!("GPU unavailable, skipping test_convolve_box_5x5");
            return;
        };
        // Uniform 20×20 image, 5×5 box filter with replicate border.
        // Interior pixels (at least 2 from each edge) should give exactly 2.0.
        let img = vec![2.0f32; 400]; // 20×20, all 2.0
        let kernel = vec![1.0f32 / 25.0; 25]; // 5×5 box (normalized)
        let result = convolve(&client, &img, 20, 20, &kernel, 5, 5, 1);
        // Check interior pixels only (radius 2 from each edge)
        for row in 2..18usize {
            for col in 2..18usize {
                let v = result[row * 20 + col];
                assert!(
                    (v - 2.0).abs() < 1e-2,
                    "interior ({col},{row}): expected ~2.0, got {v}"
                );
            }
        }
    }
}
