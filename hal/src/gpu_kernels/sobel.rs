use crate::gpu::GpuContext;
use crate::gpu_kernels::dispatch::GpuDispatch;
use crate::storage::GpuStorage;
use crate::Result;
use cv_core::Tensor;
use std::marker::PhantomData;

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct SobelParams {
    width: u32,
    height: u32,
    ksize: u32,
    border_mode: u32,
}

#[allow(clippy::type_complexity)]
pub fn sobel(
    ctx: &GpuContext,
    input: &Tensor<f32, GpuStorage<f32>>,
    dx: i32,
    dy: i32,
    ksize: usize,
) -> Result<(Tensor<f32, GpuStorage<f32>>, Tensor<f32, GpuStorage<f32>>)> {
    if dx != 1 || dy != 1 || ksize != 3 {
        return Err(crate::Error::NotSupported(
            "GPU Sobel only supports dx=1, dy=1, ksize=3 currently".into(),
        ));
    }

    let len = input.shape.len();
    let (h, w) = input.shape.hw();
    let byte_size = (len * 4) as u64;

    let params = SobelParams {
        width: w as u32,
        height: h as u32,
        ksize: ksize as u32,
        border_mode: 1, // Replicate
    };

    let wg_x = (w as u32).div_ceil(4).div_ceil(16);
    let wg_y = (h as u32).div_ceil(16);

    let outputs = GpuDispatch::new(ctx, include_str!("../../shaders/sobel.wgsl"), "Sobel")
        .input(input.storage.buffer())
        .output(byte_size) // gx
        .output(byte_size) // gy
        .params(&params)
        .dispatch_raw(wg_x, wg_y, 1)?;

    let mut iter = outputs.into_iter();
    let gx_buf = iter.next().unwrap();
    let gy_buf = iter.next().unwrap();

    Ok((
        Tensor {
            storage: GpuStorage::from_buffer(gx_buf, len),
            shape: input.shape,
            dtype: input.dtype,
            _phantom: PhantomData,
        },
        Tensor {
            storage: GpuStorage::from_buffer(gy_buf, len),
            shape: input.shape,
            dtype: input.dtype,
            _phantom: PhantomData,
        },
    ))
}
