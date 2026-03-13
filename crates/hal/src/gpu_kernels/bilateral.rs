use crate::gpu::GpuContext;
use crate::gpu_kernels::dispatch::GpuDispatch;
use crate::Result;
use cv_core::storage::Storage;
use std::marker::PhantomData;

#[derive(Copy, Clone)]
#[repr(C)]
struct BilateralParams {
    width: u32,
    height: u32,
    radius: i32,
    sigma_color_sq_inv: f32,
    sigma_space_sq_inv: f32,
}

unsafe impl bytemuck::Pod for BilateralParams {}
unsafe impl bytemuck::Zeroable for BilateralParams {}

pub fn bilateral_filter<T: cv_core::float::Float + bytemuck::Pod + bytemuck::Zeroable>(
    ctx: &GpuContext,
    input: &crate::GpuTensor<T>,
    d: i32,
    sigma_color: T,
    sigma_space: T,
) -> Result<crate::GpuTensor<T>> {
    use crate::storage::GpuStorage;
    let sigma_color_f = sigma_color.to_f32();
    let sigma_space_f = sigma_space.to_f32();
    let (h, w) = input.shape.hw();
    let c = input.shape.channels;

    if c != 1 {
        return Err(crate::Error::NotSupported(
            "GPU Bilateral currently only for grayscale".into(),
        ));
    }

    let out_len = h * w * c;
    let byte_size = (out_len * std::mem::size_of::<T>()) as u64;

    let radius = if d <= 0 {
        (sigma_space_f * 1.5).ceil() as i32
    } else {
        d / 2
    };
    let params = BilateralParams {
        width: w as u32,
        height: h as u32,
        radius,
        sigma_color_sq_inv: -0.5 / (sigma_color_f * sigma_color_f),
        sigma_space_sq_inv: -0.5 / (sigma_space_f * sigma_space_f),
    };

    let precision = crate::gpu_kernels::shader_template::precision_for_type::<T>()?;
    let shader_source = crate::gpu_kernels::shader_template::resolve(
        include_str!("../../shaders/bilateral_f32.wgsl"),
        precision,
    );

    let input_buf = input
        .storage
        .as_any()
        .downcast_ref::<GpuStorage<f32>>()
        .unwrap()
        .buffer();

    let outputs = GpuDispatch::new(ctx, &shader_source, "Bilateral")
        .input(input_buf)
        .output(byte_size)
        .params(&params)
        .dispatch_2d(w as u32, h as u32)?;

    Ok(crate::GpuTensor {
        storage: crate::storage::WgpuGpuStorage::from_buffer(
            outputs.into_iter().next().unwrap(),
            out_len,
        ),
        shape: input.shape,
        dtype: input.dtype,
        _phantom: PhantomData,
    })
}
