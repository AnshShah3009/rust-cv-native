use crate::context::ThresholdType;
use crate::gpu::GpuContext;
use crate::gpu_kernels::dispatch::GpuDispatch;
use crate::GpuTensor;
use crate::Result;
use std::marker::PhantomData;

#[derive(Copy, Clone)]
#[repr(C)]
struct ThresholdParams<T: bytemuck::Pod + bytemuck::Zeroable> {
    width: u32,
    height: u32,
    thresh: T,
    max_value: T,
    thresh_type: u32,
    len: u32,
}

unsafe impl<T: bytemuck::Pod + bytemuck::Zeroable> bytemuck::Pod for ThresholdParams<T> {}
unsafe impl<T: bytemuck::Pod + bytemuck::Zeroable> bytemuck::Zeroable for ThresholdParams<T> {}

pub fn threshold<T: cv_core::float::Float + bytemuck::Pod + bytemuck::Zeroable>(
    ctx: &GpuContext,
    input: &GpuTensor<T>,
    thresh: T,
    max_value: T,
    thresh_type: ThresholdType,
) -> Result<GpuTensor<T>> {
    let len = input.shape.len();
    let byte_size = (len * std::mem::size_of::<T>()) as u64;

    let params = ThresholdParams {
        width: input.shape.width as u32,
        height: input.shape.height as u32,
        thresh,
        max_value,
        thresh_type: thresh_type as u32,
        len: len as u32,
    };

    let precision = crate::gpu_kernels::shader_template::precision_for_type::<T>()?;
    let shader_source = crate::gpu_kernels::shader_template::resolve(
        include_str!("../../shaders/threshold_f32.wgsl"),
        precision,
    );

    let outputs = GpuDispatch::new(ctx, &shader_source, "Threshold")
        .input(input.storage.buffer())
        .output(byte_size)
        .params(&params)
        .dispatch_1d(len as u32)?;

    Ok(crate::GpuTensor {
        storage: crate::storage::WgpuGpuStorage::from_buffer(
            outputs.into_iter().next().unwrap(),
            len,
        ),
        shape: input.shape,
        dtype: input.dtype,
        _phantom: PhantomData,
    })
}
