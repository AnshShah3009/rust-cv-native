use crate::gpu::GpuContext;
use crate::gpu_kernels::dispatch::GpuDispatch;
use crate::GpuTensor;
use crate::Result;
use cv_core::storage::Storage;
use cv_core::TensorShape;
use std::marker::PhantomData;

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct ResizeParams {
    src_w: u32,
    src_h: u32,
    dst_w: u32,
    dst_h: u32,
    channels: u32,
}

pub fn resize<T: cv_core::float::Float + bytemuck::Pod>(
    ctx: &GpuContext,
    input: &GpuTensor<T>,
    new_width: u32,
    new_height: u32,
) -> Result<GpuTensor<T>> {
    use crate::storage::GpuStorage;
    let (src_h, src_w) = input.shape.hw();
    let (dst_w, dst_h) = (new_width as usize, new_height as usize);
    let c = input.shape.channels;

    if c != 1 {
        return Err(crate::Error::NotSupported(
            "GPU Resize currently only for grayscale".into(),
        ));
    }

    let out_len = dst_w * dst_h * c;
    let byte_size = (out_len * std::mem::size_of::<T>()) as u64;

    let params = ResizeParams {
        src_w: src_w as u32,
        src_h: src_h as u32,
        dst_w: dst_w as u32,
        dst_h: dst_h as u32,
        channels: c as u32,
    };

    let precision = crate::gpu_kernels::shader_template::precision_for_type::<T>()?;
    let shader_source = crate::gpu_kernels::shader_template::resolve(
        include_str!("../../shaders/resize_f32.wgsl"),
        precision,
    );

    let outputs = GpuDispatch::new(ctx, &shader_source, "Resize")
        .input(
            input
                .storage
                .as_any()
                .downcast_ref::<GpuStorage<f32>>()
                .unwrap()
                .buffer(),
        )
        .output(byte_size)
        .params(&params)
        .dispatch_2d(dst_w as u32, dst_h as u32)?;

    Ok(crate::GpuTensor {
        storage: crate::storage::WgpuGpuStorage::from_buffer(
            outputs.into_iter().next().unwrap(),
            out_len,
        ),
        shape: TensorShape::new(c, dst_h, dst_w),
        dtype: input.dtype,
        _phantom: PhantomData,
    })
}
