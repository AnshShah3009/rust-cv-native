use crate::gpu::GpuContext;
use crate::gpu_kernels::dispatch::GpuDispatch;
use crate::storage::GpuStorage;
use crate::Result;
use cv_core::Tensor;
use std::marker::PhantomData;

pub fn subtract(
    ctx: &GpuContext,
    a: &Tensor<f32, GpuStorage<f32>>,
    b: &Tensor<f32, GpuStorage<f32>>,
) -> Result<Tensor<f32, GpuStorage<f32>>> {
    assert_eq!(a.shape, b.shape);
    let size = a.shape.len();
    let byte_size = (size * std::mem::size_of::<f32>()) as u64;

    let outputs = GpuDispatch::new(ctx, include_str!("../../shaders/subtract.wgsl"), "Subtract")
        .input(a.storage.buffer())
        .input(b.storage.buffer())
        .output(byte_size)
        .dispatch_1d(size as u32)?;

    Ok(Tensor {
        storage: GpuStorage::from_buffer(outputs.into_iter().next().unwrap(), size),
        shape: a.shape,
        dtype: a.dtype,
        _phantom: PhantomData,
    })
}
