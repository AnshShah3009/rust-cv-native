use crate::gpu::GpuContext;
use crate::gpu_kernels::resize::resize;
use crate::storage::GpuStorage;
use crate::Result;
use cv_core::Tensor;

#[derive(Debug, Clone)]
pub struct ImagePyramid<S: cv_core::storage::Storage<u8>> {
    pub levels: Vec<Tensor<u8, S>>,
    pub scales: Vec<f32>,
}

pub fn build_pyramid(
    ctx: &GpuContext,
    input: &Tensor<u8, GpuStorage<u8>>,
    n_levels: usize,
    scale_factor: f32,
) -> Result<ImagePyramid<GpuStorage<u8>>> {
    let mut levels = Vec::with_capacity(n_levels);
    let mut scales = Vec::with_capacity(n_levels);

    levels.push(input.clone());
    scales.push(1.0);

    let mut current = input.clone();
    let mut current_scale = 1.0;

    for _ in 1..n_levels {
        current_scale *= scale_factor;
        let inv_scale = 1.0 / current_scale;

        let (h_orig, w_orig) = input.shape.hw();
        let new_w = (w_orig as f32 * inv_scale).round() as usize;
        let new_h = (h_orig as f32 * inv_scale).round() as usize;

        // Ensure minimum size
        if new_w < 16 || new_h < 16 {
            break;
        }

        let next = resize(ctx, &current, (new_w, new_h))?;
        levels.push(next.clone());
        scales.push(current_scale);
        current = next;
    }

    Ok(ImagePyramid { levels, scales })
}
