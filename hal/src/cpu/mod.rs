use crate::{BackendType, Capability, ComputeBackend, DeviceId, QueueId, QueueType, Result};
use crate::context::{ComputeContext, BorderMode, ThresholdType};
use cv_core::{Tensor, storage::Storage, TensorShape};
use rayon::prelude::*;
use wide::*;

pub mod simd;

pub struct CpuBackend {
    device_id: DeviceId,
    num_threads: usize,
    simd_available: bool,
}

impl CpuBackend {
    pub fn new() -> Option<Self> {
        Some(Self {
            device_id: DeviceId(0),
            num_threads: rayon::current_num_threads(),
            simd_available: true,
        })
    }

    pub fn num_threads(&self) -> usize {
        self.num_threads
    }
}

impl ComputeBackend for CpuBackend {
    fn backend_type(&self) -> BackendType {
        BackendType::Cpu
    }

    fn name(&self) -> &str {
        "CPU"
    }

    fn device_id(&self) -> DeviceId {
        self.device_id
    }

    fn supports(&self, capability: Capability) -> bool {
        match capability {
            Capability::Compute => true,
            Capability::Simd => self.simd_available,
            Capability::TensorCore => false,
            Capability::RayTracing => false,
        }
    }

    fn queue(&self, _queue_type: QueueType) -> QueueId {
        QueueId(0)
    }

    fn preferred_queue(&self) -> QueueType {
        QueueType::Compute
    }
}

impl ComputeContext for CpuBackend {
    fn backend_type(&self) -> BackendType {
        BackendType::Cpu
    }

    fn device_id(&self) -> DeviceId {
        self.device_id
    }

    fn wait_idle(&self) -> Result<()> {
        Ok(())
    }

    fn convolve_2d<S: Storage<f32> + 'static>(
        &self,
        input: &Tensor<f32, S>,
        kernel: &Tensor<f32, S>,
        border_mode: BorderMode,
    ) -> Result<Tensor<f32, S>> {
        let src = input.storage.as_slice().ok_or_else(|| crate::Error::MemoryError("Input not on CPU".into()))?;
        let k_data = kernel.storage.as_slice().ok_or_else(|| crate::Error::MemoryError("Kernel not on CPU".into()))?;

        let (h, w) = input.shape.hw();
        let (kh, kw) = kernel.shape.hw();
        let (cx, cy) = (kw / 2, kh / 2);
        
        let mut output_storage = S::new(input.shape.len(), 0.0);
        
        {
            let out_slice = output_storage.as_mut_slice().ok_or_else(|| crate::Error::MemoryError("Output not on CPU".into()))?;
            
            out_slice.par_chunks_mut(w).enumerate().for_each(|(y, row_out)| {
                for x in 0..w {
                    let mut sum = 0.0;
                    for j in 0..kh {
                        for i in 0..kw {
                            let src_y = y as isize + j as isize - cy as isize;
                            let src_x = x as isize + i as isize - cx as isize;
                            
                            let val = if src_y >= 0 && src_y < h as isize && src_x >= 0 && src_x < w as isize {
                                src[src_y as usize * w + src_x as usize]
                            } else {
                                match border_mode {
                                    BorderMode::Constant(v) => v,
                                    BorderMode::Replicate => {
                                        let c = src_x.clamp(0, w as isize - 1) as usize;
                                        let r = src_y.clamp(0, h as isize - 1) as usize;
                                        src[r * w + c]
                                    }
                                    _ => 0.0,
                                }
                            };
                            
                            sum += val * k_data[j * kw + i];
                        }
                    }
                    row_out[x] = sum;
                }
            });
        }

        Ok(Tensor {
            storage: output_storage,
            shape: input.shape,
            dtype: input.dtype,
            _phantom: std::marker::PhantomData,
        })
    }

    fn dispatch<S: Storage<u8> + 'static>(
        &self,
        _name: &str,
        _buffers: &[&Tensor<u8, S>],
        _uniforms: &[u8],
        _workgroups: (u32, u32, u32),
    ) -> Result<()> {
        Err(crate::Error::NotSupported("Generic dispatch not supported on CPU".into()))
    }

    fn threshold<S: Storage<u8> + 'static>(
        &self,
        input: &Tensor<u8, S>,
        thresh: u8,
        max_value: u8,
        typ: ThresholdType,
    ) -> Result<Tensor<u8, S>> {
        let src = input.storage.as_slice().ok_or_else(|| crate::Error::MemoryError("Input not on CPU".into()))?;
        let len = src.len();
        
        let mut output_storage = S::new(len, 0);
        let dst = output_storage.as_mut_slice().ok_or_else(|| crate::Error::MemoryError("Output not on CPU".into()))?;

        let thresh_v = f32x8::splat(thresh as f32);
        let max_v = f32x8::splat(max_value as f32);
        let zero_v = f32x8::ZERO;

        dst.par_chunks_mut(4096).enumerate().for_each(|(chunk_idx, dst_chunk)| {
            let offset = chunk_idx * 4096;
            let src_chunk = &src[offset..offset + dst_chunk.len()];
            
            let n = dst_chunk.len();
            for i in (0..n).step_by(8) {
                if i + 8 <= n {
                    let s_v = f32x8::from([
                        src_chunk[i] as f32,
                        src_chunk[i + 1] as f32,
                        src_chunk[i + 2] as f32,
                        src_chunk[i + 3] as f32,
                        src_chunk[i + 4] as f32,
                        src_chunk[i + 5] as f32,
                        src_chunk[i + 6] as f32,
                        src_chunk[i + 7] as f32,
                    ]);
                    let res = match typ {
                        ThresholdType::Binary => s_v.cmp_gt(thresh_v).blend(max_v, zero_v),
                        ThresholdType::BinaryInv => s_v.cmp_gt(thresh_v).blend(zero_v, max_v),
                        ThresholdType::Trunc => s_v.min(thresh_v),
                        ThresholdType::ToZero => s_v.cmp_gt(thresh_v).blend(s_v, zero_v),
                        ThresholdType::ToZeroInv => s_v.cmp_gt(thresh_v).blend(zero_v, s_v),
                    };
                    let res_arr: [f32; 8] = res.into();
                    for j in 0..8 {
                        dst_chunk[i + j] = res_arr[j] as u8;
                    }
                } else {
                    for j in i..n {
                        let value = src_chunk[j];
                        dst_chunk[j] = match typ {
                            ThresholdType::Binary => if value > thresh { max_value } else { 0 },
                            ThresholdType::BinaryInv => if value > thresh { 0 } else { max_value },
                            ThresholdType::Trunc => value.min(thresh),
                            ThresholdType::ToZero => if value > thresh { value } else { 0 },
                            ThresholdType::ToZeroInv => if value > thresh { 0 } else { value },
                        };
                    }
                }
            }
        });

        Ok(Tensor {
            storage: output_storage,
            shape: input.shape,
            dtype: input.dtype,
            _phantom: std::marker::PhantomData,
        })
    }
}

impl CpuBackend {
    pub fn is_available() -> bool {
        true
    }
}
