use crate::gpu::GpuContext;
use crate::cpu::CpuBackend;
use crate::context::{ComputeContext, BorderMode, ThresholdType};
use crate::Result;
use cv_core::{Tensor, storage::Storage};
use std::sync::OnceLock;

static CPU_CONTEXT: OnceLock<CpuBackend> = OnceLock::new();

#[derive(Clone, Copy)]
pub enum ComputeDevice<'a> {
    Cpu(&'a CpuBackend),
    Gpu(&'a GpuContext),
}

impl<'a> ComputeDevice<'a> {
    pub fn convolve_2d<S: Storage<f32> + 'static>(
        &self,
        input: &Tensor<f32, S>,
        kernel: &Tensor<f32, S>,
        border_mode: BorderMode,
    ) -> Result<Tensor<f32, S>> {
        match self {
            ComputeDevice::Cpu(cpu) => cpu.convolve_2d(input, kernel, border_mode),
            ComputeDevice::Gpu(gpu) => gpu.convolve_2d(input, kernel, border_mode),
        }
    }

    pub fn threshold<S: Storage<u8> + 'static>(
        &self,
        input: &Tensor<u8, S>,
        thresh: u8,
        max_value: u8,
        typ: ThresholdType,
    ) -> Result<Tensor<u8, S>> {
        match self {
            ComputeDevice::Cpu(cpu) => cpu.threshold(input, thresh, max_value, typ),
            ComputeDevice::Gpu(gpu) => gpu.threshold(input, thresh, max_value, typ),
        }
    }
}

/// Get the best available compute device.
pub fn get_device() -> ComputeDevice<'static> {
    if let Some(gpu) = GpuContext::global() {
        ComputeDevice::Gpu(gpu)
    } else {
        ComputeDevice::Cpu(CPU_CONTEXT.get_or_init(|| CpuBackend::new().unwrap()))
    }
}
