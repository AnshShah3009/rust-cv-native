use crate::{
    backend::{
        BackendType, BindGroupLayout, BindResource, BindResourceType, Buffer, BufferDescriptor,
        BufferUsage, Capability, CommandEncoder, ComputeBackend, ComputePipeline, Device, QueueId,
        QueueType, Texture, TextureFormat,
    },
    queue::CommandQueue,
    Result,
};
use rayon::prelude::*;
use std::sync::Arc;

#[cfg(feature = "simd")]
use pulp::{Arch, Simd};

pub struct CpuBackend {
    device_id: crate::backend::DeviceId,
    num_threads: usize,
    simd_available: bool,
}

impl CpuBackend {
    pub fn new() -> Option<Self> {
        Some(Self {
            device_id: crate::backend::DeviceId(0),
            num_threads: rayon::current_num_threads(),
            #[cfg(feature = "simd")]
            simd_available: true,
            #[cfg(not(feature = "simd"))]
            simd_available: false,
        })
    }

    pub fn num_threads(&self) -> usize {
        self.num_threads
    }

    pub fn parallelize<F, T, R>(data: &[T], f: F) -> Vec<R>
    where
        F: Fn(&[T]) -> R + Send + Sync,
        T: Send + Sync,
        R: Send + Sync,
    {
        data.par_iter().map(f).collect()
    }

    pub fn parallelize_mut<F, T, R>(data: &mut [T], f: F) -> Vec<R>
    where
        F: Fn(&mut [T]) -> R + Send + Sync,
        T: Send + Sync,
        R: Send + Sync,
    {
        data.par_iter_mut().map(f).collect()
    }

    #[cfg(feature = "simd")]
    pub fn simd_process<F, T, R>(data: &[T], f: F) -> Vec<R>
    where
        F: Fn(&[T]) -> R,
        T: pulp::SimdScalar,
    {
        let arch = Arch::new();
        arch.run(&|| {
            let simd = Simd::current();
            f(data)
        })
    }
}

impl ComputeBackend for CpuBackend {
    fn backend_type(&self) -> BackendType {
        BackendType::Cpu
    }

    fn name(&self) -> &str {
        "CPU"
    }

    fn device_id(&self) -> crate::backend::DeviceId {
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

    fn is_available() -> bool {
        true
    }

    fn try_create() -> Option<Self> {
        Self::new()
    }
}

pub struct CpuBuffer {
    data: Vec<u8>,
    size: usize,
    usage: BufferUsage,
}

impl CpuBuffer {
    pub fn new(size: usize) -> Self {
        Self {
            data: vec![0u8; size],
            size,
            usage: BufferUsage::READ | BufferUsage::WRITE,
        }
    }

    pub fn from_vec(data: Vec<u8>) -> Self {
        let size = data.len();
        Self {
            data,
            size,
            usage: BufferUsage::READ | BufferUsage::WRITE,
        }
    }

    pub fn from_slice(slice: &[u8]) -> Self {
        Self::from_vec(slice.to_vec())
    }
}

impl Buffer for CpuBuffer {
    fn size(&self) -> usize {
        self.size
    }

    fn as_slice(&self) -> &[u8] {
        &self.data
    }

    fn as_mut_slice(&mut self) -> &mut [u8] {
        &mut self.data
    }

    fn device(&self) -> BackendType {
        BackendType::Cpu
    }

    fn copy_to_host(&self, host: &mut [u8]) -> Result<(), crate::Error> {
        if host.len() != self.size {
            return Err(crate::Error::MemoryError("Size mismatch".into()));
        }
        host.copy_from_slice(&self.data);
        Ok(())
    }

    fn copy_from_host(&mut self, host: &[u8]) -> Result<(), crate::Error> {
        if host.len() != self.size {
            return Err(crate::Error::MemoryError("Size mismatch".into()));
        }
        self.data.copy_from_slice(host);
        Ok(())
    }

    fn copy_to_device(&self, dest: &mut dyn Buffer) -> Result<(), crate::Error> {
        if dest.device() != BackendType::Cpu {
            return Err(crate::Error::NotSupported(
                "Cross-device copy not implemented".into(),
            ));
        }
        dest.copy_from_host(&self.data)
    }
}

pub struct CpuTexture {
    width: u32,
    height: u32,
    channels: u32,
    format: TextureFormat,
    data: Vec<u8>,
}

impl CpuTexture {
    pub fn new(width: u32, height: u32, channels: u32, format: TextureFormat) -> Self {
        let bytes_per_pixel = match format {
            TextureFormat::R8Unorm => 1,
            TextureFormat::Rg8Unorm => 2,
            TextureFormat::Rgba8Unorm => 4,
            TextureFormat::R16Float => 2,
            TextureFormat::Rg16Float => 4,
            TextureFormat::Rgba16Float => 8,
            TextureFormat::R32Float => 4,
            TextureFormat::Rg32Float => 8,
            TextureFormat::Rgba32Float => 16,
        };

        Self {
            width,
            height,
            channels,
            format,
            data: vec![0u8; (width * height * channels) as usize * bytes_per_pixel],
        }
    }

    pub fn from_image(data: &[u8], width: u32, height: u32, channels: u32) -> Self {
        let format = match channels {
            1 => TextureFormat::R8Unorm,
            2 => TextureFormat::Rg8Unorm,
            4 => TextureFormat::Rgba8Unorm,
            _ => TextureFormat::R8Unorm,
        };

        Self {
            width,
            height,
            channels,
            format,
            data: data.to_vec(),
        }
    }
}

impl Texture for CpuTexture {
    fn width(&self) -> u32 {
        self.width
    }

    fn height(&self) -> u32 {
        self.height
    }

    fn channels(&self) -> u32 {
        self.channels
    }

    fn format(&self) -> TextureFormat {
        self.format
    }

    fn device(&self) -> BackendType {
        BackendType::Cpu
    }
}

pub struct CpuCommandEncoder {
    copies: Vec<CopyOp>,
}

#[derive(Clone, Copy)]
enum CopyOp {
    BufferToBuffer { src: usize, dst: usize, size: usize },
}

impl CpuCommandEncoder {
    pub fn new() -> Self {
        Self { copies: Vec::new() }
    }

    pub fn finish(self) -> Vec<CopyOp> {
        self.copies
    }
}

impl CommandEncoder for CpuCommandEncoder {
    fn copy_buffer_to_buffer(
        &mut self,
        src: &dyn Buffer,
        dst: &mut dyn Buffer,
        size: usize,
    ) -> Result<(), crate::Error> {
        let src_data = src.as_slice();
        let dst_data = dst.as_mut_slice();

        if size > src_data.len() || size > dst_data.len() {
            return Err(crate::Error::MemoryError("Copy size exceeds buffer".into()));
        }

        dst_data[..size].copy_from_slice(&src_data[..size]);
        Ok(())
    }

    fn copy_buffer_to_texture(
        &mut self,
        _src: &dyn Buffer,
        _dst: &mut dyn Texture,
    ) -> Result<(), crate::Error> {
        Err(crate::Error::NotSupported(
            "Buffer to texture copy not implemented for CPU".into(),
        ))
    }

    fn copy_texture_to_buffer(
        &mut self,
        _src: &dyn Texture,
        _dst: &mut dyn Buffer,
    ) -> Result<(), crate::Error> {
        Err(crate::Error::NotSupported(
            "Texture to buffer copy not implemented for CPU".into(),
        ))
    }

    fn dispatch(
        &mut self,
        _pipeline: &dyn ComputePipeline,
        _workgroups: (u32, u32, u32),
    ) -> Result<(), crate::Error> {
        Err(crate::Error::NotSupported(
            "GPU dispatch not available on CPU backend".into(),
        ))
    }

    fn dispatch_indirect(
        &mut self,
        _pipeline: &dyn ComputePipeline,
        _indirect_buffer: &dyn Buffer,
        _offset: usize,
    ) -> Result<(), crate::Error> {
        Err(crate::Error::NotSupported(
            "Indirect dispatch not available on CPU backend".into(),
        ))
    }
}

pub struct CpuDevice {
    backend: Arc<CpuBackend>,
}

impl CpuDevice {
    pub fn new() -> Option<Self> {
        Some(Self {
            backend: Arc::new(CpuBackend::new()?),
        })
    }
}

impl Device for CpuDevice {
    fn backend(&self) -> BackendType {
        BackendType::Cpu
    }

    fn create_buffer(&self, desc: &BufferDescriptor<'_>) -> Result<Box<dyn Buffer>, crate::Error> {
        Ok(Box::new(CpuBuffer::new(desc.size)))
    }

    fn create_texture(
        &self,
        desc: &crate::backend::TextureDescriptor,
    ) -> Result<Box<dyn Texture>, crate::Error> {
        Ok(Box::new(CpuTexture::new(
            desc.width,
            desc.height,
            desc.channels,
            desc.format,
        )))
    }

    fn create_command_encoder(&self) -> Result<Box<dyn CommandEncoder>, crate::Error> {
        Ok(Box::new(CpuCommandEncoder::new()))
    }

    fn wait(&self) -> Result<(), crate::Error> {
        Ok(())
    }
}

#[cfg(feature = "simd")]
pub mod simd {
    use wide::f32x4;

    pub fn add_f32(a: &[f32], b: &[f32], out: &mut [f32]) {
        assert_eq!(a.len(), b.len());
        assert_eq!(a.len(), out.len());

        for i in (0..a.len()).step_by(4) {
            let av = f32x4::from_slice(&a[i.min(a.len())..]);
            let bv = f32x4::from_slice(&b[i.min(b.len())..]);
            let ov = av + bv;
            ov.write_slice(&mut out[i..]);
        }
    }

    pub fn mul_f32(a: &[f32], b: &[f32], out: &mut [f32]) {
        assert_eq!(a.len(), b.len());
        assert_eq!(a.len(), out.len());

        for i in (0..a.len()).step_by(4) {
            let av = f32x4::load_or_default(&a[i.min(a.len())..]);
            let bv = f32x4::load_or_default(&b[i.min(b.len())..]);
            let ov = av * bv;
            ov.write_slice(&mut out[i..]);
        }
    }

    pub fn convolve_3x3_f32(
        input: &[f32],
        output: &mut [f32],
        width: usize,
        height: usize,
        kernel: &[f32; 9],
    ) {
        let k = f32x4::from_array([kernel[0], kernel[1], kernel[2], kernel[3]]);
        let k4 = f32x4::splat(kernel[4]);

        for y in 1..height - 1 {
            for x in 1..width - 1 {
                let idx = y * width + x;
                let mut sum = f32x4::ZERO;

                for ky in 0..3 {
                    for kx in 0..3 {
                        let ki = (y + ky - 1) * width + (x + kx - 1);
                        let kv = f32x4::splat(input[ki]);
                        sum += kv * k;
                    }
                }

                output[idx] = sum.reduce_sum();
            }
        }
    }
}

#[cfg(not(feature = "simd"))]
pub mod simd {
    pub fn add_f32(a: &[f32], b: &[f32], out: &mut [f32]) {
        for i in 0..a.len().min(b.len()).min(out.len()) {
            out[i] = a[i] + b[i];
        }
    }

    pub fn mul_f32(a: &[f32], b: &[f32], out: &mut [f32]) {
        for i in 0..a.len().min(b.len()).min(out.len()) {
            out[i] = a[i] * b[i];
        }
    }

    pub fn convolve_3x3_f32(
        input: &[f32],
        output: &mut [f32],
        width: usize,
        height: usize,
        kernel: &[f32; 9],
    ) {
        for y in 1..height - 1 {
            for x in 1..width - 1 {
                let idx = y * width + x;
                let mut sum = 0.0f32;

                for ky in 0..3 {
                    for kx in 0..3 {
                        let ki = (y + ky - 1) * width + (x + kx - 1);
                        sum += input[ki] * kernel[ky * 3 + kx];
                    }
                }

                output[idx] = sum;
            }
        }
    }
}
