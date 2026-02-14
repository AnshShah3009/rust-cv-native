use std::fmt;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum BackendType {
    Cpu,
    Cuda,
    Vulkan,
    Metal,
    DirectML,
    TensorRT,
    WebGPU,
}

impl fmt::Display for BackendType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            BackendType::Cpu => write!(f, "CPU"),
            BackendType::Cuda => write!(f, "CUDA"),
            BackendType::Vulkan => write!(f, "Vulkan"),
            BackendType::Metal => write!(f, "Metal"),
            BackendType::DirectML => write!(f, "DirectML"),
            BackendType::TensorRT => write!(f, "TensorRT"),
            BackendType::WebGPU => write!(f, "WebGPU"),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Capability {
    Compute,
    Simd,
    TensorCore,
    RayTracing,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QueueType {
    Compute,
    Transfer,
    Graphics,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Priority {
    Low,
    Medium,
    High,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DeviceId(pub u32);

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct QueueId(pub u32);

pub trait ComputeBackend: Send + Sync {
    fn backend_type(&self) -> BackendType;

    fn name(&self) -> &str;

    fn device_id(&self) -> DeviceId;

    fn supports(&self, capability: Capability) -> bool;

    fn queue(&self, queue_type: QueueType) -> QueueId;

    fn preferred_queue(&self) -> QueueType;

    fn is_available() -> bool
    where
        Self: Sized;

    fn try_create() -> Option<Self>
    where
        Self: Sized;
}

pub trait Buffer: Send + Sync {
    fn size(&self) -> usize;

    fn as_slice(&self) -> &[u8];

    fn as_mut_slice(&mut self) -> &mut [u8];

    fn device(&self) -> BackendType;

    fn copy_to_host(&self, host: &mut [u8]) -> Result<(), super::Error>;

    fn copy_from_host(&mut self, host: &[u8]) -> Result<(), super::Error>;

    fn copy_to_device(&self, dest: &mut dyn Buffer) -> Result<(), super::Error>;
}

pub trait Texture: Send + Sync {
    fn width(&self) -> u32;

    fn height(&self) -> u32;

    fn channels(&self) -> u32;

    fn format(&self) -> TextureFormat;

    fn device(&self) -> BackendType;
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TextureFormat {
    R8Unorm,
    Rg8Unorm,
    Rgba8Unorm,
    R16Float,
    Rg16Float,
    Rgba16Float,
    R32Float,
    Rg32Float,
    Rgba32Float,
}

pub trait CommandEncoder {
    fn copy_buffer_to_buffer(
        &mut self,
        src: &dyn Buffer,
        dst: &mut dyn Buffer,
        size: usize,
    ) -> Result<(), super::Error>;

    fn copy_buffer_to_texture(
        &mut self,
        src: &dyn Buffer,
        dst: &mut dyn Texture,
    ) -> Result<(), super::Error>;

    fn copy_texture_to_buffer(
        &mut self,
        src: &dyn Texture,
        dst: &mut dyn Buffer,
    ) -> Result<(), super::Error>;

    fn dispatch(
        &mut self,
        pipeline: &dyn ComputePipeline,
        workgroups: (u32, u32, u32),
    ) -> Result<(), super::Error>;

    fn dispatch_indirect(
        &mut self,
        pipeline: &dyn ComputePipeline,
        indirect_buffer: &dyn Buffer,
        offset: usize,
    ) -> Result<(), super::Error>;
}

pub trait ComputePipeline: Send + Sync {
    fn bind_group_layout(&self) -> &BindGroupLayout;

    fn bind(&self, index: u32, resource: BindResource);
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum BindResourceType {
    Buffer,
    Texture,
    Sampler,
}

pub trait BindGroupLayout: Send + Sync {
    fn bindings(&self) -> &[(u32, BindResourceType)];
}

pub struct BufferDescriptor<'a> {
    pub name: &'a str,
    pub size: usize,
    pub usage: BufferUsage,
    pub mapped_at_creation: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub struct BufferUsage(u8);

impl BufferUsage {
    pub const READ: Self = Self(1 << 0);
    pub const WRITE: Self = Self(1 << 1);
    pub const COPY_SRC: Self = Self(1 << 2);
    pub const COPY_DST: Self = Self(1 << 3);
    pub const INDEX: Self = Self(1 << 4);
    pub const VERTEX: Self = Self(1 << 5);
    pub const UNIFORM: Self = Self(1 << 6);
    pub const STORAGE: Self = Self(1 << 7);

    pub fn contains(self, other: Self) -> bool {
        (self.0 & other.0) == other.0
    }
}

impl<'a> BufferDescriptor<'a> {
    pub fn new(name: &'a str, size: usize) -> Self {
        Self {
            name,
            size,
            usage: BufferUsage::READ | BufferUsage::WRITE,
            mapped_at_creation: false,
        }
    }

    pub fn with_usage(mut self, usage: BufferUsage) -> Self {
        self.usage = usage;
        self
    }

    pub fn storage() -> Self {
        Self::new("buffer", 0)
            .with_usage(BufferUsage::STORAGE | BufferUsage::READ | BufferUsage::WRITE)
    }
}

pub trait Device: Send + Sync {
    fn backend(&self) -> BackendType;

    fn create_buffer(&self, desc: &BufferDescriptor<'_>) -> Result<Box<dyn Buffer>, super::Error>;

    fn create_texture(&self, desc: &TextureDescriptor) -> Result<Box<dyn Texture>, super::Error>;

    fn create_command_encoder(&self) -> Result<Box<dyn CommandEncoder>, super::Error>;

    fn wait(&self) -> Result<(), super::Error>;
}
