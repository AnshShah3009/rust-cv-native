use crate::backend::{Buffer, BufferDescriptor, BufferUsage, Device, TextureFormat};

pub struct MemoryArena {
    buffers: Vec<AllocatedBuffer>,
    total_allocated: usize,
    max_size: usize,
}

struct AllocatedBuffer {
    ptr: *mut u8,
    size: usize,
    device_buffer: Option<Box<dyn Buffer>>,
}

impl MemoryArena {
    pub fn new(max_size: usize) -> Self {
        Self {
            buffers: Vec::new(),
            total_allocated: 0,
            max_size,
        }
    }

    pub fn allocate(
        &mut self,
        device: &dyn Device,
        desc: &BufferDescriptor,
    ) -> Result<BufferId, crate::Error> {
        if self.total_allocated + desc.size > self.max_size {
            return Err(crate::Error::MemoryError("Arena out of memory".into()));
        }

        let buffer = device.create_buffer(desc)?;
        let id = BufferId(self.buffers.len());

        self.buffers.push(AllocatedBuffer {
            ptr: std::ptr::null_mut(),
            size: desc.size,
            device_buffer: Some(buffer),
        });

        self.total_allocated += desc.size;
        Ok(id)
    }

    pub fn get(&self, id: BufferId) -> Option<&dyn Buffer> {
        self.buffers
            .get(id.0)
            .and_then(|b| b.device_buffer.as_deref())
    }

    pub fn get_mut(&mut self, id: BufferId) -> Option<&mut dyn Buffer> {
        self.buffers
            .get_mut(id.0)
            .and_then(|b| b.device_buffer.as_deref_mut())
    }

    pub fn reset(&mut self) {
        self.buffers.clear();
        self.total_allocated = 0;
    }

    pub fn allocated_size(&self) -> usize {
        self.total_allocated
    }

    pub fn max_size(&self) -> usize {
        self.max_size
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct BufferId(pub usize);

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct TextureId(pub usize);

pub struct TextureDescriptor {
    pub width: u32,
    pub height: u32,
    pub channels: u32,
    pub format: TextureFormat,
    pub usage: TextureUsage,
    pub mip_level_count: u32,
    pub sample_count: u32,
}

impl TextureDescriptor {
    pub fn new_2d(width: u32, height: u32, channels: u32) -> Self {
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
            usage: TextureUsage::TEXTURE_BINDING | TextureUsage::COPY_DST | TextureUsage::COPY_SRC,
            mip_level_count: 1,
            sample_count: 1,
        }
    }

    pub fn with_format(mut self, format: TextureFormat) -> Self {
        self.format = format;
        self
    }

    pub fn with_mip_levels(mut self, count: u32) -> Self {
        self.mip_level_count = count;
        self
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub struct TextureUsage(u8);

impl TextureUsage {
    pub const COPY_SRC: Self = Self(1 << 0);
    pub const COPY_DST: Self = Self(1 << 1);
    pub const TEXTURE_BINDING: Self = Self(1 << 2);
    pub const STORAGE_BINDING: Self = Self(1 << 3);
    pub const RENDER_ATTACHMENT: Self = Self(1 << 4);

    pub fn contains(self, other: Self) -> bool {
        (self.0 & other.0) == other.0
    }
}

pub struct ImageTile {
    pub x: u32,
    pub y: u32,
    pub width: u32,
    pub height: u32,
}

impl ImageTile {
    pub fn new(x: u32, y: u32, width: u32, height: u32) -> Self {
        Self {
            x,
            y,
            width,
            height,
        }
    }

    pub fn from_image(width: u32, height: u32, tile_size: u32) -> Vec<Self> {
        let mut tiles = Vec::new();

        let tiles_x = (width + tile_size - 1) / tile_size;
        let tiles_y = (height + tile_size - 1) / tile_size;

        for ty in 0..tiles_y {
            for tx in 0..tiles_x {
                let x = tx * tile_size;
                let y = ty * tile_size;
                let w = tile_size.min(width.saturating_sub(x));
                let h = tile_size.min(height.saturating_sub(y));

                tiles.push(Self::new(x, y, w, h));
            }
        }

        tiles
    }
}

pub fn compute_tile_count(total_pixels: usize, tile_size: usize) -> usize {
    (total_pixels + tile_size - 1) / tile_size
}

pub fn get_optimal_tile_size(width: u32, height: u32, bytes_per_pixel: usize) -> u32 {
    let pixels = (width * height) as usize;
    let bytes = pixels * bytes_per_pixel;

    if bytes < 256 * 1024 {
        128
    } else if bytes < 1024 * 1024 {
        256
    } else if bytes < 4 * 1024 * 1024 {
        512
    } else {
        1024
    }
}
