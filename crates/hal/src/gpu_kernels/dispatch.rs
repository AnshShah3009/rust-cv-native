//! Fluent builder for GPU compute shader dispatch.
//!
//! Eliminates the ~80-line boilerplate repeated in every kernel wrapper:
//! output buffer creation → params buffer → pipeline → bind group → encoder → dispatch → submit.
//!
//! # Example
//!
//! ```ignore
//! let outputs = GpuDispatch::new(ctx, include_str!("../../shaders/subtract.wgsl"), "Subtract")
//!     .input(a.storage.buffer())
//!     .input(b.storage.buffer())
//!     .output((size * 4) as u64)
//!     .dispatch_1d(size as u32)?;
//! ```

use crate::gpu::GpuContext;
use std::sync::Arc;
use wgpu::util::DeviceExt;

/// Fluent builder that encapsulates the standard GPU kernel dispatch pattern.
///
/// Bindings are assigned sequentially: inputs first, then outputs, then params.
pub struct GpuDispatch<'a> {
    ctx: &'a GpuContext,
    shader_source: &'a str,
    label: &'a str,
    inputs: Vec<&'a wgpu::Buffer>,
    output_byte_sizes: Vec<u64>,
    params_bytes: Option<Vec<u8>>,
}

impl<'a> GpuDispatch<'a> {
    /// Create a new dispatch builder for the given shader source.
    pub fn new(ctx: &'a GpuContext, shader_source: &'a str, label: &'a str) -> Self {
        Self {
            ctx,
            shader_source,
            label,
            inputs: Vec::new(),
            output_byte_sizes: Vec::new(),
            params_bytes: None,
        }
    }

    /// Add an input buffer (read-only storage, binding N).
    pub fn input(mut self, buffer: &'a wgpu::Buffer) -> Self {
        self.inputs.push(buffer);
        self
    }

    /// Add an output buffer of the given byte size (read-write storage, binding N).
    pub fn output(mut self, byte_size: u64) -> Self {
        self.output_byte_sizes.push(byte_size);
        self
    }

    /// Set the uniform parameters buffer from a Pod struct.
    pub fn params<P: bytemuck::Pod>(mut self, p: &P) -> Self {
        self.params_bytes = Some(bytemuck::bytes_of(p).to_vec());
        self
    }

    /// Dispatch as a 1D workgroup grid and return the output buffers.
    pub fn dispatch_1d(self, count: u32) -> crate::Result<Vec<Arc<wgpu::Buffer>>> {
        let wg_x = count.div_ceil(super::WORKGROUP_SIZE_1D);
        self.dispatch_raw(wg_x, 1, 1)
    }

    /// Dispatch as a 2D workgroup grid and return the output buffers.
    pub fn dispatch_2d(self, width: u32, height: u32) -> crate::Result<Vec<Arc<wgpu::Buffer>>> {
        let wg_x = width.div_ceil(super::WORKGROUP_SIZE_2D);
        let wg_y = height.div_ceil(super::WORKGROUP_SIZE_2D);
        self.dispatch_raw(wg_x, wg_y, 1)
    }

    /// Dispatch with explicit workgroup counts and return the output buffers.
    pub fn dispatch_raw(
        self,
        wg_x: u32,
        wg_y: u32,
        wg_z: u32,
    ) -> crate::Result<Vec<Arc<wgpu::Buffer>>> {
        let device = &self.ctx.device;

        // Create output buffers
        let output_buffers: Vec<Arc<wgpu::Buffer>> = self
            .output_byte_sizes
            .iter()
            .enumerate()
            .map(|(i, &size)| {
                Arc::new(device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some(&format!("{} Output {}", self.label, i)),
                    size,
                    usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
                    mapped_at_creation: false,
                }))
            })
            .collect();

        // Create params buffer if needed
        let params_buffer = self.params_bytes.as_ref().map(|bytes| {
            device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some(&format!("{} Params", self.label)),
                contents: bytes,
                usage: wgpu::BufferUsages::UNIFORM,
            })
        });

        // Create pipeline (cached by GpuContext)
        let pipeline = self.ctx.create_compute_pipeline(self.shader_source, "main");

        // Build bind group entries: inputs → outputs → params
        let mut entries = Vec::new();
        let mut binding = 0u32;

        for buf in &self.inputs {
            entries.push(wgpu::BindGroupEntry {
                binding,
                resource: buf.as_entire_binding(),
            });
            binding += 1;
        }

        for buf in &output_buffers {
            entries.push(wgpu::BindGroupEntry {
                binding,
                resource: buf.as_entire_binding(),
            });
            binding += 1;
        }

        if let Some(ref pb) = params_buffer {
            entries.push(wgpu::BindGroupEntry {
                binding,
                resource: pb.as_entire_binding(),
            });
        }

        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some(&format!("{} Bind Group", self.label)),
            layout: &pipeline.get_bind_group_layout(0),
            entries: &entries,
        });

        // Encode and dispatch
        let mut encoder =
            device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some(self.label),
                timestamp_writes: None,
            });
            pass.set_pipeline(&pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.dispatch_workgroups(wg_x, wg_y, wg_z);
        }
        self.ctx.submit(encoder);

        Ok(output_buffers)
    }
}
