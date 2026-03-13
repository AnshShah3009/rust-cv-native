//! Dataflow pipeline for composing and executing kernel operations.

mod allocator;
mod fusion;
mod graph;
mod handle;

pub use allocator::{BufferAlloc, TransientBufferPool};
pub use fusion::{FusedKernel, FusionPattern, KernelFuser};
pub use graph::{ExecutionGraph, NodeDependency, NodeId};
pub use handle::{AsyncPipelineHandle, ExecutionEvent, PipelineResult};

use crate::device_registry::{registry, BackendContext};
use crate::orchestrator::RuntimeRunner;
use crate::Result;
use cv_hal::context::ComputeContext;
use cv_hal::DeviceId;
use std::collections::HashMap;
use std::sync::Arc;

/// Trait for dispatching named kernel operations to a compute backend.
///
/// Implementations map kernel name strings (e.g. "threshold", "conv2d") to
/// actual HAL backend calls, bridging the abstract pipeline with concrete
/// GPU/CPU execution.
pub trait KernelDispatcher: Send + Sync {
    /// Execute a named kernel on the given backend.
    ///
    /// - `ctx`: the backend context (CPU, GPU, or MLX)
    /// - `name`: kernel name (e.g. "threshold", "gaussian_blur")
    /// - `inputs`: byte slices of input buffer data
    /// - `outputs`: mutable output buffers to fill with results
    /// - `params`: serialized kernel parameters
    fn dispatch(
        &self,
        ctx: &BackendContext,
        name: &str,
        inputs: &[&[u8]],
        outputs: &mut [Vec<u8>],
        params: &[u8],
    ) -> crate::Result<()>;
}

/// A no-op dispatcher that records kernel names without executing them.
///
/// Useful for pipeline validation, dry-runs, and testing graph structure.
pub struct NoOpDispatcher;

impl KernelDispatcher for NoOpDispatcher {
    fn dispatch(
        &self,
        _ctx: &BackendContext,
        _name: &str,
        _inputs: &[&[u8]],
        _outputs: &mut [Vec<u8>],
        _params: &[u8],
    ) -> crate::Result<()> {
        Ok(())
    }
}

/// Opaque identifier for a buffer within a pipeline.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct BufferId(pub u32);

/// A single operation within a [`Pipeline`].
pub enum PipelineNode {
    /// A named GPU/CPU kernel dispatched via [`KernelDispatcher`].
    Kernel {
        /// Kernel name (maps to a HAL operation).
        name: String,
        /// Buffers consumed by this kernel.
        inputs: Vec<BufferId>,
        /// Buffers produced by this kernel.
        outputs: Vec<BufferId>,
        /// Serialized kernel parameters.
        params: Vec<u8>,
    },
    /// A CPU-side operation expressed as a closure.
    CpuOp {
        /// Buffers consumed by this operation.
        inputs: Vec<BufferId>,
        /// Buffers produced by this operation.
        outputs: Vec<BufferId>,
        /// The closure that performs the computation.
        #[allow(clippy::type_complexity)]
        op: Arc<dyn Fn(&[&[u8]]) -> Vec<Vec<u8>> + Send + Sync>,
    },
    /// A synchronization barrier; waits for all preceding GPU work to complete.
    Barrier,
}

impl std::fmt::Debug for PipelineNode {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            PipelineNode::Kernel {
                name,
                inputs,
                outputs,
                params,
            } => f
                .debug_struct("Kernel")
                .field("name", name)
                .field("inputs", inputs)
                .field("outputs", outputs)
                .field("params", &format!("[{} bytes]", params.len()))
                .finish(),
            PipelineNode::CpuOp {
                inputs, outputs, ..
            } => f
                .debug_struct("CpuOp")
                .field("inputs", inputs)
                .field("outputs", outputs)
                .field("op", &"<function>")
                .finish(),
            PipelineNode::Barrier => write!(f, "Barrier"),
        }
    }
}

impl Clone for PipelineNode {
    fn clone(&self) -> Self {
        match self {
            PipelineNode::Kernel {
                name,
                inputs,
                outputs,
                params,
            } => PipelineNode::Kernel {
                name: name.clone(),
                inputs: inputs.clone(),
                outputs: outputs.clone(),
                params: params.clone(),
            },
            PipelineNode::CpuOp {
                inputs,
                outputs,
                op,
            } => PipelineNode::CpuOp {
                inputs: inputs.clone(),
                outputs: outputs.clone(),
                op: op.clone(),
            },
            PipelineNode::Barrier => PipelineNode::Barrier,
        }
    }
}

/// A dataflow graph of kernel and CPU operations with automatic dependency tracking.
///
/// Build a pipeline by chaining `add_kernel` / `add_cpu_op` / `add_barrier` calls,
/// then call [`build()`](Pipeline::build) to compute the execution order and
/// [`execute()`](Pipeline::execute) to run it.
pub struct Pipeline {
    nodes: Vec<PipelineNode>,
    buffers: HashMap<BufferId, usize>,
    preferred_device: Option<DeviceId>,
    execution_graph: Option<ExecutionGraph>,
    dispatcher: Option<Arc<dyn KernelDispatcher>>,
    enable_fusion: bool,
}

impl Pipeline {
    /// Create an empty pipeline with no nodes or buffers.
    pub fn new() -> Self {
        Self {
            nodes: Vec::new(),
            buffers: HashMap::new(),
            preferred_device: None,
            execution_graph: None,
            dispatcher: None,
            enable_fusion: false,
        }
    }

    /// Set the preferred device for execution.
    pub fn with_device(mut self, device_id: DeviceId) -> Self {
        self.preferred_device = Some(device_id);
        self
    }

    /// Set the kernel dispatcher that maps kernel names to actual backend calls.
    ///
    /// Without a dispatcher, kernel nodes are no-ops (buffers allocated but not computed).
    pub fn with_dispatcher(mut self, dispatcher: Arc<dyn KernelDispatcher>) -> Self {
        self.dispatcher = Some(dispatcher);
        self
    }

    /// Enable kernel fusion optimization during `build()`.
    pub fn with_fusion(mut self, enable: bool) -> Self {
        self.enable_fusion = enable;
        self
    }

    /// Register a buffer with the given size (in bytes) for use in pipeline nodes.
    pub fn register_buffer(&mut self, id: BufferId, size: usize) {
        self.buffers.insert(id, size);
    }

    /// Append a GPU/CPU kernel node to the pipeline.
    pub fn add_kernel(
        mut self,
        name: &str,
        inputs: &[BufferId],
        outputs: &[BufferId],
        params: Vec<u8>,
    ) -> Self {
        self.nodes.push(PipelineNode::Kernel {
            name: name.to_string(),
            inputs: inputs.to_vec(),
            outputs: outputs.to_vec(),
            params,
        });
        self
    }

    /// Append a CPU-side operation node to the pipeline.
    pub fn add_cpu_op<F>(mut self, inputs: &[BufferId], outputs: &[BufferId], op: F) -> Self
    where
        F: Fn(&[&[u8]]) -> Vec<Vec<u8>> + Send + Sync + 'static,
    {
        self.nodes.push(PipelineNode::CpuOp {
            inputs: inputs.to_vec(),
            outputs: outputs.to_vec(),
            op: Arc::new(op),
        });
        self
    }

    /// Append a GPU synchronization barrier.
    pub fn add_barrier(mut self) -> Self {
        self.nodes.push(PipelineNode::Barrier);
        self
    }

    /// Finalize the pipeline: optionally fuse kernels, then build the execution graph.
    pub fn build(mut self) -> Result<Self> {
        // Optionally apply kernel fusion before building the execution graph
        if self.enable_fusion {
            let fuser = KernelFuser::new();
            self.nodes = fuser.optimize(self.nodes)?;
        }

        self.execution_graph = Some(ExecutionGraph::build(&self.nodes)?);
        Ok(self)
    }

    /// Execute the pipeline synchronously on the given runner. Must call `build()` first.
    pub fn execute(self, runner: &RuntimeRunner) -> Result<()> {
        let graph = self.execution_graph.as_ref().ok_or_else(|| {
            crate::Error::RuntimeError("Pipeline not built. Call .build() first.".into())
        })?;

        let device_id = self.preferred_device.unwrap_or_else(|| runner.device_id());
        let reg = registry()?;
        let device_runtime = reg.get_device(device_id).ok_or_else(|| {
            crate::Error::RuntimeError(format!("Device {:?} not found", device_id))
        })?;

        let allocator = TransientBufferPool::new(device_id);

        for &node_id in &graph.topology_order {
            match &self.nodes[node_id.0] {
                PipelineNode::Kernel {
                    name,
                    inputs,
                    outputs,
                    params,
                } => {
                    // Collect input data from previously computed buffers
                    let input_data: Vec<Vec<u8>> = inputs
                        .iter()
                        .filter_map(|&id| allocator.get_buffer_data(id))
                        .collect();
                    let input_slices: Vec<&[u8]> =
                        input_data.iter().map(|v| v.as_slice()).collect();

                    // Prepare output buffers
                    let output_sizes: Vec<usize> = outputs
                        .iter()
                        .filter_map(|&id| self.buffers.get(&id).copied())
                        .collect();
                    let mut output_bufs: Vec<Vec<u8>> =
                        output_sizes.iter().map(|&sz| vec![0u8; sz]).collect();

                    // Dispatch via the KernelDispatcher if available
                    if let Some(ref dispatcher) = self.dispatcher {
                        dispatcher.dispatch(
                            device_runtime.context(),
                            name,
                            &input_slices,
                            &mut output_bufs,
                            params,
                        )?;
                    }

                    // Store output data into the allocator
                    for (i, &output_id) in outputs.iter().enumerate() {
                        if i < output_bufs.len() {
                            let data = &output_bufs[i];
                            if let Some(&size) = self.buffers.get(&output_id) {
                                allocator.allocate_or_update(output_id, size, data)?;
                            }
                        }
                    }
                }
                PipelineNode::CpuOp {
                    inputs,
                    outputs,
                    op,
                } => {
                    let input_data: Vec<Vec<u8>> = inputs
                        .iter()
                        .filter_map(|&id| allocator.get_buffer_data(id))
                        .collect();

                    let input_slices: Vec<&[u8]> =
                        input_data.iter().map(|v| v.as_slice()).collect();
                    let results = op(&input_slices);

                    for (i, &output_id) in outputs.iter().enumerate() {
                        if i < results.len() {
                            if let Some(size) = self.buffers.get(&output_id) {
                                allocator.allocate_or_update(output_id, *size, &results[i])?;
                            }
                        }
                    }
                }
                PipelineNode::Barrier => {
                    if let BackendContext::Gpu(gpu_ctx) = device_runtime.context() {
                        let _ = gpu_ctx.wait_idle();
                    }
                }
            }
        }

        Ok(())
    }

    /// Spawn the pipeline on a Tokio task and return a handle for tracking progress.
    pub fn execute_async(self, runner: &RuntimeRunner) -> AsyncPipelineHandle {
        let device_id = self.preferred_device.unwrap_or_else(|| runner.device_id());
        let graph = self.execution_graph.clone();
        let buffers = self.buffers.clone();
        let nodes = self.nodes.clone();
        let dispatcher = self.dispatcher.clone();

        handle::spawn_pipeline_execution(nodes, buffers, graph, device_id, dispatcher)
    }

    /// Return the number of nodes in this pipeline.
    pub fn node_count(&self) -> usize {
        self.nodes.len()
    }

    /// Return the number of registered buffers.
    pub fn buffer_count(&self) -> usize {
        self.buffers.len()
    }
}

impl Default for Pipeline {
    fn default() -> Self {
        Self::new()
    }
}

/// Trait for types that can construct a [`Pipeline`].
pub trait PipelineBuilder {
    /// Build and return a ready-to-execute pipeline.
    fn build_pipeline(&self) -> Result<Pipeline>;
}
