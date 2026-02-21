use cv_hal::DeviceId;
use crate::{Result, RuntimeRunner};
use std::collections::HashMap;

/// Identifier for a buffer within a pipeline.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct BufferId(pub u32);

/// A node in a compute pipeline.
pub enum PipelineNode {
    /// A single kernel execution.
    Kernel {
        name: String,
        inputs: Vec<BufferId>,
        outputs: Vec<BufferId>,
        params: Vec<u8>,
    },
    /// A custom CPU function that operates on pipeline buffers.
    CpuOp {
        inputs: Vec<BufferId>,
        outputs: Vec<BufferId>,
        op: Box<dyn FnOnce(&[u8]) -> Vec<u8> + Send>,
    },
}

/// A declarative compute pipeline that can be executed as a unit.
/// This allows the orchestrator to optimize data flow (e.g., keeping data on GPU).
pub struct Pipeline {
    nodes: Vec<PipelineNode>,
    buffers: HashMap<BufferId, usize>, // BufferId -> Size in bytes
    preferred_device: Option<DeviceId>,
}

impl Pipeline {
    pub fn new() -> Self {
        Self {
            nodes: Vec::new(),
            buffers: HashMap::new(),
            preferred_device: None,
        }
    }

    pub fn register_buffer(&mut self, id: BufferId, size: usize) {
        self.buffers.insert(id, size);
    }

    pub fn add_kernel(
        mut self, 
        name: &str, 
        inputs: &[BufferId], 
        outputs: &[BufferId], 
        params: Vec<u8>
    ) -> Self {
        self.nodes.push(PipelineNode::Kernel {
            name: name.to_string(),
            inputs: inputs.to_vec(),
            outputs: outputs.to_vec(),
            params,
        });
        self
    }

    pub fn execute(self, runner: &RuntimeRunner) -> Result<()> {
        runner.run(move || {
            let _device = runner.device();
            // TODO: In a real implementation, this would:
            // 1. Allocate/Reuse UnifiedBuffers for each BufferId.
            // 2. Translate Kernel nodes into a single CommandEncoder submission.
            // 3. Handle synchronization between CPU and GPU nodes.
            Ok(())
        })
    }
}

/// A trait for builders that can construct a pipeline.
pub trait PipelineBuilder {
    fn build(&self) -> Pipeline;
}
