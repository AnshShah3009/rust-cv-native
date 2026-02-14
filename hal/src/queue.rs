use crate::backend::{BackendType, CommandEncoder, Device, QueueId, QueueType};

pub struct CommandQueue {
    device: Box<dyn Device>,
    queue_type: QueueType,
    pending_commands: Vec<PendingCommand>,
}

enum PendingCommand {
    Submit(Vec<Box<dyn CommandEncoder>>),
    Wait,
}

impl CommandQueue {
    pub fn new(device: Box<dyn Device>, queue_type: QueueType) -> Self {
        Self {
            device,
            queue_type,
            pending_commands: Vec::new(),
        }
    }

    pub fn submit(&mut self, encoder: Box<dyn CommandEncoder>) {
        self.pending_commands
            .push(PendingCommand::Submit(vec![encoder]));
    }

    pub fn submit_many(&mut self, encoders: Vec<Box<dyn CommandEncoder>>) {
        self.pending_commands.push(PendingCommand::Submit(encoders));
    }

    pub fn wait(&mut self) {
        self.pending_commands.push(PendingCommand::Wait);
    }

    pub fn flush(&mut self) -> Result<(), crate::Error> {
        for cmd in self.pending_commands.drain(..) {
            match cmd {
                PendingCommand::Submit(_encoders) => {
                    self.device.wait()?;
                }
                PendingCommand::Wait => {
                    self.device.wait()?;
                }
            }
        }
        Ok(())
    }

    pub fn queue_type(&self) -> QueueType {
        self.queue_type
    }

    pub fn backend(&self) -> BackendType {
        self.device.backend()
    }
}

pub struct Fence {
    signaled: bool,
    device: Box<dyn Device>,
}

impl Fence {
    pub fn new(device: Box<dyn Device>) -> Self {
        Self {
            signaled: false,
            device,
        }
    }

    pub fn signal(&mut self) {
        self.signaled = true;
    }

    pub fn wait(&self) -> Result<(), crate::Error> {
        if !self.signaled {
            self.device.wait()?;
        }
        Ok(())
    }

    pub fn is_signaled(&self) -> bool {
        self.signaled
    }
}

pub struct Timeline {
    device: Box<dyn Device>,
    current_value: u64,
}

impl Timeline {
    pub fn new(device: Box<dyn Device>) -> Self {
        Self {
            device,
            current_value: 0,
        }
    }

    pub fn signal(&mut self, value: u64) -> Result<Fence, crate::Error> {
        if value <= self.current_value {
            return Err(crate::Error::QueueError(
                "Timeline value must increase".into(),
            ));
        }

        self.current_value = value;
        let fence = Fence::new(self.device.box_clone());
        Ok(fence)
    }

    pub fn wait(&self, value: u64) -> Result<(), crate::Error> {
        if value > self.current_value {
            return Err(crate::Error::QueueError(
                "Cannot wait for future timeline value".into(),
            ));
        }

        self.device.wait()
    }

    pub fn current_value(&self) -> u64 {
        self.current_value
    }
}

pub trait CommandBuffer: Send + Sync {
    fn encode(&mut self, encoder: &mut dyn CommandEncoder);

    fn submit(&self, queue: &mut CommandQueue) -> Result<(), crate::Error>;
}

pub trait ComputePass {
    fn set_pipeline(&mut self, pipeline: &dyn crate::backend::ComputePipeline);

    fn set_bind_group(&mut self, index: u32, bind_group: &dyn crate::backend::BindGroupLayout);

    fn dispatch(&mut self, x: u32, y: u32, z: u32);

    fn dispatch_workgroups(&mut self, workgroup_count: (u32, u32, u32));
}
