//! Observability subsystem: events, metrics, and submission tracking.

pub mod events;
pub mod layer;
pub mod metrics;
pub mod submission;

pub use events::{DeviceHealth, MemoryEventKind, RuntimeEvent};
pub use layer::{observability, ObservabilityLayer};
pub use metrics::Metrics;
pub use submission::SubmissionContext;
