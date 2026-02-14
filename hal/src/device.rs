use crate::backend::{BackendType, Capability, Device};
use std::sync::Arc;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BackendPriority {
    Performance,
    Compatibility,
    Embedded,
}

pub struct DeviceManager {
    backends: Vec<BackendInfo>,
    selected_backend: Option<BackendType>,
}

#[derive(Debug, Clone)]
pub struct BackendInfo {
    pub backend_type: BackendType,
    pub name: String,
    pub priority: i32,
    pub available: bool,
    pub capabilities: Vec<Capability>,
}

impl DeviceManager {
    pub fn new() -> Self {
        Self {
            backends: Vec::new(),
            selected_backend: None,
        }
    }

    pub fn detect_available(&mut self) -> &Self {
        self.backends.clear();

        if crate::cpu::CpuBackend::is_available() {
            self.backends.push(BackendInfo {
                backend_type: BackendType::Cpu,
                name: "CPU".to_string(),
                priority: 100,
                available: true,
                capabilities: vec![Capability::Compute, Capability::Simd],
            });
        }

        #[cfg(feature = "cuda")]
        if ort::CudaBackend::is_available() {
            self.backends.push(BackendInfo {
                backend_type: BackendType::Cuda,
                name: "CUDA".to_string(),
                priority: 80,
                available: true,
                capabilities: vec![
                    Capability::Compute,
                    Capability::Simd,
                    Capability::TensorCore,
                ],
            });
        }

        #[cfg(feature = "webgpu")]
        if crate::wgpu::WgpuBackend::is_available() {
            self.backends.push(BackendInfo {
                backend_type: BackendType::WebGPU,
                name: "WebGPU".to_string(),
                priority: 70,
                available: true,
                capabilities: vec![Capability::Compute, Capability::Simd],
            });
        }

        #[cfg(feature = "directml")]
        if crate::directml::DirectMLBackend::is_available() {
            self.backends.push(BackendInfo {
                backend_type: BackendType::DirectML,
                name: "DirectML".to_string(),
                priority: 60,
                available: true,
                capabilities: vec![Capability::Compute, Capability::Simd],
            });
        }

        self
    }

    pub fn select_backend(&mut self, backend: BackendType) -> Result<&BackendInfo, crate::Error> {
        self.selected_backend = Some(backend);

        self.backends
            .iter()
            .find(|b| b.backend_type == backend && b.available)
            .ok_or_else(|| crate::Error::backend_not_available(backend.to_string()))
    }

    pub fn auto_select(&mut self, priority: BackendPriority) -> Result<&BackendInfo, crate::Error> {
        self.detect_available();

        if self.backends.is_empty() {
            return Err(crate::Error::backend_not_available("No backends available"));
        }

        let selected = match priority {
            BackendPriority::Performance => self
                .backends
                .iter()
                .filter(|b| b.available)
                .max_by_key(|b| b.priority),
            BackendPriority::Compatibility => self
                .backends
                .iter()
                .find(|b| b.backend_type == BackendType::Cpu)
                .or_else(|| self.backends.first()),
            BackendPriority::Embedded => self
                .backends
                .iter()
                .find(|b| b.backend_type == BackendType::Cpu)
                .ok_or_else(|| self.backends.first()),
        };

        match selected {
            Some(b) => {
                self.selected_backend = Some(b.backend_type);
                Ok(b)
            }
            None => Err(crate::Error::backend_not_available(
                "No suitable backend found",
            )),
        }
    }

    pub fn create_device(&self, backend: BackendType) -> Result<Box<dyn Device>, crate::Error> {
        match backend {
            BackendType::Cpu => crate::cpu::CpuDevice::new()
                .ok_or_else(|| crate::Error::InitError("Failed to create CPU device".into())),
            #[cfg(feature = "webgpu")]
            BackendType::WebGPU | BackendType::Vulkan | BackendType::Metal => {
                crate::wgpu::create_device(backend)
            }
            #[cfg(feature = "cuda")]
            BackendType::Cuda | BackendType::TensorRT => Err(crate::Error::NotSupported(
                "CUDA backend not yet implemented".into(),
            )),
            #[cfg(feature = "directml")]
            BackendType::DirectML => Err(crate::Error::NotSupported(
                "DirectML backend not yet implemented".into(),
            )),
            #[cfg(not(any(feature = "cuda", feature = "directml", feature = "webgpu")))]
            _ => Err(crate::Error::NotSupported(format!(
                "Backend {:?} not enabled. Enable with feature flags: cuda, directml, webgpu",
                backend
            ))),
        }
    }

    pub fn available_backends(&self) -> &[BackendInfo] {
        &self.backends
    }

    pub fn selected(&self) -> Option<BackendType> {
        self.selected_backend
    }
}

impl Default for DeviceManager {
    fn default() -> Self {
        Self::new()
    }
}

pub fn detect_hardware() -> DeviceManager {
    let mut manager = DeviceManager::new();
    manager.detect_available();
    manager
}

pub fn get_best_backend() -> BackendType {
    let manager = detect_hardware();

    manager
        .available_backends()
        .iter()
        .filter(|b| b.available)
        .max_by_key(|b| b.priority)
        .map(|b| b.backend_type)
        .unwrap_or(BackendType::Cpu)
}

pub fn get_cpu_only() -> BackendType {
    BackendType::Cpu
}
