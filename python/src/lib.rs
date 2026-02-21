use cv_core::CameraIntrinsics;
use cv_runtime::orchestrator::{WorkloadHint, scheduler};
use cv_slam::Slam as RustSlam;
use numpy::{PyReadonlyArray2, PyUntypedArrayMethods};
use pyo3::prelude::*;
use std::sync::Arc;

#[pyclass]
#[derive(Clone, Copy)]
pub enum PyWorkloadHint {
    Latency,
    Throughput,
    PowerSave,
    Default,
}

impl From<PyWorkloadHint> for WorkloadHint {
    fn from(hint: PyWorkloadHint) -> Self {
        match hint {
            PyWorkloadHint::Latency => WorkloadHint::Latency,
            PyWorkloadHint::Throughput => WorkloadHint::Throughput,
            PyWorkloadHint::PowerSave => WorkloadHint::PowerSave,
            PyWorkloadHint::Default => WorkloadHint::Default,
        }
    }
}

#[pyclass]
pub struct PySlam {
    inner: RustSlam,
    _group: Arc<cv_runtime::ResourceGroup>,
}

#[pymethods]
impl PySlam {
    #[new]
    pub fn new(hint: PyWorkloadHint) -> PyResult<Self> {
        let s = scheduler().map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
        let group = s.best_gpu_or_cpu_for(hint.into())
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
        
        let intrinsics = CameraIntrinsics::new_ideal(640, 480); // Default
        Ok(Self {
            inner: RustSlam::new(group.clone(), intrinsics),
            _group: group,
        })
    }

    pub fn process_frame(&mut self, image: PyReadonlyArray2<u8>) -> PyResult<(Vec<f32>, Vec<usize>)> {
        let shape = image.shape();
        let rows = shape[0];
        let cols = shape[1];
        
        let mut gray = image::GrayImage::new(cols as u32, rows as u32);
        gray.copy_from_slice(image.as_slice()?);
        
        let (pose, tracked) = self.inner.process_image(&gray)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e))?;
        
        // Convert Pose to flat Vec for Python (4x4 matrix)
        let mat = pose.matrix();
        let flat_mat: Vec<f32> = mat.iter().map(|&v| v as f32).collect();
        
        Ok((flat_mat, tracked))
    }
}

#[pyclass]
pub struct PyRuntime {
}

#[pymethods]
impl PyRuntime {
    #[staticmethod]
    pub fn get_global_load() -> PyResult<Vec<(u32, usize)>> {
        let _s = scheduler().map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
        Ok(vec![])
    }
}

#[pymodule]
fn cv_native(_py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyWorkloadHint>()?;
    m.add_class::<PySlam>()?;
    m.add_class::<PyRuntime>()?;
    Ok(())
}
