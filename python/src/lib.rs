use cv_core::{CameraIntrinsics, Tensor, TensorShape, storage::CpuStorage};
use cv_runtime::orchestrator::{WorkloadHint, scheduler, RuntimeRunner};
use cv_slam::Slam as RustSlam;
use cv_videoio::{open_video, VideoCapture};
use cv_dnn::{DnnNet, DnnError};
use numpy::{PyReadonlyArray2, PyUntypedArrayMethods, PyArray3};
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
pub struct PyVideoCapture {
    inner: Box<dyn VideoCapture>,
}

#[pymethods]
impl PyVideoCapture {
    #[new]
    pub fn new(path: &str) -> PyResult<Self> {
        let cap = open_video(path).map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
        Ok(Self { inner: cap })
    }

    pub fn read(&mut self) -> PyResult<Option<Vec<u8>>> {
        match self.inner.read() {
            Ok(img) => Ok(Some(img.into_vec())),
            Err(_) => Ok(None),
        }
    }
    
    pub fn is_opened(&self) -> bool {
        self.inner.is_opened()
    }
}

#[pyclass]
pub struct PyDnnNet {
    inner: DnnNet,
    runner: Arc<cv_runtime::ResourceGroup>,
}

#[pymethods]
impl PyDnnNet {
    #[new]
    pub fn new(path: &str, hint: PyWorkloadHint) -> PyResult<Self> {
        let inner = DnnNet::load(path).map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
        let s = scheduler().map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
        let runner = s.best_gpu_or_cpu_for(hint.into())
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
            
        Ok(Self { inner, runner })
    }

    pub fn forward(&self, image: PyReadonlyArray2<u8>) -> PyResult<Vec<Vec<f32>>> {
        let shape = image.shape();
        let gray = image::GrayImage::from_raw(shape[1] as u32, shape[0] as u32, image.as_slice()?.to_vec())
            .ok_or_else(|| PyErr::new::<pyo3::exceptions::PyValueError, _>("Invalid image dimensions"))?;
        
        let dyn_img = image::DynamicImage::ImageLuma8(gray);
        
        // Preprocess
        let input_tensor = self.inner.preprocess(&dyn_img, &self.runner)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
            
        // Inference
        let outputs = self.inner.forward(&input_tensor)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
            
        // Convert outputs to Vec<Vec<f32>>
        let mut py_outputs = Vec::new();
        for tensor in outputs {
            if let Ok(slice) = tensor.as_slice() {
                py_outputs.push(slice.to_vec());
            }
        }
        
        Ok(py_outputs)
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
    m.add_class::<PyVideoCapture>()?;
    m.add_class::<PyDnnNet>()?;
    Ok(())
}
