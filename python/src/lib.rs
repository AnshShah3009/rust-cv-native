use cv_3d::PointCloud;
use cv_calib3d::{
    find_chessboard_corners as rust_find_chessboard_corners, project_points as rust_project_points,
    solve_pnp_ransac as rust_solve_pnp_ransac,
    pnp::solve_pnp_refine as rust_solve_pnp_refine,
};
use cv_core::{CameraExtrinsics, CameraIntrinsics, Distortion as RustDistortion, Rect as RustRect, Tensor, TensorShape, storage::CpuStorage, CpuTensor};
use cv_registration::{registration_icp_point_to_plane, GlobalRegistrationResult, ICPResult};
use cv_scientific::geometry::vectorized_iou as rust_vectorized_iou;
use cv_scientific::point_cloud::{
    estimate_normals as sci_estimate_normals, orient_normals as sci_orient_normals,
};
use cv_stereo::{compute_validity_mask, DisparityMap, StereoParams};
use cv_features::{Akaze as RustAkaze, AkazeParams as RustAkazeParams, Lbd as RustLbd, LbdParams, LineMatcher as RustLineMatcher};
use cv_videoio::{VideoCapture as RustVideoCapture};
use cv_imgproc::hough::{hough_lines_p as rust_hough_lines_p, LineSegment as RustLineSegment};
use geo::Area;
use nalgebra::{Matrix4, Point3, Vector3};
#[allow(deprecated)]
use numpy::{IntoPyArray, PyArray2, PyArrayMethods, PyUntypedArrayMethods, PyArray3};
use pyo3::prelude::*;

#[pyclass]
#[derive(Clone, Copy)]
pub struct PyLineSegment {
    #[pyo3(get, set)]
    pub x1: f32,
    #[pyo3(get, set)]
    pub y1: f32,
    #[pyo3(get, set)]
    pub x2: f32,
    #[pyo3(get, set)]
    pub y2: f32,
}

#[pymethods]
impl PyLineSegment {
    #[new]
    pub fn new(x1: f32, y1: f32, x2: f32, y2: f32) -> Self {
        Self { x1, y1, x2, y2 }
    }
}

impl From<RustLineSegment> for PyLineSegment {
    fn from(s: RustLineSegment) -> Self {
        Self { x1: s.x1, y1: s.y1, x2: s.x2, y2: s.y2 }
    }
}

impl From<PyLineSegment> for RustLineSegment {
    fn from(s: PyLineSegment) -> Self {
        Self { x1: s.x1, y1: s.y1, x2: s.x2, y2: s.y2 }
    }
}

#[pyfunction]
fn hough_lines_p(
    image: Bound<'_, PyArray2<u8>>,
    rho: f32,
    theta: f32,
    threshold: u32,
    min_line_len: f32,
    max_line_gap: f32,
) -> PyResult<Vec<PyLineSegment>> {
    let view = image.readonly();
    let shape = view.shape();
    let mut gray = image::GrayImage::new(shape[1] as u32, shape[0] as u32);
    gray.copy_from_slice(view.as_slice()?);
    
    let segments = rust_hough_lines_p(&gray, rho, theta, threshold, min_line_len, max_line_gap);
    Ok(segments.into_iter().map(|s| s.into()).collect())
}

#[pyclass]
pub struct PyLineDescriptor {
    pub data: Vec<u8>,
    pub segment: PyLineSegment,
}

#[pymethods]
impl PyLineDescriptor {
    #[getter]
    pub fn segment(&self) -> PyLineSegment { self.segment }
    
    pub fn get_data(&self) -> Vec<u8> { self.data.clone() }
}

#[pyfunction]
fn compute_lbd(
    image: Bound<'_, PyArray2<u8>>,
    segments: Vec<PyLineSegment>,
) -> PyResult<Vec<PyLineDescriptor>> {
    let view = image.readonly();
    let shape = view.shape();
    let mut gray = image::GrayImage::new(shape[1] as u32, shape[0] as u32);
    gray.copy_from_slice(view.as_slice()?);
    
    let rust_segments: Vec<RustLineSegment> = segments.iter().map(|s| (*s).into()).collect();
    let lbd = RustLbd::new(LbdParams::default());
    let descriptors = lbd.compute(&gray, &rust_segments);
    
    Ok(descriptors.into_iter().map(|d| PyLineDescriptor {
        data: d.data,
        segment: d.segment.into(),
    }).collect())
}

#[pyclass]
#[derive(Clone, Copy)]
pub struct PyLineMatch {
    #[pyo3(get, set)]
    pub query_idx: usize,
    #[pyo3(get, set)]
    pub train_idx: usize,
    #[pyo3(get, set)]
    pub distance: f32,
}

#[pyfunction]
fn match_lines(
    query: Vec<PyLineDescriptor>,
    train: Vec<PyLineDescriptor>,
    threshold: f32,
) -> Vec<PyLineMatch> {
    let q_rust: Vec<cv_features::LineDescriptor> = query.into_iter().map(|d| cv_features::LineDescriptor {
        data: d.data,
        segment: d.segment.into(),
    }).collect();
    let t_rust: Vec<cv_features::LineDescriptor> = train.into_iter().map(|d| cv_features::LineDescriptor {
        data: d.data,
        segment: d.segment.into(),
    }).collect();
    
    let matcher = RustLineMatcher::new(threshold);
    let matches = matcher.match_lines(&q_rust, &t_rust);
    
    matches.into_iter().map(|m| PyLineMatch {
        query_idx: m.query_idx,
        train_idx: m.train_idx,
        distance: m.distance,
    }).collect()
}

// ... (Rest of the previous Python bindings classes and functions)

#[pyclass]
#[derive(Clone, Copy)]
pub struct PyRect {
    pub inner: RustRect,
}

#[pymethods]
impl PyRect {
    #[new]
    pub fn new(x: f32, y: f32, w: f32, h: f32) -> Self {
        Self {
            inner: RustRect::new(x, y, w, h),
        }
    }

    #[getter]
    pub fn x(&self) -> f32 {
        self.inner.x
    }
    #[getter]
    pub fn y(&self) -> f32 {
        self.inner.y
    }
    #[getter]
    pub fn w(&self) -> f32 {
        self.inner.w
    }
    #[getter]
    pub fn h(&self) -> f32 {
        self.inner.h
    }

    pub fn area(&self) -> f32 {
        self.inner.area()
    }
    pub fn iou(&self, other: &PyRect) -> f32 {
        self.inner.iou(&other.inner)
    }
}

#[pyfunction]
fn iou(r1: &PyRect, r2: &PyRect) -> f32 {
    r1.inner.iou(&r2.inner)
}

#[pyclass]
pub struct PyComputeDevice {
    pub inner: cv_hal::compute::ComputeDevice<'static>,
    _cpu: Option<Box<cv_hal::cpu::CpuBackend>>,
    _gpu: Option<Box<cv_hal::gpu::GpuContext>>,
}

#[pymethods]
impl PyComputeDevice {
    #[staticmethod]
    pub fn cpu() -> Self {
        let cpu = Box::new(cv_hal::cpu::CpuBackend::new().unwrap());
        let inner = unsafe { cv_hal::compute::ComputeDevice::Cpu(std::mem::transmute(&*cpu)) };
        Self { inner, _cpu: Some(cpu), _gpu: None }
    }

    #[staticmethod]
    pub fn gpu() -> Self {
        let gpu = Box::new(cv_hal::gpu::GpuContext::new().unwrap());
        let inner = unsafe { cv_hal::compute::ComputeDevice::Gpu(std::mem::transmute(&*gpu)) };
        Self { inner, _cpu: None, _gpu: Some(gpu) }
    }
}

#[pyclass]
pub struct PyTensor {
    pub shape: [usize; 3],
    pub data: Vec<u8>,
}

#[pymethods]
impl PyTensor {
    #[new]
    pub fn new(data: Vec<u8>, c: usize, h: usize, w: usize) -> Self {
        Self { data, shape: [c, h, w] }
    }
}

#[pyclass]
#[derive(Clone)]
pub struct PyKeyPoint {
    #[pyo3(get, set)]
    pub x: f32,
    #[pyo3(get, set)]
    pub y: f32,
    #[pyo3(get, set)]
    pub size: f32,
    #[pyo3(get, set)]
    pub angle: f32,
    #[pyo3(get, set)]
    pub response: f32,
}

#[pymodule]
fn cv_native(_py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyRect>()?;
    m.add_class::<PyComputeDevice>()?;
    m.add_class::<PyTensor>()?;
    m.add_class::<PyKeyPoint>()?;
    m.add_class::<PyLineSegment>()?;
    m.add_class::<PyLineDescriptor>()?;
    m.add_class::<PyLineMatch>()?;

    m.add_function(wrap_pyfunction!(iou, m)?)?;
    m.add_function(wrap_pyfunction!(hough_lines_p, m)?)?;
    m.add_function(wrap_pyfunction!(compute_lbd, m)?)?;
    m.add_function(wrap_pyfunction!(match_lines, m)?)?;

    Ok(())
}
