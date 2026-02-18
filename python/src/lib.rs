use cv_3d::PointCloud;
use cv_calib3d::{
    find_chessboard_corners as rust_find_chessboard_corners, project_points as rust_project_points,
    solve_pnp_ransac as rust_solve_pnp_ransac,
    pnp::solve_pnp_refine as rust_solve_pnp_refine,
};
use cv_core::{CameraExtrinsics, CameraIntrinsics, Distortion as RustDistortion, Rect as RustRect, Tensor, TensorShape, storage::CpuStorage};
use cv_registration::{registration_icp_point_to_plane, GlobalRegistrationResult, ICPResult};
use cv_scientific::geometry::vectorized_iou as rust_vectorized_iou;
use cv_scientific::point_cloud::{
    estimate_normals as sci_estimate_normals, orient_normals as sci_orient_normals,
};
use cv_stereo::{compute_validity_mask, DisparityMap, StereoParams};
use cv_features::{Akaze as RustAkaze, AkazeParams as RustAkazeParams};
use cv_videoio::{VideoCapture as RustVideoCapture};
use geo::Area;
use nalgebra::{Matrix4, Point3, Vector3};
#[allow(deprecated)]
use numpy::{IntoPyArray, PyArray2, PyArrayMethods, PyUntypedArrayMethods, PyArray3};
use pyo3::prelude::*;

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

#[pyfunction]
#[pyo3(name = "vectorized_iou")]
fn py_vectorized_iou<'py>(
    py: Python<'py>,
    boxes1: &Bound<'_, PyArray2<f32>>,
    boxes2: &Bound<'_, PyArray2<f32>>,
) -> PyResult<Bound<'py, PyArray2<f32>>> {
    let b1 = boxes1.readonly();
    let b2 = boxes2.readonly();

    if b1.shape()[1] != 4 || b2.shape()[1] != 4 {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "Boxes must be (N, 4) arrays",
        ));
    }

    let b1_slice = b1.as_slice()?;
    let b2_slice = b2.as_slice()?;

    let rects1: Vec<RustRect> = b1_slice
        .chunks(4)
        .map(|c| RustRect::new(c[0], c[1], c[2], c[3]))
        .collect();
    let rects2: Vec<RustRect> = b2_slice
        .chunks(4)
        .map(|c| RustRect::new(c[0], c[1], c[2], c[3]))
        .collect();

    let result = rust_vectorized_iou(&rects1, &rects2);

    #[allow(deprecated)]
    let pyarray = result.into_pyarray(py);
    #[allow(deprecated)]
    let result = pyarray.to_owned().into_bound(py);
    Ok(result)
}

#[pyclass]
#[derive(Clone)]
pub struct PyPolygon {
    pub inner: geo::Polygon<f64>,
}

#[pymethods]
impl PyPolygon {
    #[new]
    pub fn new(points: Vec<(f64, f64)>) -> PyResult<Self> {
        if points.len() < 3 {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "Polygon must have at least 3 points",
            ));
        }
        let line_string = geo::LineString::from(points);
        Ok(Self {
            inner: geo::Polygon::new(line_string, vec![]),
        })
    }

    pub fn area(&self) -> f64 {
        self.inner.unsigned_area()
    }

    pub fn iou(&self, other: &PyPolygon) -> f64 {
        cv_scientific::geometry::polygon_iou(&self.inner, &other.inner)
    }
}

#[pyfunction]
#[pyo3(name = "polygon_iou")]
fn py_polygon_iou(p1: &PyPolygon, p2: &PyPolygon) -> f64 {
    cv_scientific::geometry::polygon_iou(&p1.inner, &p2.inner)
}

#[pyclass]
pub struct PySpatialIndex {
    pub inner: cv_scientific::geometry::SpatialIndex,
}

#[pymethods]
impl PySpatialIndex {
    #[new]
    pub fn new(polygons: Vec<PyPolygon>) -> Self {
        let polys: Vec<geo::Polygon<f64>> = polygons.into_iter().map(|p| p.inner).collect();
        Self {
            inner: cv_scientific::geometry::SpatialIndex::new(polys),
        }
    }

    pub fn query(&self, min_x: f64, min_y: f64, max_x: f64, max_y: f64) -> Vec<usize> {
        self.inner.query_bbox(min_x, min_y, max_x, max_y)
    }

    pub fn nearest(&self, x: f64, y: f64) -> Option<usize> {
        self.inner.nearest(x, y)
    }
}

#[pyclass]
#[derive(Clone)]
pub struct PyPoint3D {
    #[pyo3(get, set)]
    pub x: f32,
    #[pyo3(get, set)]
    pub y: f32,
    #[pyo3(get, set)]
    pub z: f32,
}

#[pymethods]
impl PyPoint3D {
    #[new]
    pub fn new(x: f32, y: f32, z: f32) -> Self {
        Self { x, y, z }
    }
}

impl From<PyPoint3D> for Point3<f32> {
    fn from(p: PyPoint3D) -> Self {
        Point3::new(p.x, p.y, p.z)
    }
}

#[pyclass]
pub struct PyPointCloud3D {
    pub inner: PointCloud<f32>,
}

#[pymethods]
impl PyPointCloud3D {
    #[new]
    pub fn new(points: Vec<PyPoint3D>) -> Self {
        let pts: Vec<Point3<f32>> = points.into_iter().map(|p| p.into()).collect();
        Self {
            inner: PointCloud::new(pts),
        }
    }

    #[staticmethod]
    pub fn from_arrays(xs: Vec<f32>, ys: Vec<f32>, zs: Vec<f32>) -> PyResult<Self> {
        if xs.len() != ys.len() || xs.len() != zs.len() {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "Arrays must have same length",
            ));
        }
        let points: Vec<Point3<f32>> = xs
            .iter()
            .enumerate()
            .map(|(i, &x)| Point3::new(x, ys[i], zs[i]))
            .collect();
        Ok(Self {
            inner: PointCloud::new(points),
        })
    }

    pub fn with_colors_rgb(&mut self, r: Vec<f32>, g: Vec<f32>, b: Vec<f32>) -> PyResult<()> {
        if r.len() != g.len() || r.len() != b.len() || r.len() != self.inner.points.len() {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "Color arrays must match point count",
            ));
        }
        let colors: Vec<Point3<f32>> = r
            .iter()
            .enumerate()
            .map(|(i, &r)| Point3::new(r, g[i], b[i]))
            .collect();
        self.inner.colors = Some(colors);
        Ok(())
    }

    pub fn with_normals(&mut self, nx: Vec<f32>, ny: Vec<f32>, nz: Vec<f32>) -> PyResult<()> {
        if nx.len() != ny.len() || nx.len() != nz.len() || nx.len() != self.inner.points.len() {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "Normal arrays must match point count",
            ));
        }
        let normals: Vec<Vector3<f32>> = nx
            .iter()
            .enumerate()
            .map(|(i, &nx)| Vector3::new(nx, ny[i], nz[i]))
            .collect();
        self.inner.normals = Some(normals);
        Ok(())
    }

    pub fn num_points(&self) -> usize {
        self.inner.len()
    }
}

#[pyclass]
#[derive(Clone)]
pub struct PyICPResult {
    #[pyo3(get, set)]
    pub fitness: f32,
    #[pyo3(get, set)]
    pub inlier_rmse: f32,
    #[pyo3(get, set)]
    pub num_iterations: u32,
    pub transformation: Vec<f32>,
}

#[pymethods]
impl PyICPResult {
    #[new]
    pub fn new() -> Self {
        Self {
            fitness: 0.0,
            inlier_rmse: 0.0,
            num_iterations: 0,
            transformation: vec![
                1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0,
            ],
        }
    }

    pub fn get_transform(&self) -> Vec<f32> {
        self.transformation.clone()
    }
}

impl From<ICPResult> for PyICPResult {
    fn from(r: ICPResult) -> Self {
        let mut t = vec![0.0f32; 16];
        for i in 0..4 {
            for j in 0..4 {
                t[i * 4 + j] = r.transformation[(i, j)];
            }
        }
        Self {
            fitness: r.fitness,
            inlier_rmse: r.inlier_rmse,
            num_iterations: r.num_iterations as u32,
            transformation: t,
        }
    }
}

#[pyclass]
#[derive(Clone)]
pub struct PyGlobalRegistrationResult {
    #[pyo3(get, set)]
    pub fitness: f32,
    #[pyo3(get, set)]
    pub inlier_rmse: f32,
    pub transformation: Vec<f32>,
}

#[pymethods]
impl PyGlobalRegistrationResult {
    #[new]
    pub fn new() -> Self {
        Self {
            fitness: 0.0,
            inlier_rmse: 0.0,
            transformation: vec![
                1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0,
            ],
        }
    }

    pub fn get_transform(&self) -> Vec<f32> {
        self.transformation.clone()
    }
}

impl From<GlobalRegistrationResult> for PyGlobalRegistrationResult {
    fn from(r: GlobalRegistrationResult) -> Self {
        let mut t = vec![0.0f32; 16];
        for i in 0..4 {
            for j in 0..4 {
                t[i * 4 + j] = r.transformation[(i, j)];
            }
        }
        Self {
            fitness: r.fitness,
            inlier_rmse: r.inlier_rmse,
            transformation: t,
        }
    }
}

#[pyfunction]
fn registration_icp(
    source: &PyPointCloud3D,
    target: &PyPointCloud3D,
    max_distance: f32,
    max_iterations: u32,
) -> PyResult<Option<PyICPResult>> {
    let transform = Matrix4::identity();
    let result = registration_icp_point_to_plane(
        &source.inner,
        &target.inner,
        max_distance,
        &transform,
        max_iterations as usize,
    );
    Ok(result.map(|r| r.into()))
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

#[pymethods]
impl PyKeyPoint {
    #[new]
    pub fn new(x: f32, y: f32, size: f32, angle: f32, response: f32) -> Self {
        Self {
            x,
            y,
            size,
            angle,
            response,
        }
    }
}

#[pyclass]
#[derive(Clone)]
pub struct PyFeatureMatch {
    #[pyo3(get, set)]
    pub query_idx: i32,
    #[pyo3(get, set)]
    pub train_idx: i32,
    #[pyo3(get, set)]
    pub distance: f32,
}

#[pyclass]
pub struct PyResourceGroup {
    pub inner: std::sync::Arc<cv_runtime::ResourceGroup>,
}

#[pymethods]
impl PyResourceGroup {
    pub fn name(&self) -> String {
        self.inner.name.clone()
    }

    pub fn install(&self, py: Python<'_>, f: Py<PyAny>) -> PyResult<PyObject> {
        let group = self.inner.clone();
        py.allow_threads(move || group.install(move || Python::with_gil(|py| f.call0(py))))
    }
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
pub struct PySlam {
    inner: Option<cv_slam::Slam<'static>>,
    _device: Option<PyComputeDevice>,
}

#[pymethods]
impl PySlam {
    #[new]
    pub fn new(device: &PyComputeDevice, fx: f64, fy: f64, cx: f64, cy: f64, width: u32, height: u32) -> Self {
        let intrinsics = CameraIntrinsics::new(fx, fy, cx, cy, width, height);
        let slam = unsafe {
            let d_ptr = &device.inner as *const _;
            cv_slam::Slam::new(&*d_ptr, intrinsics)
        };
        Self {
            inner: Some(slam),
            _device: Some(PyComputeDevice {
                inner: unsafe { std::mem::transmute_copy(&device.inner) },
                _cpu: None,
                _gpu: None,
            }),
        }
    }

    pub fn process_frame(&mut self, image: Bound<'_, PyArray2<u8>>) -> PyResult<()> {
        let view = image.readonly();
        let shape = view.shape();
        let mut gray = image::GrayImage::new(shape[1] as u32, shape[0] as u32);

        let data = view
            .as_slice()
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyTypeError, _>(e.to_string()))?;
        gray.copy_from_slice(data);

        if let Some(ref mut slam) = self.inner {
            slam.process_image(&gray).map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e))?;
        }
        Ok(())
    }
}

#[pyclass]
pub struct PyAkaze {
    inner: RustAkaze,
}

#[pymethods]
impl PyAkaze {
    #[new]
    #[pyo3(signature = (n_octaves=4, n_sublevels=4, threshold=0.001))]
    pub fn new(n_octaves: usize, n_sublevels: usize, threshold: f32) -> Self {
        let params = RustAkazeParams {
            n_octaves,
            n_sublevels,
            threshold,
            ..Default::default()
        };
        Self { inner: RustAkaze::new(params) }
    }

    pub fn detect_and_compute(&self, device: &PyComputeDevice, image: &PyTensor) -> (Vec<PyKeyPoint>, Bound<'_, PyArray2<u8>>) {
        let shape = TensorShape::new(image.shape[0], image.shape[1], image.shape[2]);
        let tensor: Tensor<u8, CpuStorage<u8>> = Tensor::from_vec(image.data.clone(), shape);
        
        let (kps, descs) = self.inner.detect_and_compute_ctx(&device.inner, &tensor);
        
        let py_kps: Vec<PyKeyPoint> = kps.keypoints.iter().map(|kp| PyKeyPoint {
            x: kp.x as f32,
            y: kp.y as f32,
            size: kp.size as f32,
            angle: kp.angle as f32,
            response: kp.response as f32,
        }).collect();

        let n_descs = descs.descriptors.len();
        let desc_size = if n_descs > 0 { descs.descriptors[0].data.len() } else { 0 };
        
        Python::with_gil(|py| {
            let py_descs = PyArray2::zeros_bound(py, [n_descs, desc_size], false);
            unsafe {
                let res_ptr = py_descs.as_slice_mut().unwrap();
                for (i, d) in descs.descriptors.iter().enumerate() {
                    res_ptr[i * desc_size .. (i + 1) * desc_size].copy_from_slice(&d.data);
                }
            }
            (py_kps, py_descs)
        })
    }
}

#[pyclass]
pub struct PyVideoCapture {
    inner: RustVideoCapture,
}

#[pymethods]
impl PyVideoCapture {
    #[new]
    pub fn new(path: &str) -> PyResult<Self> {
        let inner = RustVideoCapture::from_file(path)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyIOError, _>(e.to_string()))?;
        Ok(Self { inner })
    }

    pub fn read(&mut self, py: Python<'_>) -> PyResult<Option<Bound<'_, PyArray3<u8>>>> {
        match self.inner.read() {
            Ok(Some(frame)) => {
                let (h, w, c) = (frame.height, frame.width, frame.channels);
                let py_array = PyArray3::zeros_bound(py, [h, w, c], false);
                unsafe {
                    let res_ptr = py_array.as_slice_mut().unwrap();
                    res_ptr.copy_from_slice(&frame.data);
                }
                Ok(Some(py_array))
            },
            Ok(None) => Ok(None),
            Err(e) => Err(PyErr::new::<pyo3::exceptions::PyIOError, _>(e.to_string())),
        }
    }
}

#[pyfunction]
fn gaussian_blur<'py>(
    py: Python<'py>,
    input: Bound<'py, PyArray2<u8>>,
    sigma: f64,
) -> PyResult<Bound<'py, PyArray2<u8>>> {
    let input_view = input.readonly();
    let shape = input_view.shape();
    let mut gray = image::GrayImage::new(shape[1] as u32, shape[0] as u32);
    let data = input_view.as_slice()?;
    gray.copy_from_slice(data);
    let result = cv_imgproc::gaussian_blur(&gray, sigma as f32);
    let py_array = PyArray2::zeros_bound(py, [shape[0], shape[1]], false);
    unsafe {
        let res_ptr = py_array.as_slice_mut()?;
        res_ptr.copy_from_slice(result.as_raw());
    }
    Ok(py_array)
}

#[pyclass]
#[derive(Clone)]
pub struct PyCameraIntrinsics {
    pub inner: CameraIntrinsics,
}

#[pymethods]
impl PyCameraIntrinsics {
    #[new]
    pub fn new(fx: f64, fy: f64, cx: f64, cy: f64, width: u32, height: u32) -> Self {
        Self { inner: CameraIntrinsics::new(fx, fy, cx, cy, width, height) }
    }
}

#[pyclass]
#[derive(Clone)]
pub struct PyCameraExtrinsics {
    pub inner: CameraExtrinsics,
}

#[pymethods]
impl PyCameraExtrinsics {
    #[new]
    pub fn new(rvec: Vec<f64>, tvec: Vec<f64>) -> PyResult<Self> {
        let axis = nalgebra::Vector3::new(rvec[0], rvec[1], rvec[2]);
        let angle = axis.norm();
        let rotation = if angle < 1e-10 { nalgebra::Matrix3::identity() } else {
            nalgebra::Rotation3::from_axis_angle(&nalgebra::Unit::new_normalize(axis), angle).into()
        };
        let translation = nalgebra::Vector3::new(tvec[0], tvec[1], tvec[2]);
        Ok(Self { inner: CameraExtrinsics { rotation, translation } })
    }
}

#[pyfunction]
fn solve_pnp_refine(
    object_points: Vec<(f64, f64, f64)>,
    image_points: Vec<(f64, f64)>,
    intrinsics: &PyCameraIntrinsics,
    init_extrinsics: &PyCameraExtrinsics,
) -> PyResult<PyCameraExtrinsics> {
    let obj_pts: Vec<Point3<f64>> = object_points.iter().map(|p| Point3::new(p.0, p.1, p.2)).collect();
    let img_pts: Vec<nalgebra::Point2<f64>> = image_points.iter().map(|p| nalgebra::Point2::new(p.0, p.1)).collect();
    let refined = rust_solve_pnp_refine(&obj_pts, &img_pts, &intrinsics.inner, &init_extrinsics.inner)
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?;
    Ok(PyCameraExtrinsics { inner: refined })
}

#[pymodule]
fn cv_native(_py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyRect>()?;
    m.add_class::<PyPolygon>()?;
    m.add_class::<PySpatialIndex>()?;
    m.add_class::<PyPoint3D>()?;
    m.add_class::<PyPointCloud3D>()?;
    m.add_class::<PyICPResult>()?;
    m.add_class::<PyGlobalRegistrationResult>()?;
    m.add_class::<PyKeyPoint>()?;
    m.add_class::<PyFeatureMatch>()?;
    m.add_class::<PyResourceGroup>()?;
    m.add_class::<PyComputeDevice>()?;
    m.add_class::<PyTensor>()?;
    m.add_class::<PySlam>()?;
    m.add_class::<PyAkaze>()?;
    m.add_class::<PyVideoCapture>()?;
    m.add_class::<PyCameraIntrinsics>()?;
    m.add_class::<PyCameraExtrinsics>()?;

    m.add_function(wrap_pyfunction!(iou, m)?)?;
    m.add_function(wrap_pyfunction!(py_vectorized_iou, m)?)?;
    m.add_function(wrap_pyfunction!(py_polygon_iou, m)?)?;
    m.add_function(wrap_pyfunction!(registration_icp, m)?)?;
    m.add_function(wrap_pyfunction!(gaussian_blur, m)?)?;
    m.add_function(wrap_pyfunction!(solve_pnp_refine, m)?)?;
    m.add_function(wrap_pyfunction!(is_gpu_available_fn, m)?)?;
    m.add_function(wrap_pyfunction!(compute_normals, m)?)?;

    Ok(())
}
