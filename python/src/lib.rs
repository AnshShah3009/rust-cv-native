use cv_3d::PointCloud;
use cv_calib3d::{
    find_chessboard_corners as rust_find_chessboard_corners, project_points as rust_project_points,
    solve_pnp_ransac as rust_solve_pnp_ransac,
};
use cv_core::CameraExtrinsics;
use cv_core::CameraIntrinsics;
use cv_core::Distortion as RustDistortion;
use cv_core::Rect as RustRect;
use cv_registration::{registration_icp_point_to_plane, GlobalRegistrationResult, ICPResult};
use cv_scientific::geometry::vectorized_iou as rust_vectorized_iou;
use cv_scientific::point_cloud::{
    estimate_normals as sci_estimate_normals, orient_normals as sci_orient_normals,
};
use cv_slam::SlamSystem;
use cv_stereo::{compute_validity_mask, DisparityMap, StereoParams};
use geo::Area;
use nalgebra::{Matrix4, Point3, Vector3};
#[allow(deprecated)]
use numpy::{IntoPyArray, PyArray2, PyArrayMethods, PyUntypedArrayMethods};
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

/// Calculate Intersection over Union (IoU) between two rectangles.
///
/// Args:
///     r1: First rectangle
///     r2: Second rectangle
///
/// Returns:
///     float: IoU value between 0 and 1
///
/// Example:
///     >>> rect1 = PyRect(0, 0, 100, 100)
///     >>> rect2 = PyRect(50, 50, 100, 100)
///     >>> iou = iou(rect1, rect2)  # 0.143
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
    // boxes1 and boxes2 are expected to be (N, 4) arrays of [x, y, w, h]
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

    // Convert ndarray::Array2 to PyArray2
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

    /// Find indices of polygons intersecting the bounding box of the query polygon.
    /// This is a filter step.
    pub fn query(&self, min_x: f64, min_y: f64, max_x: f64, max_y: f64) -> Vec<usize> {
        self.inner.query_bbox(min_x, min_y, max_x, max_y)
    }

    pub fn nearest(&self, x: f64, y: f64) -> Option<usize> {
        self.inner.nearest(x, y)
    }
}

#[pyclass]
pub struct PyPointCloud {
    pub inner: PointCloud,
}

#[pymethods]
impl PyPointCloud {
    #[new]
    pub fn new() -> Self {
        Self {
            inner: PointCloud::new(Vec::new()),
        }
    }

    pub fn num_points(&self) -> usize {
        self.inner.points.len()
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
pub struct PySlam {
    inner: Box<dyn SlamSystem + Send>,
}

#[pymethods]
impl PySlam {
    #[new]
    pub fn new(fx: f64, fy: f64, cx: f64, cy: f64, width: u32, height: u32) -> Self {
        let intrinsics = CameraIntrinsics::new(fx, fy, cx, cy, width, height);
        Self {
            inner: Box::new(cv_slam::Slam::new(intrinsics)),
        }
    }

    pub fn process_frame(&mut self, image: Bound<'_, PyArray2<u8>>) -> PyResult<()> {
        let view = image.readonly();
        let shape = view.shape();
        let mut gray = image::GrayImage::new(shape[1] as u32, shape[0] as u32);

        // Safety: view.as_slice() returns Result<&[u8], ...>
        let data = view
            .as_slice()
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyTypeError, _>(e.to_string()))?;
        gray.copy_from_slice(data);

        self.inner.process_frame(&gray);
        Ok(())
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
    let height = shape[0];
    let width = shape[1];

    let mut gray = image::GrayImage::new(width as u32, height as u32);
    let data = input_view
        .as_slice()
        .map_err(|e: numpy::NotContiguousError| {
            PyErr::new::<pyo3::exceptions::PyTypeError, _>(e.to_string())
        })?;
    gray.copy_from_slice(data);

    let result = cv_imgproc::gaussian_blur(&gray, sigma as f32);

    let py_array = PyArray2::zeros_bound(py, [height, width], false);
    unsafe {
        let res_ptr = py_array
            .as_slice_mut()
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyTypeError, _>(e.to_string()))?;
        res_ptr.copy_from_slice(result.as_raw());
    }
    Ok(py_array)
}

#[pyfunction]
fn detect_orb<'py>(
    py: Python<'py>,
    input: Bound<'py, PyArray2<u8>>,
    n_features: usize,
) -> PyResult<(Vec<PyKeyPoint>, Bound<'py, PyArray2<u8>>)> {
    let input_view = input.readonly();
    let shape = input_view.shape();
    let height = shape[0];
    let width = shape[1];

    let mut gray = image::GrayImage::new(width as u32, height as u32);
    let data = input_view
        .as_slice()
        .map_err(|e: numpy::NotContiguousError| {
            PyErr::new::<pyo3::exceptions::PyTypeError, _>(e.to_string())
        })?;
    gray.copy_from_slice(data);

    let (kps, descs) = cv_features::orb_detect_and_compute(&gray, n_features);

    let py_kps: Vec<PyKeyPoint> = kps
        .keypoints
        .iter()
        .map(|kp| PyKeyPoint {
            x: kp.x as f32,
            y: kp.y as f32,
            size: kp.size as f32,
            angle: kp.angle as f32,
            response: kp.response as f32,
        })
        .collect();

    let n_descs = descs.descriptors.len();
    let desc_size = if n_descs > 0 {
        descs.descriptors[0].data.len()
    } else {
        0
    };
    let py_descs = PyArray2::zeros_bound(py, [n_descs, desc_size], false);

    let res_ptr = unsafe {
        py_descs
            .as_slice_mut()
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyTypeError, _>(e.to_string()))?
    };
    for (i, d) in descs.descriptors.iter().enumerate() {
        res_ptr[i * desc_size..(i + 1) * desc_size].copy_from_slice(&d.data);
    }

    Ok((py_kps, py_descs))
}

#[pyfunction]
fn match_descriptors<'py>(
    _py: Python<'_>,
    query: Bound<'py, PyArray2<u8>>,
    train: Bound<'py, PyArray2<u8>>,
) -> PyResult<Vec<PyFeatureMatch>> {
    let q_view = query.readonly();
    let t_view = train.readonly();

    let q_shape = q_view.shape();
    let t_shape = t_view.shape();

    let q_data = q_view
        .as_slice()
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyTypeError, _>(e.to_string()))?;
    let t_data = t_view
        .as_slice()
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyTypeError, _>(e.to_string()))?;

    let mut q_descs = cv_features::Descriptors::new();
    for i in 0..q_shape[0] {
        let data = q_data[i * q_shape[1]..(i + 1) * q_shape[1]].to_vec();
        q_descs.descriptors.push(cv_features::Descriptor::new(
            data,
            cv_core::KeyPoint::default(),
        ));
    }

    let mut t_descs = cv_features::Descriptors::new();
    for i in 0..t_shape[0] {
        let data = t_data[i * t_shape[1]..(i + 1) * t_shape[1]].to_vec();
        t_descs.descriptors.push(cv_features::Descriptor::new(
            data,
            cv_core::KeyPoint::default(),
        ));
    }

    let matcher = cv_features::Matcher::new(cv_features::MatchType::BruteForce);
    let matches = matcher.match_descriptors(&q_descs, &t_descs);

    Ok(matches
        .matches
        .iter()
        .map(|m| PyFeatureMatch {
            query_idx: m.query_idx,
            train_idx: m.train_idx,
            distance: m.distance,
        })
        .collect())
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
        Self {
            inner: CameraIntrinsics::new(fx, fy, cx, cy, width, height),
        }
    }

    #[getter]
    pub fn fx(&self) -> f64 {
        self.inner.fx
    }
    #[getter]
    pub fn fy(&self) -> f64 {
        self.inner.fy
    }
    #[getter]
    pub fn cx(&self) -> f64 {
        self.inner.cx
    }
    #[getter]
    pub fn cy(&self) -> f64 {
        self.inner.cy
    }
    #[getter]
    pub fn width(&self) -> u32 {
        self.inner.width
    }
    #[getter]
    pub fn height(&self) -> u32 {
        self.inner.height
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
        if rvec.len() != 3 || tvec.len() != 3 {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "rvec and tvec must have 3 elements each",
            ));
        }
        let axis = nalgebra::Vector3::new(rvec[0], rvec[1], rvec[2]);
        let angle = axis.norm();
        let rotation = if angle < 1e-10 {
            nalgebra::Matrix3::identity()
        } else {
            nalgebra::Rotation3::from_axis_angle(&nalgebra::Unit::new_normalize(axis), angle).into()
        };
        let translation = nalgebra::Vector3::new(tvec[0], tvec[1], tvec[2]);
        Ok(Self {
            inner: CameraExtrinsics {
                rotation,
                translation,
            },
        })
    }

    pub fn get_rvec(&self) -> Vec<f64> {
        let r = &self.inner.rotation;
        vec![r[(0, 0)], r[(1, 0)], r[(2, 0)]]
    }

    pub fn get_tvec(&self) -> Vec<f64> {
        vec![
            self.inner.translation[0],
            self.inner.translation[1],
            self.inner.translation[2],
        ]
    }
}

#[pyclass]
#[derive(Clone)]
pub struct PyDistortion {
    pub inner: RustDistortion,
}

#[pymethods]
impl PyDistortion {
    #[new]
    pub fn new(k1: f64, k2: f64, p1: f64, p2: f64, k3: f64) -> Self {
        Self {
            inner: RustDistortion::new(k1, k2, p1, p2, k3),
        }
    }

    #[getter]
    pub fn k1(&self) -> f64 {
        self.inner.k1
    }
    #[getter]
    pub fn k2(&self) -> f64 {
        self.inner.k2
    }
    #[getter]
    pub fn p1(&self) -> f64 {
        self.inner.p1
    }
    #[getter]
    pub fn p2(&self) -> f64 {
        self.inner.p2
    }
    #[getter]
    pub fn k3(&self) -> f64 {
        self.inner.k3
    }
}

#[pyclass]
#[derive(Clone)]
pub struct PyDisparityMap {
    pub inner: DisparityMap,
}

#[pymethods]
impl PyDisparityMap {
    #[new]
    pub fn new(width: u32, height: u32, min_d: i32, max_d: i32) -> Self {
        Self {
            inner: DisparityMap::new(width, height, min_d, max_d),
        }
    }

    pub fn get(&self, x: u32, y: u32) -> f32 {
        self.inner.get(x, y)
    }

    pub fn is_valid(&self, x: u32, y: u32) -> bool {
        self.inner.is_valid(x, y)
    }

    pub fn width(&self) -> u32 {
        self.inner.width
    }

    pub fn height(&self) -> u32 {
        self.inner.height
    }
}

#[pyclass]
#[derive(Clone)]
pub struct PyStereoParams {
    pub inner: StereoParams,
}

#[pymethods]
impl PyStereoParams {
    #[new]
    pub fn new(focal_length: f64, baseline: f64, cx: f64, cy: f64) -> Self {
        Self {
            inner: StereoParams::new(focal_length, baseline, cx, cy),
        }
    }

    pub fn disparity_to_depth(&self, disparity: f64) -> Option<f64> {
        self.inner.disparity_to_depth(disparity)
    }
}

#[pyfunction]
fn find_chessboard_corners(
    image: Bound<'_, PyArray2<u8>>,
    pattern_size: (usize, usize),
) -> PyResult<Vec<(f64, f64)>> {
    let view = image.readonly();
    let shape = view.shape();
    let height = shape[0];
    let width = shape[1];

    let mut gray = image::GrayImage::new(width as u32, height as u32);
    let data = view.as_slice().map_err(|e: numpy::NotContiguousError| {
        PyErr::new::<pyo3::exceptions::PyTypeError, _>(e.to_string())
    })?;
    gray.copy_from_slice(data);

    let corners = rust_find_chessboard_corners(&gray, pattern_size)
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?;

    Ok(corners.into_iter().map(|p| (p.x, p.y)).collect())
}

#[pyfunction]
fn project_points(
    object_points: Vec<(f64, f64, f64)>,
    intrinsics: &PyCameraIntrinsics,
    extrinsics: &PyCameraExtrinsics,
) -> PyResult<Vec<(f64, f64)>> {
    let pts: Vec<nalgebra::Point3<f64>> = object_points
        .iter()
        .map(|p| nalgebra::Point3::new(p.0, p.1, p.2))
        .collect();

    let result = rust_project_points(&pts, &intrinsics.inner, &extrinsics.inner)
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?;

    Ok(result.into_iter().map(|p| (p.x, p.y)).collect())
}

#[pyfunction]
fn solve_pnp_ransac(
    object_points: Vec<(f64, f64, f64)>,
    image_points: Vec<(f64, f64)>,
    intrinsics: &PyCameraIntrinsics,
    threshold: f64,
    max_iterations: usize,
) -> PyResult<(PyCameraExtrinsics, Vec<bool>)> {
    let obj_pts: Vec<nalgebra::Point3<f64>> = object_points
        .iter()
        .map(|p| nalgebra::Point3::new(p.0, p.1, p.2))
        .collect();
    let img_pts: Vec<nalgebra::Point2<f64>> = image_points
        .iter()
        .map(|p| nalgebra::Point2::new(p.0, p.1))
        .collect();

    let (extrinsics, inliers) = rust_solve_pnp_ransac(
        &obj_pts,
        &img_pts,
        &intrinsics.inner,
        threshold,
        max_iterations,
    )
    .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?;

    let py_ext = PyCameraExtrinsics { inner: extrinsics };
    Ok((py_ext, inliers))
}

#[pyfunction]
fn py_stereo_block_match(
    left: Bound<'_, PyArray2<u8>>,
    right: Bound<'_, PyArray2<u8>>,
    block_size: usize,
    max_disparity: i32,
) -> PyResult<PyDisparityMap> {
    let left_view = left.readonly();
    let right_view = right.readonly();

    let left_shape = left_view.shape();
    let right_shape = right_view.shape();

    if left_shape != right_shape {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "Left and right images must have same dimensions",
        ));
    }

    let height = left_shape[0];
    let width = left_shape[1];

    let mut left_gray = image::GrayImage::new(width as u32, height as u32);
    let mut right_gray = image::GrayImage::new(width as u32, height as u32);

    let left_data = left_view
        .as_slice()
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyTypeError, _>(e.to_string()))?;
    let right_data = right_view
        .as_slice()
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyTypeError, _>(e.to_string()))?;

    left_gray.copy_from_slice(left_data);
    right_gray.copy_from_slice(right_data);

    let result = cv_stereo::stereo_block_match(&left_gray, &right_gray, block_size, max_disparity)
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;

    Ok(PyDisparityMap { inner: result })
}

#[pyfunction]
fn compute_disparity_validity(disparity: &PyDisparityMap, threshold: f32) -> Vec<bool> {
    compute_validity_mask(&disparity.inner, threshold)
}

#[pyfunction]
fn get_resource_group(name: &str) -> PyResult<PyResourceGroup> {
    match cv_runtime::scheduler().get_group(name) {
        Ok(Some(group)) => Ok(PyResourceGroup { inner: group }),
        Ok(None) => Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
            "Resource group '{}' not found",
            name
        ))),
        Err(e) => Err(PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(
            e.to_string(),
        )),
    }
}

#[pyfunction]
#[pyo3(signature = (name, num_threads, cores=None, allow_work_stealing=true, allow_dynamic_scaling=true))]
fn create_resource_group(
    name: &str,
    num_threads: usize,
    cores: Option<Vec<usize>>,
    allow_work_stealing: bool,
    allow_dynamic_scaling: bool,
) -> PyResult<PyResourceGroup> {
    let policy = cv_runtime::orchestrator::GroupPolicy {
        allow_work_stealing,
        allow_dynamic_scaling,
    };
    let group = cv_runtime::scheduler()
        .create_group(name, num_threads, cores, policy)
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
    Ok(PyResourceGroup { inner: group })
}

#[pyfunction]
fn estimate_normals<'a>(
    points: &'a Bound<'_, PyArray2<f32>>,
    k: usize,
) -> PyResult<Bound<'a, PyArray2<f32>>> {
    let view = points.readonly();
    let shape = view.shape();
    let n = shape[0];

    let pts: Vec<Point3<f32>> = view
        .as_slice()?
        .chunks(3)
        .map(|c| Point3::new(c[0], c[1], c[2]))
        .collect();

    let mut pc = cv_core::PointCloud::new(pts);
    sci_estimate_normals(&mut pc, k);

    let normals = pc.normals.unwrap();

    let py = points.py();
    let normals_array = PyArray2::zeros_bound(py, [n, 3], false);
    unsafe {
        let mut slice = normals_array.as_slice_mut()?;
        for (i, n) in normals.iter().enumerate() {
            slice[i * 3] = n.x;
            slice[i * 3 + 1] = n.y;
            slice[i * 3 + 2] = n.z;
        }
    }

    Ok(normals_array)
}

#[pyfunction]
fn orient_normals<'a>(
    points: &'a Bound<'_, PyArray2<f32>>,
    normals: &'a Bound<'_, PyArray2<f32>>,
    k: usize,
) -> PyResult<Bound<'a, PyArray2<f32>>> {
    let view = points.readonly();
    let n = view.shape()[0];

    let pts: Vec<Point3<f32>> = view
        .as_slice()?
        .chunks(3)
        .map(|c| Point3::new(c[0], c[1], c[2]))
        .collect();

    let norms_view = normals.readonly();
    let norms: Vec<Vector3<f32>> = norms_view
        .as_slice()?
        .chunks(3)
        .map(|c| Vector3::new(c[0], c[1], c[2]))
        .collect();

    let mut pc = cv_core::PointCloud::new(pts);
    pc.normals = Some(norms);
    sci_orient_normals(&mut pc, k);

    let oriented = pc.normals.unwrap();

    let py = points.py();
    let result = PyArray2::zeros_bound(py, [n, 3], false);
    unsafe {
        let mut slice = result.as_slice_mut()?;
        for (i, n) in oriented.iter().enumerate() {
            slice[i * 3] = n.x;
            slice[i * 3 + 1] = n.y;
            slice[i * 3 + 2] = n.z;
        }
    }

    Ok(result)
}

#[pyfunction]
fn compute_normals<'a>(
    points: &'a Bound<'_, PyArray2<f32>>,
    k: usize,
) -> PyResult<Bound<'a, PyArray2<f32>>> {
    let view = points.readonly();
    let n = view.shape()[0];

    let pts: Vec<Point3<f32>> = view
        .as_slice()?
        .chunks(3)
        .map(|c| Point3::new(c[0], c[1], c[2]))
        .collect();

    let normals = cv_3d::gpu::point_cloud::compute_normals_simple(&pts, k);

    let py = points.py();
    let result = PyArray2::zeros_bound(py, [n, 3], false);
    unsafe {
        let mut slice = result.as_slice_mut()?;
        for (i, n) in normals.iter().enumerate() {
            slice[i * 3] = n.x;
            slice[i * 3 + 1] = n.y;
            slice[i * 3 + 2] = n.z;
        }
    }

    Ok(result)
}

/// Check if a GPU is available on this system.
#[pyfunction]
fn is_gpu_available_fn() -> bool {
    cv_hal::gpu::GpuContext::is_available()
}

/// Get information about the available GPU.
#[pyfunction]
fn gpu_info() -> Option<String> {
    cv_3d::gpu::gpu_info()
}

/// Downsample a point cloud using voxel grid filtering.
///
/// Args:
///     points: Nx3 numpy array of point coordinates
///     voxel_size: Size of voxel grid cells
///
/// Returns:
///     Mx3 numpy array of downsampled point coordinates
#[pyfunction]
fn voxel_downsample_gpu<'a>(
    py: Python<'a>,
    points: &Bound<'_, PyArray2<f32>>,
    voxel_size: f32,
) -> PyResult<Bound<'a, PyArray2<f32>>> {
    let view = points.readonly();
    let n = view.shape()[0];

    let pts: Vec<Point3<f32>> = view
        .as_slice()?
        .chunks(3)
        .map(|c| Point3::new(c[0], c[1], c[2]))
        .collect();

    let downsampled = cv_3d::gpu::point_cloud::voxel_downsample(&pts, voxel_size);

    let result = PyArray2::zeros_bound(py, [downsampled.len(), 3], false);
    unsafe {
        let mut slice = result.as_slice_mut()?;
        for (i, p) in downsampled.iter().enumerate() {
            slice[i * 3] = p.x;
            slice[i * 3 + 1] = p.y;
            slice[i * 3 + 2] = p.z;
        }
    }

    Ok(result)
}

#[pymodule]
fn cv_native(_py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyPointCloud>()?;
    m.add_class::<PyPointCloud3D>()?;
    m.add_class::<PyPoint3D>()?;
    m.add_class::<PyKeyPoint>()?;
    m.add_class::<PyFeatureMatch>()?;
    m.add_class::<PyResourceGroup>()?;
    m.add_class::<PySlam>()?;
    m.add_class::<PyRect>()?;
    m.add_class::<PyPolygon>()?;
    m.add_class::<PySpatialIndex>()?;
    m.add_class::<PyICPResult>()?;
    m.add_class::<PyGlobalRegistrationResult>()?;
    m.add_class::<PyCameraIntrinsics>()?;
    m.add_class::<PyCameraExtrinsics>()?;
    m.add_class::<PyDistortion>()?;
    m.add_class::<PyDisparityMap>()?;
    m.add_class::<PyStereoParams>()?;

    m.add_function(wrap_pyfunction!(gaussian_blur, m)?)?;
    m.add_function(wrap_pyfunction!(detect_orb, m)?)?;
    m.add_function(wrap_pyfunction!(match_descriptors, m)?)?;
    m.add_function(wrap_pyfunction!(get_resource_group, m)?)?;
    m.add_function(wrap_pyfunction!(create_resource_group, m)?)?;
    m.add_function(wrap_pyfunction!(iou, m)?)?;
    m.add_function(wrap_pyfunction!(py_vectorized_iou, m)?)?;
    m.add_function(wrap_pyfunction!(py_polygon_iou, m)?)?;
    m.add_function(wrap_pyfunction!(registration_icp, m)?)?;
    m.add_function(wrap_pyfunction!(find_chessboard_corners, m)?)?;
    m.add_function(wrap_pyfunction!(project_points, m)?)?;
    m.add_function(wrap_pyfunction!(solve_pnp_ransac, m)?)?;
    m.add_function(wrap_pyfunction!(py_stereo_block_match, m)?)?;
    m.add_function(wrap_pyfunction!(compute_disparity_validity, m)?)?;
    m.add_function(wrap_pyfunction!(estimate_normals, m)?)?;
    m.add_function(wrap_pyfunction!(orient_normals, m)?)?;
    m.add_function(wrap_pyfunction!(compute_normals, m)?)?;
    m.add_function(wrap_pyfunction!(is_gpu_available_fn, m)?)?;
    m.add_function(wrap_pyfunction!(gpu_info, m)?)?;
    m.add_function(wrap_pyfunction!(voxel_downsample_gpu, m)?)?;

    Ok(())
}
