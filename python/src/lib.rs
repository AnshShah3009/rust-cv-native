use cv_core::Rect as RustRect;
use cv_scientific::geometry::vectorized_iou as rust_vectorized_iou;
#[allow(deprecated)]
use numpy::{PyArray2, PyUntypedArrayMethods, PyArrayMethods, IntoPyArray};
use pyo3::prelude::*;
use cv_core::CameraIntrinsics;
use cv_3d::PointCloud;
use cv_slam::SlamSystem;
use geo::Area;

#[pyclass]
#[derive(Clone, Copy)]
pub struct PyRect {
    pub inner: RustRect,
}

#[pymethods]
impl PyRect {
    #[new]
    pub fn new(x: f32, y: f32, w: f32, h: f32) -> Self {
        Self { inner: RustRect::new(x, y, w, h) }
    }

    #[getter]
    pub fn x(&self) -> f32 { self.inner.x }
    #[getter]
    pub fn y(&self) -> f32 { self.inner.y }
    #[getter]
    pub fn w(&self) -> f32 { self.inner.w }
    #[getter]
    pub fn h(&self) -> f32 { self.inner.h }

    pub fn area(&self) -> f32 { self.inner.area() }
    pub fn iou(&self, other: &PyRect) -> f32 { self.inner.iou(&other.inner) }
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
    boxes2: &Bound<'_, PyArray2<f32>>
) -> PyResult<Bound<'py, PyArray2<f32>>> {
    // boxes1 and boxes2 are expected to be (N, 4) arrays of [x, y, w, h]
    let b1 = boxes1.readonly();
    let b2 = boxes2.readonly();
    
    if b1.shape()[1] != 4 || b2.shape()[1] != 4 {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>("Boxes must be (N, 4) arrays"));
    }

    let b1_slice = b1.as_slice()?;
    let b2_slice = b2.as_slice()?;

    let rects1: Vec<RustRect> = b1_slice.chunks(4).map(|c| RustRect::new(c[0], c[1], c[2], c[3])).collect();
    let rects2: Vec<RustRect> = b2_slice.chunks(4).map(|c| RustRect::new(c[0], c[1], c[2], c[3])).collect();

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
             return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>("Polygon must have at least 3 points"));
        }
        let line_string = geo::LineString::from(points);
        Ok(Self { inner: geo::Polygon::new(line_string, vec![]) })
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
    pub fn new(num_points: usize) -> Self {
        Self {
            inner: PointCloud::new(num_points),
        }
    }

    pub fn num_points(&self) -> usize {
        self.inner.num_points
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

#[pymethods]
impl PyKeyPoint {
    #[new]
    pub fn new(x: f32, y: f32, size: f32, angle: f32, response: f32) -> Self {
        Self { x, y, size, angle, response }
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
        py.allow_threads(move || {
            group.install(move || {
                Python::with_gil(|py| {
                    f.call0(py)
                })
            })
        })
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
        let data = view.as_slice().map_err(|e| PyErr::new::<pyo3::exceptions::PyTypeError, _>(e.to_string()))?;
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
    let data = input_view.as_slice().map_err(|e: numpy::NotContiguousError| PyErr::new::<pyo3::exceptions::PyTypeError, _>(e.to_string()))?;
    gray.copy_from_slice(data);

    let result = cv_imgproc::gaussian_blur(&gray, sigma as f32);

    let py_array = PyArray2::zeros_bound(py, [height, width], false);
    unsafe {
        let res_ptr = py_array.as_slice_mut().map_err(|e| PyErr::new::<pyo3::exceptions::PyTypeError, _>(e.to_string()))?;
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
    let data = input_view.as_slice().map_err(|e: numpy::NotContiguousError| PyErr::new::<pyo3::exceptions::PyTypeError, _>(e.to_string()))?;
    gray.copy_from_slice(data);

    let (kps, descs) = cv_features::orb_detect_and_compute(&gray, n_features);

    let py_kps: Vec<PyKeyPoint> = kps.keypoints.iter().map(|kp| PyKeyPoint {
        x: kp.x as f32,
        y: kp.y as f32,
        size: kp.size as f32,
        angle: kp.angle as f32,
        response: kp.response as f32,
    }).collect();

    let n_descs = descs.descriptors.len();
    let desc_size = if n_descs > 0 { descs.descriptors[0].data.len() } else { 0 };
    let py_descs = PyArray2::zeros_bound(py, [n_descs, desc_size], false);
    
    let res_ptr = unsafe {
        py_descs.as_slice_mut().map_err(|e| PyErr::new::<pyo3::exceptions::PyTypeError, _>(e.to_string()))?
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

    let q_data = q_view.as_slice().map_err(|e| PyErr::new::<pyo3::exceptions::PyTypeError, _>(e.to_string()))?;
    let t_data = t_view.as_slice().map_err(|e| PyErr::new::<pyo3::exceptions::PyTypeError, _>(e.to_string()))?;

    let mut q_descs = cv_features::Descriptors::new();
    for i in 0..q_shape[0] {
        let data = q_data[i * q_shape[1]..(i + 1) * q_shape[1]].to_vec();
        q_descs.descriptors.push(cv_features::Descriptor::new(data, cv_core::KeyPoint::default()));
    }

    let mut t_descs = cv_features::Descriptors::new();
    for i in 0..t_shape[0] {
        let data = t_data[i * t_shape[1]..(i + 1) * t_shape[1]].to_vec();
        t_descs.descriptors.push(cv_features::Descriptor::new(data, cv_core::KeyPoint::default()));
    }

    let matcher = cv_features::Matcher::new(cv_features::MatchType::BruteForce);
    let matches = matcher.match_descriptors(&q_descs, &t_descs);

    Ok(matches.matches.iter().map(|m| PyFeatureMatch {
        query_idx: m.query_idx,
        train_idx: m.train_idx,
        distance: m.distance,
    }).collect())
}

#[pyfunction]
fn get_resource_group(name: &str) -> PyResult<PyResourceGroup> {
    if let Some(group) = cv_runtime::scheduler().get_group(name) {
        Ok(PyResourceGroup { inner: group })
    } else {
        Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("Resource group '{}' not found", name)))
    }
}

#[pyfunction]
fn create_resource_group(name: &str, num_threads: usize, cores: Option<Vec<usize>>) -> PyResult<PyResourceGroup> {
    let group = cv_runtime::scheduler().create_group(name, num_threads, cores)
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
    Ok(PyResourceGroup { inner: group })
}

#[pymodule]
fn cv_native(_py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyPointCloud>()?;
    m.add_class::<PyKeyPoint>()?;
    m.add_class::<PyFeatureMatch>()?;
    m.add_class::<PyResourceGroup>()?;
    m.add_class::<PySlam>()?;
    m.add_class::<PyRect>()?;
    m.add_class::<PyPolygon>()?;
    m.add_class::<PySpatialIndex>()?;
    
    m.add_function(wrap_pyfunction!(gaussian_blur, m)?)?;
    m.add_function(wrap_pyfunction!(detect_orb, m)?)?;
    m.add_function(wrap_pyfunction!(match_descriptors, m)?)?;
    m.add_function(wrap_pyfunction!(get_resource_group, m)?)?;
    m.add_function(wrap_pyfunction!(create_resource_group, m)?)?;
    m.add_function(wrap_pyfunction!(iou, m)?)?;
    m.add_function(wrap_pyfunction!(py_vectorized_iou, m)?)?;
    m.add_function(wrap_pyfunction!(py_polygon_iou, m)?)?;
    
    Ok(())
}
