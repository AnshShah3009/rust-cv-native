use cv_3d::mesh::{reconstruction, TriangleMesh};
use cv_core::point_cloud::PointCloud;
use cv_core::{storage::CpuStorage, CameraIntrinsics, KeyPoint, KeyPoints, Tensor, TensorShape};
use cv_dnn::{DnnError, DnnNet};
use cv_features::{akaze, brief, fast, gftt, harris, hog, sift};
use cv_optimize::isam2::Isam2;
use cv_runtime::orchestrator::{scheduler, RuntimeRunner, WorkloadHint};
use cv_slam::Slam as RustSlam;
use cv_videoio::{open_video, VideoCapture};
use nalgebra::{Point2, Point3, Vector3};
use numpy::{PyArray3, PyReadonlyArray2, PyUntypedArrayMethods};
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

// ============== Video Bindings ==============

#[pyclass]
pub struct PyVideoCapture {
    inner: Box<dyn VideoCapture>,
}

#[pymethods]
impl PyVideoCapture {
    #[new]
    pub fn new(path: &str) -> PyResult<Self> {
        let cap = open_video(path)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
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

// ============== DNN Bindings ==============

#[pyclass]
pub struct PyDnnNet {
    inner: DnnNet,
    runner: Arc<cv_runtime::ResourceGroup>,
}

#[pymethods]
impl PyDnnNet {
    #[new]
    pub fn new(path: &str, hint: PyWorkloadHint) -> PyResult<Self> {
        let inner = DnnNet::load(path)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
        let s = scheduler()
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
        let runner = s
            .best_gpu_or_cpu_for(hint.into())
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;

        Ok(Self { inner, runner })
    }

    pub fn forward(&self, image: PyReadonlyArray2<u8>) -> PyResult<Vec<Vec<f32>>> {
        let shape = image.shape();
        let gray = image::GrayImage::from_raw(
            shape[1] as u32,
            shape[0] as u32,
            image.as_slice()?.to_vec(),
        )
        .ok_or_else(|| {
            PyErr::new::<pyo3::exceptions::PyValueError, _>("Invalid image dimensions")
        })?;

        let dyn_img = image::DynamicImage::ImageLuma8(gray);

        let input_tensor = self
            .inner
            .preprocess(&dyn_img, &self.runner)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;

        let outputs = self
            .inner
            .forward(&input_tensor)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;

        let mut py_outputs = Vec::new();
        for tensor in outputs {
            if let Ok(slice) = tensor.as_slice() {
                py_outputs.push(slice.to_vec());
            }
        }

        Ok(py_outputs)
    }
}

// ============== SLAM Bindings ==============

#[pyclass]
pub struct PySlam {
    inner: RustSlam,
    _group: Arc<cv_runtime::ResourceGroup>,
}

#[pymethods]
impl PySlam {
    #[new]
    pub fn new(hint: PyWorkloadHint) -> PyResult<Self> {
        let s = scheduler()
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
        let group = s
            .best_gpu_or_cpu_for(hint.into())
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;

        let intrinsics = CameraIntrinsics::new_ideal(640, 480);
        Ok(Self {
            inner: RustSlam::new(group.clone(), intrinsics),
            _group: group,
        })
    }

    pub fn process_frame(
        &mut self,
        image: PyReadonlyArray2<u8>,
    ) -> PyResult<(Vec<f32>, Vec<usize>)> {
        let shape = image.shape();
        let rows = shape[0];
        let cols = shape[1];

        let mut gray = image::GrayImage::new(cols as u32, rows as u32);
        gray.copy_from_slice(image.as_slice()?);

        let (pose, tracked) = self
            .inner
            .process_image(&gray)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e))?;

        let mat = pose.matrix();
        let flat_mat: Vec<f32> = mat.iter().map(|&v| v as f32).collect();

        Ok((flat_mat, tracked))
    }
}

// ============== Feature Detection Bindings ==============

#[pyclass]
pub struct PyKeyPoints {
    keypoints: Vec<KeyPoint>,
}

#[pymethods]
impl PyKeyPoints {
    #[new]
    pub fn new() -> Self {
        Self {
            keypoints: Vec::new(),
        }
    }

    pub fn __len__(&self) -> usize {
        self.keypoints.len()
    }

    pub fn to_list(&self) -> Vec<(f64, f64)> {
        self.keypoints.iter().map(|kp| (kp.x, kp.y)).collect()
    }
}

#[pyclass]
pub struct PyFeatureDetector;

#[pymethods]
impl PyFeatureDetector {
    #[staticmethod]
    pub fn fast_detect(image: PyReadonlyArray2<u8>, threshold: i32) -> PyResult<PyKeyPoints> {
        let shape = image.shape();
        let gray = image::GrayImage::from_raw(
            shape[1] as u32,
            shape[0] as u32,
            image.as_slice()?.to_vec(),
        )
        .ok_or_else(|| PyErr::new::<pyo3::exceptions::PyValueError, _>("Invalid image"))?;

        let kps = fast::fast_detect(&gray, threshold);
        Ok(PyKeyPoints {
            keypoints: kps.keypoints,
        })
    }

    #[staticmethod]
    pub fn harris_detect(
        image: PyReadonlyArray2<u8>,
        k: f64,
        threshold: f64,
    ) -> PyResult<PyKeyPoints> {
        let shape = image.shape();
        let gray = image::GrayImage::from_raw(
            shape[1] as u32,
            shape[0] as u32,
            image.as_slice()?.to_vec(),
        )
        .ok_or_else(|| PyErr::new::<pyo3::exceptions::PyValueError, _>("Invalid image"))?;

        let kps = harris::harris_detect(&gray, 3, 3, k, threshold);
        Ok(PyKeyPoints {
            keypoints: kps.keypoints,
        })
    }

    #[staticmethod]
    pub fn gftt_detect(
        image: PyReadonlyArray2<u8>,
        max_corners: usize,
        quality_level: f64,
        min_distance: f64,
    ) -> PyResult<PyKeyPoints> {
        let shape = image.shape();
        let gray = image::GrayImage::from_raw(
            shape[1] as u32,
            shape[0] as u32,
            image.as_slice()?.to_vec(),
        )
        .ok_or_else(|| PyErr::new::<pyo3::exceptions::PyValueError, _>("Invalid image"))?;

        let kps = gftt::gftt_detect(&gray, max_corners, quality_level, min_distance);
        Ok(PyKeyPoints {
            keypoints: kps.keypoints,
        })
    }

    #[staticmethod]
    pub fn shi_tomasi_detect(
        image: PyReadonlyArray2<u8>,
        max_corners: usize,
        quality_level: f64,
    ) -> PyResult<PyKeyPoints> {
        let shape = image.shape();
        let gray = image::GrayImage::from_raw(
            shape[1] as u32,
            shape[0] as u32,
            image.as_slice()?.to_vec(),
        )
        .ok_or_else(|| PyErr::new::<pyo3::exceptions::PyValueError, _>("Invalid image"))?;

        let kps = harris::shi_tomasi_detect(&gray, max_corners, quality_level, 1.0);
        Ok(PyKeyPoints {
            keypoints: kps.keypoints,
        })
    }
}

// ============== ISAM2 Bindings ==============

#[pyclass]
pub struct PyIsam2 {
    inner: Isam2,
}

#[pymethods]
impl PyIsam2 {
    #[new]
    pub fn new(optimize_on_update: bool, batch_optimize: bool) -> Self {
        Self {
            inner: Isam2::with_config(optimize_on_update, batch_optimize),
        }
    }

    pub fn add_pose(&self, id: usize, x: f64, y: f64, z: f64) {
        self.inner.add_pose(id, Vector3::new(x, y, z));
    }

    pub fn add_point(&self, id: usize, x: f64, y: f64, z: f64) {
        self.inner.add_point(id, Point3::new(x, y, z));
    }

    pub fn add_factor(&self, from: usize, to: usize, tx: f64, ty: f64, tz: f64, noise: f64) {
        use nalgebra::DVector;
        let measurement = DVector::from_vec(vec![tx, ty, tz]);
        self.inner.add_factor(from, to, measurement, noise);
    }

    pub fn update(&self) -> PyResult<()> {
        self.inner
            .update()
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e))
    }

    pub fn optimize(&self) -> PyResult<()> {
        self.inner
            .optimize()
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e))
    }

    pub fn get_pose(&self, id: usize) -> Option<(f64, f64, f64)> {
        self.inner.get_pose(id).map(|p| (p.x, p.y, p.z))
    }

    pub fn get_point(&self, id: usize) -> Option<(f64, f64, f64)> {
        self.inner.get_point(id).map(|p| (p.x, p.y, p.z))
    }

    pub fn get_all_poses(&self) -> Vec<(usize, f64, f64, f64)> {
        self.inner
            .get_all_poses()
            .into_iter()
            .map(|(id, p)| (id, p.x, p.y, p.z))
            .collect()
    }

    pub fn num_nodes(&self) -> usize {
        self.inner.num_nodes()
    }

    pub fn num_factors(&self) -> usize {
        self.inner.num_factors()
    }
}

// ============== Point Cloud & Mesh Bindings ==============

#[pyclass]
pub struct PyPointCloud {
    points: Vec<Point3<f32>>,
    normals: Option<Vec<Vector3<f32>>>,
}

#[pymethods]
impl PyPointCloud {
    #[new]
    pub fn new(points: Vec<(f32, f32, f32)>) -> Self {
        Self {
            points: points
                .iter()
                .map(|(x, y, z)| Point3::new(*x, *y, *z))
                .collect(),
            normals: None,
        }
    }

    pub fn from_numpy(points: PyReadonlyArray2<f32>) -> PyResult<Self> {
        let shape = points.shape();
        if shape.len() != 2 || shape[1] != 3 {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "Expected Nx3 array",
            ));
        }
        let data = points.as_slice()?;
        let points: Vec<Point3<f32>> = data
            .chunks(3)
            .map(|c| Point3::new(c[0], c[1], c[2]))
            .collect();
        Ok(Self {
            points,
            normals: None,
        })
    }

    pub fn to_numpy(&self) -> Vec<f32> {
        let mut result = Vec::with_capacity(self.points.len() * 3);
        for p in &self.points {
            result.push(p.x);
            result.push(p.y);
            result.push(p.z);
        }
        result
    }

    pub fn num_points(&self) -> usize {
        self.points.len()
    }

    pub fn set_normals(&mut self, normals: PyReadonlyArray2<f32>) -> PyResult<()> {
        let shape = normals.shape();
        if shape.len() != 2 || shape[1] != 3 {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "Expected Nx3 array",
            ));
        }
        let data = normals.as_slice()?;
        self.normals = Some(
            data.chunks(3)
                .map(|c| Vector3::new(c[0], c[1], c[2]))
                .collect(),
        );
        Ok(())
    }
}

#[pyclass]
pub struct PyTriangleMesh {
    vertices: Vec<Point3<f32>>,
    faces: Vec<[usize; 3]>,
}

#[pymethods]
impl PyTriangleMesh {
    #[new]
    pub fn new() -> Self {
        Self {
            vertices: Vec::new(),
            faces: Vec::new(),
        }
    }

    pub fn add_vertex(&mut self, x: f32, y: f32, z: f32) -> usize {
        let idx = self.vertices.len();
        self.vertices.push(Point3::new(x, y, z));
        idx
    }

    pub fn add_face(&mut self, v0: usize, v1: usize, v2: usize) {
        self.faces.push([v0, v1, v2]);
    }

    pub fn num_vertices(&self) -> usize {
        self.vertices.len()
    }

    pub fn num_faces(&self) -> usize {
        self.faces.len()
    }

    pub fn to_obj(&self) -> String {
        let mut output = String::new();
        for v in &self.vertices {
            output.push_str(&format!("v {} {} {}\n", v.x, v.y, v.z));
        }
        for f in &self.faces {
            output.push_str(&format!("f {} {} {}\n", f[0] + 1, f[1] + 1, f[2] + 1));
        }
        output
    }
}

#[pyclass]
pub struct PyMeshReconstruction;

#[pymethods]
impl PyMeshReconstruction {
    #[staticmethod]
    pub fn create_sphere(center: (f32, f32, f32), radius: f32, num_points: usize) -> PyPointCloud {
        let pc = reconstruction::create_sphere_point_cloud(
            Point3::new(center.0, center.1, center.2),
            radius,
            num_points,
        );
        PyPointCloud {
            points: pc.points,
            normals: pc.normals,
        }
    }

    #[staticmethod]
    pub fn create_plane(
        origin: (f32, f32, f32),
        normal: (f32, f32, f32),
        size: f32,
        num_points: usize,
    ) -> PyPointCloud {
        let pc = reconstruction::create_plane_point_cloud(
            Point3::new(origin.0, origin.1, origin.2),
            Vector3::new(normal.0, normal.1, normal.2),
            size,
            num_points,
        );
        PyPointCloud {
            points: pc.points,
            normals: pc.normals,
        }
    }

    #[staticmethod]
    pub fn ball_pivoting(cloud: &PyPointCloud, ball_radius: f32) -> PyTriangleMesh {
        let pc = PointCloud {
            points: cloud.points.clone(),
            normals: cloud.normals.clone(),
            colors: None,
        };
        let mesh = reconstruction::ball_pivoting(&pc, ball_radius);
        PyTriangleMesh {
            vertices: mesh.vertices,
            faces: mesh.faces,
        }
    }

    #[staticmethod]
    pub fn alpha_shapes(cloud: &PyPointCloud, alpha: f32) -> PyTriangleMesh {
        let pc = PointCloud {
            points: cloud.points.clone(),
            normals: cloud.normals.clone(),
            colors: None,
        };
        let mesh = reconstruction::alpha_shapes(&pc, alpha);
        PyTriangleMesh {
            vertices: mesh.vertices,
            faces: mesh.faces,
        }
    }

    #[staticmethod]
    pub fn poisson(cloud: &PyPointCloud, depth: usize) -> Option<PyTriangleMesh> {
        let pc = PointCloud {
            points: cloud.points.clone(),
            normals: cloud.normals.clone(),
            colors: None,
        };
        reconstruction::poisson_reconstruction(&pc, depth, 1.0).map(|mesh| PyTriangleMesh {
            vertices: mesh.vertices,
            faces: mesh.faces,
        })
    }
}

// ============== Tensor Bindings ==============

#[pyclass]
pub struct PyTensor {
    data: Vec<f32>,
    shape: (usize, usize, usize),
}

#[pymethods]
impl PyTensor {
    #[new]
    pub fn new(shape: (usize, usize, usize)) -> Self {
        let size = shape.0 * shape.1 * shape.2;
        Self {
            data: vec![0.0; size],
            shape,
        }
    }

    pub fn zeros(shape: (usize, usize, usize)) -> Self {
        let size = shape.0 * shape.1 * shape.2;
        Self {
            data: vec![0.0; size],
            shape,
        }
    }

    pub fn ones(shape: (usize, usize, usize)) -> Self {
        let size = shape.0 * shape.1 * shape.2;
        Self {
            data: vec![1.0; size],
            shape,
        }
    }

    pub fn from_numpy(data: PyReadonlyArray2<f32>) -> PyResult<Self> {
        let shape = data.shape();
        if shape.len() != 3 {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "Expected 3D array",
            ));
        }
        let data = data.as_slice()?.to_vec();
        Ok(Self {
            data,
            shape: (shape[0], shape[1], shape[2]),
        })
    }

    pub fn to_numpy(&self) -> Vec<f32> {
        self.data.clone()
    }

    pub fn shape(&self) -> (usize, usize, usize) {
        self.shape
    }
}

// ============== Runtime Bindings ==============

#[pyclass]
pub struct PyRuntime;

#[pymethods]
impl PyRuntime {
    #[staticmethod]
    pub fn get_global_load() -> PyResult<Vec<(u32, usize)>> {
        let _s = scheduler()
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
        Ok(vec![])
    }

    #[staticmethod]
    pub fn get_num_devices() -> usize {
        1
    }
}

#[pymodule]
fn cv_native(_py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyWorkloadHint>()?;
    m.add_class::<PySlam>()?;
    m.add_class::<PyRuntime>()?;
    m.add_class::<PyVideoCapture>()?;
    m.add_class::<PyDnnNet>()?;
    m.add_class::<PyKeyPoints>()?;
    m.add_class::<PyFeatureDetector>()?;
    m.add_class::<PyIsam2>()?;
    m.add_class::<PyPointCloud>()?;
    m.add_class::<PyTriangleMesh>()?;
    m.add_class::<PyMeshReconstruction>()?;
    m.add_class::<PyTensor>()?;
    Ok(())
}
