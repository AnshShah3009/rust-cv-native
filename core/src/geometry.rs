use nalgebra::{Matrix3, Matrix4, Point2, Point3, Vector3};

pub type Vector6<T> = nalgebra::Vector6<T>;

#[derive(Debug, Clone, Copy)]
pub struct CameraIntrinsics {
    pub fx: f64,
    pub fy: f64,
    pub cx: f64,
    pub cy: f64,
    pub width: u32,
    pub height: u32,
}

impl CameraIntrinsics {
    pub fn new(fx: f64, fy: f64, cx: f64, cy: f64, width: u32, height: u32) -> Self {
        Self {
            fx,
            fy,
            cx,
            cy,
            width,
            height,
        }
    }

    pub fn new_ideal(width: u32, height: u32) -> Self {
        let fx = width as f64;
        Self {
            fx,
            fy: fx,
            cx: fx / 2.0,
            cy: height as f64 / 2.0,
            width,
            height,
        }
    }

    pub fn matrix(&self) -> Matrix3<f64> {
        Matrix3::new(self.fx, 0.0, self.cx, 0.0, self.fy, self.cy, 0.0, 0.0, 1.0)
    }

    pub fn inverse_matrix(&self) -> Matrix3<f64> {
        self.matrix().try_inverse().unwrap_or(Matrix3::identity())
    }

    pub fn project(&self, point: &Point3<f64>) -> Point2<f64> {
        let z = point.z;
        if z == 0.0 {
            return Point2::new(self.cx, self.cy);
        }
        let x = point.x / z;
        let y = point.y / z;
        Point2::new(x * self.fx + self.cx, y * self.fy + self.cy)
    }

    pub fn unproject(&self, pixel: Point2<f64>, depth: f64) -> Point3<f64> {
        let x = (pixel.x - self.cx) / self.fx;
        let y = (pixel.y - self.cy) / self.fy;
        Point3::new(x * depth, y * depth, depth)
    }
}

pub type CameraIntrinsicsf32 = CameraIntrinsicsF32;

#[derive(Debug, Clone, Copy)]
pub struct CameraIntrinsicsF32 {
    pub fx: f32,
    pub fy: f32,
    pub cx: f32,
    pub cy: f32,
    pub width: u32,
    pub height: u32,
}

impl CameraIntrinsicsF32 {
    pub fn new(fx: f32, fy: f32, cx: f32, cy: f32, width: u32, height: u32) -> Self {
        Self {
            fx,
            fy,
            cx,
            cy,
            width,
            height,
        }
    }

    pub fn from_intrinsics(i: &CameraIntrinsics) -> Self {
        Self {
            fx: i.fx as f32,
            fy: i.fy as f32,
            cx: i.cx as f32,
            cy: i.cy as f32,
            width: i.width,
            height: i.height,
        }
    }

    pub fn matrix(&self) -> Matrix3<f32> {
        Matrix3::new(self.fx, 0.0, self.cx, 0.0, self.fy, self.cy, 0.0, 0.0, 1.0)
    }

    pub fn project(&self, point: &Point3<f32>) -> Point2<f32> {
        let z = point.z;
        if z == 0.0 {
            return Point2::new(self.cx, self.cy);
        }
        let x = point.x / z;
        let y = point.y / z;
        Point2::new(x * self.fx + self.cx, y * self.fy + self.cy)
    }

    pub fn unproject(&self, pixel: Point2<f32>, depth: f32) -> Point3<f32> {
        let x = (pixel.x - self.cx) / self.fx;
        let y = (pixel.y - self.cy) / self.fy;
        Point3::new(x * depth, y * depth, depth)
    }
}

#[derive(Debug, Clone, Copy)]
pub struct CameraExtrinsics {
    pub rotation: Matrix3<f64>,
    pub translation: Vector3<f64>,
}

impl CameraExtrinsics {
    pub fn new(rotation: Matrix3<f64>, translation: Vector3<f64>) -> Self {
        Self {
            rotation,
            translation,
        }
    }

    pub fn from_rotation_translation(r: &Matrix3<f64>, t: &Vector3<f64>) -> Self {
        Self {
            rotation: *r,
            translation: *t,
        }
    }

    pub fn matrix(&self) -> Matrix4<f64> {
        let mut m = Matrix4::identity();
        m.fixed_view_mut::<3, 3>(0, 0).copy_from(&self.rotation);
        m.fixed_view_mut::<3, 1>(0, 3).copy_from(&self.translation);
        m
    }

    pub fn transform_point(&self, point: &Point3<f64>) -> Point3<f64> {
        let transformed = self.rotation * point.coords + self.translation;
        Point3::from(transformed)
    }

    pub fn inverse(&self) -> Self {
        let r_inv = self.rotation.transpose();
        let t_inv = -r_inv * self.translation;
        Self {
            rotation: r_inv,
            translation: t_inv,
        }
    }

    pub fn compose(&self, other: &Self) -> Self {
        Self {
            rotation: self.rotation * other.rotation,
            translation: self.rotation * other.translation + self.translation,
        }
    }
}

impl Default for CameraExtrinsics {
    fn default() -> Self {
        Self {
            rotation: Matrix3::identity(),
            translation: Vector3::zeros(),
        }
    }
}

pub type CameraExtrinsicsf32 = CameraExtrinsicsF32;

#[derive(Debug, Clone, Copy)]
pub struct CameraExtrinsicsF32 {
    pub rotation: Matrix3<f32>,
    pub translation: Vector3<f32>,
}

impl CameraExtrinsicsF32 {
    pub fn new(rotation: Matrix3<f32>, translation: Vector3<f32>) -> Self {
        Self {
            rotation,
            translation,
        }
    }

    pub fn from_extrinsics(e: &CameraExtrinsics) -> Self {
        Self {
            rotation: Matrix3::new(
                e.rotation.m11 as f32,
                e.rotation.m12 as f32,
                e.rotation.m13 as f32,
                e.rotation.m21 as f32,
                e.rotation.m22 as f32,
                e.rotation.m23 as f32,
                e.rotation.m31 as f32,
                e.rotation.m32 as f32,
                e.rotation.m33 as f32,
            ),
            translation: Vector3::new(
                e.translation[0] as f32,
                e.translation[1] as f32,
                e.translation[2] as f32,
            ),
        }
    }

    pub fn matrix(&self) -> Matrix4<f32> {
        let mut m = Matrix4::identity();
        m.fixed_view_mut::<3, 3>(0, 0).copy_from(&self.rotation);
        m.fixed_view_mut::<3, 1>(0, 3).copy_from(&self.translation);
        m
    }

    pub fn transform_point(&self, point: &Point3<f32>) -> Point3<f32> {
        let transformed = self.rotation * point.coords + self.translation;
        Point3::from(transformed)
    }

    pub fn inverse(&self) -> Self {
        let r_inv = self.rotation.transpose();
        let t_inv = -r_inv * self.translation;
        Self {
            rotation: r_inv,
            translation: t_inv,
        }
    }
}

impl Default for CameraExtrinsicsF32 {
    fn default() -> Self {
        Self {
            rotation: Matrix3::identity(),
            translation: Vector3::zeros(),
        }
    }
}

pub fn twist_to_se3(twist: &Vector6<f64>) -> (Matrix3<f64>, Vector3<f64>) {
    let omega = Vector3::new(twist[0], twist[1], twist[2]);
    let v = Vector3::new(twist[3], twist[4], twist[5]);

    let theta = omega.norm();
    if theta < 1e-10 {
        (Matrix3::identity(), v)
    } else {
        let axis = omega / theta;
        let skew = skew_symmetric(&axis);
        let r = Matrix3::identity() + theta * skew + (1.0 - theta.cos()) * skew * skew;
        let t =
            (Matrix3::identity() + theta * skew + (1.0 - theta.cos()) * skew * skew) * v / theta;
        (r, t)
    }
}

pub fn skew_symmetric(v: &Vector3<f64>) -> Matrix3<f64> {
    Matrix3::new(0.0, -v[2], v[1], v[2], 0.0, -v[0], -v[1], v[0], 0.0)
}

#[derive(Debug, Clone, Copy)]
pub struct Distortion {
    pub k1: f64,
    pub k2: f64,
    pub p1: f64,
    pub p2: f64,
    pub k3: f64,
}

impl Distortion {
    pub fn new(k1: f64, k2: f64, p1: f64, p2: f64, k3: f64) -> Self {
        Self { k1, k2, p1, p2, k3 }
    }

    pub fn none() -> Self {
        Self {
            k1: 0.0,
            k2: 0.0,
            p1: 0.0,
            p2: 0.0,
            k3: 0.0,
        }
    }

    pub fn apply(&self, x: f64, y: f64) -> (f64, f64) {
        let r2 = x * x + y * y;
        let radial = 1.0 + self.k1 * r2 + self.k2 * r2 * r2 + self.k3 * r2 * r2 * r2;
        let dx = 2.0 * self.p1 * x * y + self.p2 * (r2 + 2.0 * x * x);
        let dy = self.p1 * (r2 + 2.0 * y * y) + 2.0 * self.p2 * x * y;
        (x * radial + dx, y * radial + dy)
    }

    pub fn remove(&self, x: f64, y: f64) -> (f64, f64) {
        let mut xd = x;
        let mut yd = y;
        for _ in 0..10 {
            let (xu, yu) = self.apply(xd, yd);
            xd += x - xu;
            yd += y - yu;
        }
        (xd, yd)
    }
}

impl Default for Distortion {
    fn default() -> Self {
        Self::none()
    }
}

pub type Distortionf32 = DistortionF32;

#[derive(Debug, Clone, Copy)]
pub struct DistortionF32 {
    pub k1: f32,
    pub k2: f32,
    pub p1: f32,
    pub p2: f32,
    pub k3: f32,
}

impl DistortionF32 {
    pub fn new(k1: f32, k2: f32, p1: f32, p2: f32, k3: f32) -> Self {
        Self { k1, k2, p1, p2, k3 }
    }

    pub fn from_distortion(d: &Distortion) -> Self {
        Self {
            k1: d.k1 as f32,
            k2: d.k2 as f32,
            p1: d.p1 as f32,
            p2: d.p2 as f32,
            k3: d.k3 as f32,
        }
    }

    pub fn none() -> Self {
        Self {
            k1: 0.0,
            k2: 0.0,
            p1: 0.0,
            p2: 0.0,
            k3: 0.0,
        }
    }

    pub fn apply(&self, x: f32, y: f32) -> (f32, f32) {
        let r2 = x * x + y * y;
        let radial = 1.0 + self.k1 * r2 + self.k2 * r2 * r2 + self.k3 * r2 * r2 * r2;
        let dx = 2.0 * self.p1 * x * y + self.p2 * (r2 + 2.0 * x * x);
        let dy = self.p1 * (r2 + 2.0 * y * y) + 2.0 * self.p2 * x * y;
        (x * radial + dx, y * radial + dy)
    }

    pub fn remove(&self, x: f32, y: f32) -> (f32, f32) {
        let mut xd = x;
        let mut yd = y;
        for _ in 0..10 {
            let (xu, yu) = self.apply(xd, yd);
            xd += x - xu;
            yd += y - yu;
        }
        (xd, yd)
    }
}

impl Default for DistortionF32 {
    fn default() -> Self {
        Self::none()
    }
}

#[derive(Debug, Clone, Copy, Default)]
pub struct Rect {
    pub x: f32,
    pub y: f32,
    pub w: f32,
    pub h: f32,
}

impl Rect {
    pub fn new(x: f32, y: f32, w: f32, h: f32) -> Self {
        Self { x, y, w, h }
    }

    pub fn x1(&self) -> f32 {
        self.x
    }
    pub fn y1(&self) -> f32 {
        self.y
    }
    pub fn x2(&self) -> f32 {
        self.x + self.w
    }
    pub fn y2(&self) -> f32 {
        self.y + self.h
    }

    pub fn area(&self) -> f32 {
        self.w * self.h
    }

    pub fn iou(&self, other: &Rect) -> f32 {
        let x1 = self.x1().max(other.x1());
        let y1 = self.y1().max(other.y1());
        let x2 = self.x2().min(other.x2());
        let y2 = self.y2().min(other.y2());

        let w = (x2 - x1).max(0.0);
        let h = (y2 - y1).max(0.0);
        let intersection = w * h;

        if intersection == 0.0 {
            return 0.0;
        }

        let union = self.area() + other.area() - intersection;
        intersection / union
    }
}

#[derive(Debug, Clone, Copy, Default)]
pub struct RotatedRect {
    pub cx: f32,
    pub cy: f32,
    pub w: f32,
    pub h: f32,
    pub angle: f32, // Degrees
}

impl RotatedRect {
    pub fn new(cx: f32, cy: f32, w: f32, h: f32, angle: f32) -> Self {
        Self { cx, cy, w, h, angle }
    }

    pub fn area(&self) -> f32 {
        self.w * self.h
    }

    /// Get the 4 corners of the rotated rectangle
    pub fn points(&self) -> [[f32; 2]; 4] {
        let angle_rad = self.angle.to_radians();
        let cos_a = angle_rad.cos();
        let sin_a = angle_rad.sin();

        let half_w = self.w / 2.0;
        let half_h = self.h / 2.0;

        let mut pts = [[0.0, 0.0]; 4];
        // Relative corners before rotation
        let corners = [
            [-half_w, -half_h],
            [half_w, -half_h],
            [half_w, half_h],
            [-half_w, half_h],
        ];

        for i in 0..4 {
            pts[i][0] = self.cx + corners[i][0] * cos_a - corners[i][1] * sin_a;
            pts[i][1] = self.cy + corners[i][0] * sin_a + corners[i][1] * cos_a;
        }
        pts
    }
}

#[derive(Debug, Clone, Default)]
pub struct Polygon {
    pub points: Vec<[f32; 2]>,
}

impl Polygon {
    pub fn new(points: Vec<[f32; 2]>) -> Self {
        Self { points }
    }

    pub fn area(&self) -> f32 {
        if self.points.len() < 3 {
            return 0.0;
        }
        let mut area = 0.0;
        for i in 0..self.points.len() {
            let p1 = self.points[i];
            let p2 = self.points[(i + 1) % self.points.len()];
            area += p1[0] * p2[1] - p2[0] * p1[1];
        }
        area.abs() * 0.5
    }
}


