use nalgebra::{Matrix3, Matrix4, Point2, Point3, UnitQuaternion, Vector3, Vector4};
use std::ops::Mul;

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
        let fy = width as f64;
        let cx = width as f64 / 2.0;
        let cy = height as f64 / 2.0;
        Self {
            fx,
            fy,
            cx,
            cy,
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
        let x = point.x / point.z;
        let y = point.y / self.z;
        Point2::new(x * self.fx + self.cx, y * self.fy + self.cy)
    }

    pub fn unproject(&self, pixel: Point2<f64>, depth: f64) -> Point3<f64> {
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

    pub fn from_twist(twist: &Vector6<f64>) -> Self {
        let (r, t) = twist_to_se3(twist);
        Self {
            rotation: r,
            translation: t,
        }
    }

    pub fn matrix(&self) -> Matrix4<f64> {
        let mut m = Matrix4::identity();
        m.fixed_view_mut::<3, 3>(0, 0).copy_from(&self.rotation);
        m.fixed_view_mut::<3, 1>(0, 3).copy_from(&self.translation);
        m
    }

    pub fn transform_point(&self, point: &Point3<f64>) -> Point3<f64> {
        self.rotation * point.coords + self.translation
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

impl Default for CameraExtrinsics {
    fn default() -> Self {
        Self {
            rotation: Matrix3::identity(),
            translation: Vector3::zeros(),
        }
    }
}

pub type Vector6<T> = nalgebra::Vector6<T>;

#[derive(Debug, Clone, Copy)]
pub struct Pose {
    pub rotation: UnitQuaternion<f64>,
    pub translation: Vector3<f64>,
}

impl Pose {
    pub fn new(rotation: UnitQuaternion<f64>, translation: Vector3<f64>) -> Self {
        Self {
            rotation,
            translation,
        }
    }

    pub fn from_matrix(transform: &Matrix4<f64>) -> Self {
        let r = Matrix3::from(transform.fixed_view::<3, 3>(0, 0));
        let t = Vector3::from(transform.fixed_view::<3, 1>(0, 3));
        let rotation = UnitQuaternion::from_rotation_matrix(&r);
        Self {
            rotation,
            translation: t,
        }
    }

    pub fn matrix(&self) -> Matrix4<f64> {
        let mut m = Matrix4::identity();
        m.fixed_view_mut::<3, 3>(0, 0)
            .copy_from(self.rotation.matrix());
        m.fixed_view_mut::<3, 1>(0, 3).copy_from(&self.translation);
        m
    }

    pub fn transform_point(&self, point: &Point3<f64>) -> Point3<f64> {
        self.rotation * point + self.translation
    }
}

impl Default for Pose {
    fn default() -> Self {
        Self {
            rotation: UnitQuaternion::identity(),
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
