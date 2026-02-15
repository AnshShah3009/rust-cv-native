pub mod geometry;
pub mod integrate;
pub mod jit;
pub mod point_cloud;
pub mod special;

pub use geometry::*;
pub use integrate::*;
pub use jit::*;
pub use point_cloud::*;
pub use special::*;

pub fn mean(data: &[f64]) -> Option<f64> {
    if data.is_empty() {
        return None;
    }
    let sum: f64 = data.iter().sum();
    Some(sum / data.len() as f64)
}

pub fn std(data: &[f64]) -> Option<f64> {
    let m = mean(data)?;
    let variance = data.iter().map(|&x| (x - m).powi(2)).sum::<f64>() / data.len() as f64;
    Some(variance.sqrt())
}

pub fn lerp(a: f64, b: f64, t: f64) -> f64 {
    a * (1.0 - t) + b * t
}

pub struct Interp1d {
    pub x: Vec<f64>,
    pub y: Vec<f64>,
}

impl Interp1d {
    pub fn new(x: Vec<f64>, y: Vec<f64>) -> Option<Self> {
        if x.len() != y.len() || x.is_empty() {
            return None;
        }
        Some(Self { x, y })
    }

    pub fn call(&self, x_val: f64) -> f64 {
        if x_val <= self.x[0] {
            return self.y[0];
        }
        if x_val >= *self.x.last().unwrap() {
            return *self.y.last().unwrap();
        }

        let idx = self
            .x
            .binary_search_by(|v| v.partial_cmp(&x_val).unwrap())
            .unwrap_or_else(|pos| pos - 1);
        let t = (x_val - self.x[idx]) / (self.x[idx + 1] - self.x[idx]);
        lerp(self.y[idx], self.y[idx + 1], t)
    }
}
