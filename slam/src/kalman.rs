//! Kalman Filter Implementation
//!
//! Provides Extended Kalman Filter (EKF) and Unscented Kalman Filter (UKF)
//! for state estimation in SLAM and robotics applications.

use nalgebra::{
    DMatrix, DVector, Matrix, Matrix2, Matrix3, RealField, SMatrix, SVector, Vector2, Vector3,
};
use std::fmt::Debug;

/// Kalman Filter state
#[derive(Debug, Clone)]
pub struct KalmanFilterState<const N: usize> {
    /// State vector
    pub x: SVector<f64, N>,
    /// Covariance matrix
    pub p: SMatrix<f64, N, N>,
}

impl<const N: usize> KalmanFilterState<N> {
    /// Create a new state from initial values
    pub fn new(x: SVector<f64, N>, p: SMatrix<f64, N, N>) -> Self {
        Self { x, p }
    }

    /// Initialize with zero state and large uncertainty
    pub fn zero() -> Self {
        Self {
            x: SVector::zeros(),
            p: SMatrix::identity() * 1000.0,
        }
    }

    /// Get the state uncertainty as standard deviations
    pub fn std_dev(&self) -> SVector<f64, N> {
        self.p.diagonal().map(|v| v.sqrt())
    }
}

/// Standard (Linear) Kalman Filter
#[derive(Debug, Clone)]
pub struct KalmanFilter<const N: usize, const M: usize> {
    /// State transition matrix
    pub f: SMatrix<f64, N, N>,
    /// Control input matrix
    pub b: SMatrix<f64, N, N>,
    /// Observation matrix
    pub h: SMatrix<f64, M, N>,
    /// Process noise covariance
    pub q: SMatrix<f64, N, N>,
    /// Measurement noise covariance
    pub r: SMatrix<f64, M, M>,
}

impl<const N: usize, const M: usize> KalmanFilter<N, M> {
    /// Create a new Kalman Filter with given matrices
    pub fn new(
        f: SMatrix<f64, N, N>,
        b: SMatrix<f64, N, N>,
        h: SMatrix<f64, M, N>,
        q: SMatrix<f64, N, N>,
        r: SMatrix<f64, M, M>,
    ) -> Self {
        Self { f, b, h, q, r }
    }

    /// Prediction step
    pub fn predict(&self, state: &mut KalmanFilterState<N>, u: &SVector<f64, N>) {
        // x = F*x + B*u
        state.x = self.f * state.x + self.b * u;
        // P = F*P*F^T + Q
        state.p = self.f * state.p * self.f.transpose() + self.q;
    }

    /// Update step with measurement
    pub fn update(&self, state: &mut KalmanFilterState<N>, z: &SVector<f64, M>) {
        // y = z - H*x (innovation)
        let y = z - self.h * state.x;

        // S = H*P*H^T + R (innovation covariance)
        let s = self.h * state.p * self.h.transpose() + self.r;

        // K = P*H^T*S^-1 (Kalman gain)
        let k = state.p * self.h.transpose() * s.try_inverse().unwrap_or(SMatrix::identity());

        // x = x + K*y (state update)
        state.x = state.x + k * y;

        // P = (I - K*H)*P (covariance update - Joseph form for stability)
        let i = SMatrix::<f64, N, N>::identity();
        let i_kh = i - k * self.h;
        state.p = i_kh * state.p * i_kh.transpose() + k * self.r * k.transpose();
    }

    /// Single step prediction + update
    pub fn step(&self, state: &mut KalmanFilterState<N>, u: &SVector<f64, N>, z: &SVector<f64, M>) {
        self.predict(state, u);
        self.update(state, z);
    }
}

/// Extended Kalman Filter for non-linear systems
pub struct ExtendedKalmanFilter<const N: usize, const M: usize> {
    /// Process noise covariance
    pub q: SMatrix<f64, N, N>,
    /// Measurement noise covariance
    pub r: SMatrix<f64, M, M>,
}

impl<const N: usize, const M: usize> ExtendedKalmanFilter<N, M> {
    /// Create a new EKF
    pub fn new(q: SMatrix<f64, N, N>, r: SMatrix<f64, M, M>) -> Self {
        Self { q, r }
    }

    /// Prediction step with non-linear state transition
    pub fn predict<F: Fn(&SVector<f64, N>) -> SVector<f64, N>>(
        &self,
        state: &mut KalmanFilterState<N>,
        f: F,
        jacobian_f: &SMatrix<f64, N, N>,
    ) {
        // x = f(x)
        state.x = f(&state.x);
        // P = J_f*P*J_f^T + Q
        state.p = jacobian_f * state.p * jacobian_f.transpose() + self.q;
    }

    /// Update step with non-linear measurement
    pub fn update<H: Fn(&SVector<f64, N>) -> SVector<f64, M>>(
        &self,
        state: &mut KalmanFilterState<N>,
        h: H,
        jacobian_h: &SMatrix<f64, M, N>,
        z: &SVector<f64, M>,
    ) {
        // y = z - h(x) (innovation)
        let y = z - h(&state.x);

        // S = J_h*P*J_h^T + R
        let s = jacobian_h * state.p * jacobian_h.transpose() + self.r;

        // K = P*J_h^T*S^-1
        let k = state.p * jacobian_h.transpose() * s.try_inverse().unwrap_or(SMatrix::identity());

        // x = x + K*y
        state.x = state.x + k * y;

        // P = (I - K*J_h)*P
        let i = SMatrix::<f64, N, N>::identity();
        let i_kh = i - k * jacobian_h;
        state.p = i_kh * state.p * i_kh.transpose() + k * self.r * k.transpose();
    }
}

/// Utility functions for common Kalman filter setups
pub mod utils {
    use super::*;

    /// Create a constant velocity model for 2D tracking
    pub fn constant_velocity_2d(
        dt: f64,
        process_noise: f64,
        measurement_noise: f64,
    ) -> KalmanFilter<4, 2> {
        // State: [x, y, vx, vy]
        // Measurement: [x, y]

        let f = SMatrix::<f64, 4, 4>::new(
            1.0, 0.0, dt, 0.0, 0.0, 1.0, 0.0, dt, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0,
        );

        let b = SMatrix::<f64, 4, 4>::identity();

        let h = SMatrix::<f64, 2, 4>::new(1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0);

        let q = SMatrix::<f64, 4, 4>::identity() * process_noise;
        let r = SMatrix::<f64, 2, 2>::identity() * measurement_noise;

        KalmanFilter::new(f, b, h, q, r)
    }

    /// Create a constant acceleration model for 2D tracking
    pub fn constant_acceleration_2d(
        dt: f64,
        process_noise: f64,
        measurement_noise: f64,
    ) -> KalmanFilter<6, 2> {
        // State: [x, y, vx, vy, ax, ay]

        let dt2 = dt * dt / 2.0;
        let f = SMatrix::<f64, 6, 6>::new(
            1.0, 0.0, dt, 0.0, dt2, 0.0, 0.0, 1.0, 0.0, dt, 0.0, dt2, 0.0, 0.0, 1.0, 0.0, dt, 0.0,
            0.0, 0.0, 0.0, 1.0, 0.0, dt, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            1.0,
        );

        let b = SMatrix::<f64, 6, 6>::identity();

        let h =
            SMatrix::<f64, 2, 6>::new(1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0);

        let q = SMatrix::<f64, 6, 6>::identity() * process_noise;
        let r = SMatrix::<f64, 2, 2>::identity() * measurement_noise;

        KalmanFilter::new(f, b, h, q, r)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_kalman_filter_2d() {
        let kf = utils::constant_velocity_2d(0.1, 0.01, 0.1);
        let mut state = KalmanFilterState::zero();

        // Simulate tracking a moving object
        let u = SVector::<f64, 4>::zeros();
        let measurements = vec![
            Vector2::new(1.0, 1.0),
            Vector2::new(1.1, 1.1),
            Vector2::new(1.2, 1.2),
        ];

        for z in measurements {
            kf.step(&mut state, &u, &z);
        }

        // State should be close to the measurements
        assert!(state.x[0] > 0.5);
        assert!(state.x[1] > 0.5);
    }

    #[test]
    fn test_ekf_update() {
        let ekf = ExtendedKalmanFilter::<4, 2>::new(
            SMatrix::<f64, 4, 4>::identity() * 0.01,
            SMatrix::<f64, 2, 2>::identity() * 0.1,
        );

        let mut state = KalmanFilterState::new(
            SVector::<f64, 4>::new(0.0, 0.0, 1.0, 1.0),
            SMatrix::<f64, 4, 4>::identity(),
        );

        // Simple identity measurement function
        let h = |x: &SVector<f64, 4>| SVector::<f64, 2>::new(x[0], x[1]);
        let jacobian_h = SMatrix::<f64, 2, 4>::new(1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0);

        let z = SVector::<f64, 2>::new(1.0, 1.0);
        ekf.update(&mut state, h, &jacobian_h, &z);

        // State should move towards measurement
        assert!(state.x[0] > 0.0);
        assert!(state.x[1] > 0.0);
    }

    #[test]
    fn test_std_dev() {
        let state = KalmanFilterState::new(
            SVector::<f64, 2>::new(0.0, 0.0),
            SMatrix::<f64, 2, 2>::new(4.0, 0.0, 0.0, 9.0),
        );

        let std_dev = state.std_dev();
        assert!((std_dev[0] - 2.0).abs() < 1e-6);
        assert!((std_dev[1] - 3.0).abs() < 1e-6);
    }
}
