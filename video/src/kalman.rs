use nalgebra::{DMatrix, DVector};

pub struct KalmanFilter {
    pub state_pre: DVector<f64>,   // x_k|k-1
    pub state_post: DVector<f64>,  // x_k|k
    pub trans_matrix: DMatrix<f64>, // A
    pub control_matrix: DMatrix<f64>, // B
    pub meas_matrix: DMatrix<f64>,  // H
    pub process_noise_cov: DMatrix<f64>, // Q
    pub meas_noise_cov: DMatrix<f64>,    // R
    pub error_cov_pre: DMatrix<f64>,     // P_k|k-1
    pub gain: DMatrix<f64>,              // K
    pub error_cov_post: DMatrix<f64>,    // P_k|k
}

impl KalmanFilter {
    pub fn new(dynam_params: usize, meas_params: usize, control_params: usize) -> Self {
        Self {
            state_pre: DVector::zeros(dynam_params),
            state_post: DVector::zeros(dynam_params),
            trans_matrix: DMatrix::identity(dynam_params, dynam_params),
            control_matrix: DMatrix::zeros(dynam_params, control_params),
            meas_matrix: DMatrix::zeros(meas_params, dynam_params),
            process_noise_cov: DMatrix::identity(dynam_params, dynam_params),
            meas_noise_cov: DMatrix::identity(meas_params, meas_params),
            error_cov_pre: DMatrix::zeros(dynam_params, dynam_params),
            gain: DMatrix::zeros(dynam_params, meas_params),
            error_cov_post: DMatrix::zeros(dynam_params, dynam_params),
        }
    }

    pub fn predict(&mut self, control: Option<&DVector<f64>>) -> &DVector<f64> {
        self.state_pre = &self.trans_matrix * &self.state_post;
        if let Some(u) = control {
            self.state_pre += &self.control_matrix * u;
        }
        
        self.error_cov_pre = &self.trans_matrix * &self.error_cov_post * self.trans_matrix.transpose() + &self.process_noise_cov;
        
        &self.state_pre
    }

    pub fn correct(&mut self, measurement: &DVector<f64>) -> &DVector<f64> {
        let s = &self.meas_matrix * &self.error_cov_pre * self.meas_matrix.transpose() + &self.meas_noise_cov;
        self.gain = &self.error_cov_pre * self.meas_matrix.transpose() * s.pseudo_inverse(1e-6).unwrap();
        
        let innovation = measurement - &self.meas_matrix * &self.state_pre;
        self.state_post = &self.state_pre + &self.gain * innovation;
        
        let i = DMatrix::identity(self.state_post.len(), self.state_post.len());
        self.error_cov_post = (&i - &self.gain * &self.meas_matrix) * &self.error_cov_pre;
        
        &self.state_post
    }
}
