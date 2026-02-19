//! Robust Estimation Module
//!
//! Provides a generic RANSAC implementation that can be used for any model estimation task.

use std::marker::PhantomData;
use rand::seq::SliceRandom;

/// Configuration for robust estimation
#[derive(Debug, Clone)]
pub struct RobustConfig {
    pub threshold: f64,
    pub max_iterations: usize,
    pub confidence: f64,
    pub min_sample_size: usize,
}

impl Default for RobustConfig {
    fn default() -> Self {
        Self {
            threshold: 1.0,
            max_iterations: 1000,
            confidence: 0.99,
            min_sample_size: 4,
        }
    }
}

/// Result of robust estimation
#[derive(Debug, Clone)]
pub struct RobustResult<M> {
    pub model: Option<M>,
    pub inliers: Vec<bool>,
    pub num_inliers: usize,
    pub residual: f64,
}

/// Trait for models that can be estimated robustly
pub trait RobustModel<D> {
    type Model: Clone;

    /// Minimum number of data points required to estimate the model
    fn min_sample_size(&self) -> usize;

    /// Estimate model from a minimal sample
    fn estimate(&self, data: &[&D]) -> Option<Self::Model>;

    /// Compute error for a single data point against the model
    fn compute_error(&self, model: &Self::Model, data: &D) -> f64;
}

/// Generic RANSAC engine
pub struct Ransac<D, M: RobustModel<D>> {
    config: RobustConfig,
    _phantom: PhantomData<(D, M)>,
}

impl<D, M: RobustModel<D>> Ransac<D, M> {
    pub fn new(config: RobustConfig) -> Self {
        Self {
            config,
            _phantom: PhantomData,
        }
    }

    pub fn run(&self, estimator: &M, data: &[D]) -> RobustResult<M::Model> {
        let n = data.len();
        let k = estimator.min_sample_size();

        if n < k {
            return RobustResult {
                model: None,
                inliers: vec![false; n],
                num_inliers: 0,
                residual: f64::INFINITY,
            };
        }

        let mut best_model = None;
        let mut best_inliers = vec![false; n];
        let mut best_num_inliers = 0;
        let mut best_residual = f64::INFINITY;

        let mut rng = rand::thread_rng();
        let mut indices: Vec<usize> = (0..n).collect();

        for _ in 0..self.config.max_iterations {
            // 1. Sample
            indices.shuffle(&mut rng);
            let sample: Vec<&D> = (0..k).map(|i| &data[indices[i]]).collect();

            // 2. Estimate
            if let Some(model) = estimator.estimate(&sample) {
                // 3. Score
                let mut inliers = vec![false; n];
                let mut num_inliers = 0;
                let mut total_error = 0.0;

                for (j, d) in data.iter().enumerate() {
                    let err = estimator.compute_error(&model, d);
                    if err < self.config.threshold {
                        inliers[j] = true;
                        num_inliers += 1;
                        total_error += err;
                    }
                }

                let residual = if num_inliers > 0 { total_error / num_inliers as f64 } else { f64::INFINITY };

                if num_inliers > best_num_inliers || (num_inliers == best_num_inliers && residual < best_residual) {
                    best_num_inliers = num_inliers;
                    best_inliers = inliers;
                    best_model = Some(model);
                    best_residual = residual;

                    // Early exit check
                    if num_inliers as f64 > n as f64 * self.config.confidence {
                        break;
                    }
                }
            }
        }

        RobustResult {
            model: best_model,
            inliers: best_inliers,
            num_inliers: best_num_inliers,
            residual: best_residual,
        }
    }
}

/// Generic LMedS (Least Median of Squares) engine
pub struct LMedS<D, M: RobustModel<D>> {
    config: RobustConfig,
    _phantom: PhantomData<(D, M)>,
}

impl<D, M: RobustModel<D>> LMedS<D, M> {
    pub fn new(config: RobustConfig) -> Self {
        Self {
            config,
            _phantom: PhantomData,
        }
    }

    pub fn run(&self, estimator: &M, data: &[D]) -> RobustResult<M::Model> {
        let n = data.len();
        let k = estimator.min_sample_size();

        if n < k {
            return RobustResult {
                model: None,
                inliers: vec![false; n],
                num_inliers: 0,
                residual: f64::INFINITY,
            };
        }

        let mut best_model = None;
        let mut best_median_error = f64::INFINITY;

        let mut rng = rand::thread_rng();
        let mut indices: Vec<usize> = (0..n).collect();

        for _ in 0..self.config.max_iterations {
            indices.shuffle(&mut rng);
            let sample: Vec<&D> = (0..k).map(|i| &data[indices[i]]).collect();

            if let Some(model) = estimator.estimate(&sample) {
                let mut errors: Vec<f64> = data.iter()
                    .map(|d| estimator.compute_error(&model, d))
                    .collect();
                
                // Sort errors to find median
                errors.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
                
                let median_error = errors[n / 2];

                if median_error < best_median_error {
                    best_median_error = median_error;
                    best_model = Some(model);
                }
            }
        }

        // Final inlier count based on the best model and a derived threshold
        // Standard LMedS threshold: 2.5 * 1.4826 * (1 + 5/(n-k)) * sqrt(best_median_error)
        let mut inliers = vec![false; n];
        let mut num_inliers = 0;
        let mut best_residual = f64::INFINITY;

        if let Some(ref model) = best_model {
            let sigma = 1.4826 * (1.0 + 5.0 / (n - k) as f64) * best_median_error.sqrt();
            let threshold = 2.5 * sigma;
            let mut total_error = 0.0;

            for (j, d) in data.iter().enumerate() {
                let err = estimator.compute_error(model, d);
                if err < threshold {
                    inliers[j] = true;
                    num_inliers += 1;
                    total_error += err;
                }
            }
            best_residual = if num_inliers > 0 { total_error / num_inliers as f64 } else { f64::INFINITY };
        }

        RobustResult {
            model: best_model,
            inliers,
            num_inliers,
            residual: best_residual,
        }
    }
}
