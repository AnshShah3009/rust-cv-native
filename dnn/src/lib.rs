//! Deep Learning Module
//!
//! Provides a high-level interface for running neural network inference
//! using ONNX Runtime (via the `ort` crate).

use cv_core::Tensor;
use cv_hal::compute::ComputeDevice;
use cv_runtime::orchestrator::ResourceGroup;
use image::DynamicImage;
use ort::session::Session;
use ort::value::Value;
use std::path::Path;
use std::sync::Arc;
use rayon::prelude::*;

pub type Result<T> = std::result::Result<T, DnnError>;

#[derive(Debug, thiserror::Error)]
pub enum DnnError {
    #[error("Inference error: {0}")]
    Inference(String),

    #[error("Initialization error: {0}")]
    Initialization(String),

    #[error("Preprocessing error: {0}")]
    Preprocessing(String),

    #[error("Ort error: {0}")]
    Ort(#[from] ort::Error),
}

pub struct DnnNet {
    session: Arc<Session>,
    input_shape: Vec<usize>,
}

impl DnnNet {
    /// Load an ONNX model from file
    pub fn load<P: AsRef<Path>>(path: P) -> Result<Self> {
        let session = Session::builder()?
            .commit_from_file(path)
            .map_err(|e| DnnError::Initialization(e.to_string()))?;
        
        let session = Arc::new(session);
        
        // Extract input shape from model (assuming first input)
        let input_shape = session.inputs[0].input_type.tensor_type().unwrap().shape.clone();
        let shape: Vec<usize> = input_shape.iter().map(|&s| s.unwrap_or(1) as usize).collect();

        Ok(Self {
            session,
            input_shape: shape,
        })
    }

    /// Forward pass through the network
    pub fn forward(&self, input: &Tensor<f32>) -> Result<Vec<Tensor<f32>>> {
        let input_shape = (
            self.input_shape[0],
            self.input_shape[1],
            self.input_shape[2],
            self.input_shape[3]
        );
        
        let array = ndarray::Array4::from_shape_vec(input_shape, input.as_slice().to_vec())
            .map_err(|e| DnnError::Inference(e.to_string()))?;

        // ort::Value::from_array returns Result<Value, Error>
        let input_value = Value::from_array(array)
            .map_err(|e| DnnError::Inference(e.to_string()))?;

        // Session::run takes something that can be converted into SessionInputs
        let outputs = self.session.run(vec![input_value])
            .map_err(|e| DnnError::Inference(e.to_string()))?;

        let mut results = Vec::new();
        for (_, value) in outputs {
            // try_extract_tensor returns Result<TensorView<T>, Error>
            let view = value.try_extract_tensor::<f32>()
                .map_err(|e| DnnError::Inference(e.to_string()))?;
            
            let shape = view.shape();
            
            let tensor_shape = match shape.len() {
                1 => cv_core::TensorShape::new(1, 1, shape[0] as usize),
                2 => cv_core::TensorShape::new(1, shape[0] as usize, shape[1] as usize),
                3 => cv_core::TensorShape::new(shape[0] as usize, shape[1] as usize, shape[2] as usize),
                4 => cv_core::TensorShape::new(shape[1] as usize, shape[2] as usize, shape[3] as usize), // Assuming NCHW
                _ => cv_core::TensorShape::new(1, 1, shape.iter().map(|&s| s as usize).product()),
            };
            
            results.push(Tensor::from_vec(view.as_slice().unwrap().to_vec(), tensor_shape));
        }

        Ok(results)
    }

    /// Preprocess an image for the network (Resize + Normalize)
    pub fn preprocess(&self, img: &DynamicImage, _ctx: &ComputeDevice, group: &ResourceGroup) -> Result<Tensor<f32>> {
        let target_w = self.input_shape[3];
        let target_h = self.input_shape[2];
        let channels = self.input_shape[1];
        
        let gray = img.to_luma8();
        let resized = cv_imgproc::resize_ctx(&gray, target_w as u32, target_h as u32, cv_imgproc::Interpolation::Linear, group);
        
        // Use standard iter for now to avoid par_iter trait issues
        let data: Vec<f32> = resized.as_raw().iter().map(|&v| v as f32 / 255.0).collect();

        Ok(Tensor::from_vec(data, cv_core::TensorShape::new(channels, target_h, target_w)))
    }
}
