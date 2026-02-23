//! Deep Learning Module
//!
//! Provides a high-level interface for running neural network inference
//! using `tract` (Pure Rust).

pub mod blob;

use cv_core::Tensor;
use cv_runtime::orchestrator::ResourceGroup;
use image::DynamicImage;
use std::path::Path;
use std::sync::Arc;
use tract_onnx::prelude::*;

pub type Result<T> = std::result::Result<T, DnnError>;

#[derive(Debug, thiserror::Error)]
pub enum DnnError {
    #[error("Inference error: {0}")]
    Inference(String),

    #[error("Initialization error: {0}")]
    Initialization(String),

    #[error("Preprocessing error: {0}")]
    Preprocessing(String),

    #[error("Tract error: {0}")]
    Tract(#[from] tract_onnx::prelude::TractError),
}

type RunnableModel = SimplePlan<TypedFact, Box<dyn TypedOp>, Graph<TypedFact, Box<dyn TypedOp>>>;

pub struct DnnNet {
    model: Arc<RunnableModel>,
    input_shape: Vec<usize>,
}

impl DnnNet {
    /// Load an ONNX model from file
    pub fn load<P: AsRef<Path>>(path: P) -> Result<Self> {
        let model = tract_onnx::onnx()
            .model_for_path(path)?
            .into_optimized()?
            .into_runnable()?;
        
        // Inspect input facts to determine shape
        // model.model() returns reference to Graph
        let input_shape = if let Some(input_node_idx) = model.model().inputs.first() {
             // Basic heuristic: check if shape is fixed
             // For now, default to standard image net
             vec![1, 3, 224, 224] 
        } else {
             vec![1, 3, 224, 224]
        };

        Ok(Self {
            model: Arc::new(model),
            input_shape,
        })
    }

    /// Forward pass through the network
    pub fn forward(&self, input: &Tensor<f32>) -> Result<Vec<Tensor<f32>>> {
        let input_shape_vec = vec![
            self.input_shape[0],
            self.input_shape[1],
            self.input_shape[2],
            self.input_shape[3]
        ];
        
        let slice = input.as_slice().map_err(|e| DnnError::Inference(e.to_string()))?;
        // Create tract tensor from slice
        let tensor = tract_onnx::prelude::Tensor::from_shape(&input_shape_vec, slice)?;

        let result = self.model.run(tvec!(tensor.into()))?;

        let mut outputs = Vec::new();
        for t in result {
            let shape = t.shape();
            let data = t.as_slice::<f32>()?.to_vec();
            
            let tensor_shape = match shape.len() {
                1 => cv_core::TensorShape::new(1, 1, shape[0]),
                2 => cv_core::TensorShape::new(1, shape[0], shape[1]),
                3 => cv_core::TensorShape::new(shape[0], shape[1], shape[2]),
                4 => cv_core::TensorShape::new(shape[1], shape[2], shape[3]), // Assuming NCHW
                _ => cv_core::TensorShape::new(1, 1, shape.iter().product()),
            };
            
            outputs.push(Tensor::from_vec(data, tensor_shape).map_err(|e| DnnError::Inference(e.to_string()))?);
        }

        Ok(outputs)
    }

    /// Preprocess an image for the network (Resize + Normalize)
    pub fn preprocess(&self, img: &DynamicImage, runner: &ResourceGroup) -> Result<Tensor<f32>> {
        let target_w = self.input_shape[3];
        let target_h = self.input_shape[2];
        let channels = self.input_shape[1];
        
        let gray = img.to_luma8();
        let resized = cv_imgproc::resize_ctx(&gray, target_w as u32, target_h as u32, cv_imgproc::Interpolation::Linear, runner);
        
        let data: Vec<f32> = resized.as_raw().iter().map(|&v| v as f32 / 255.0).collect();

        Tensor::from_vec(data, cv_core::TensorShape::new(channels, target_h, target_w)).map_err(|e| DnnError::Preprocessing(e.to_string()))
    }
}
