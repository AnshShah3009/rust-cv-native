use nalgebra::{ArrayStorage, Matrix, Matrix3, Matrix4, Vector, Vector3, Vector4};
use std::fmt;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DeviceType {
    Cpu,
    Cuda,
    Vulkan,
    Metal,
    Dml,
    TensorRT,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DataType {
    U8,
    U16,
    U32,
    I32,
    F32,
    F64,
}

impl DataType {
    pub fn size(&self) -> usize {
        match self {
            DataType::U8 => 1,
            DataType::U16 => 2,
            DataType::U32 | DataType::I32 | DataType::F32 => 4,
            DataType::F64 => 8,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TensorShape {
    pub channels: usize,
    pub height: usize,
    pub width: usize,
}

impl TensorShape {
    pub fn new(channels: usize, height: usize, width: usize) -> Self {
        Self {
            channels,
            height,
            width,
        }
    }

    pub fn hw(&self) -> (usize, usize) {
        (self.height, self.width)
    }

    pub fn chw(&self) -> (usize, usize, usize) {
        (self.channels, self.height, self.width)
    }

    pub fn len(&self) -> usize {
        self.channels * self.height * self.width
    }

    pub fn is_1d(&self) -> bool {
        self.height == 1 && self.width == 1
    }

    pub fn is_2d(&self) -> bool {
        self.channels == 1
    }

    pub fn is_3d(&self) -> bool {
        self.channels > 1
    }
}

#[derive(Debug, Clone)]
pub struct Tensor<T: Clone + Copy> {
    pub data: Vec<T>,
    pub shape: TensorShape,
    pub dtype: DataType,
    pub device: DeviceType,
}

impl<T: Clone + Copy> Tensor<T> {
    pub fn new(shape: TensorShape) -> Self {
        let dtype = match std::any::type_name::<T>() {
            "u8" => DataType::U8,
            "u16" => DataType::U16,
            "u32" => DataType::U32,
            "i32" => DataType::I32,
            "f32" => DataType::F32,
            "f64" => DataType::F64,
            _ => DataType::F32,
        };

        Self {
            data: vec![T::default(); shape.len()],
            shape,
            dtype,
            device: DeviceType::Cpu,
        }
    }

    pub fn from_vec(data: Vec<T>, shape: TensorShape) -> Self {
        assert_eq!(data.len(), shape.len(), "Data size mismatch with shape");
        Self {
            data,
            shape,
            dtype: DataType::F32,
            device: DeviceType::Cpu,
        }
    }

    pub fn zeros(shape: TensorShape) -> Self
    where
        T: Default,
    {
        Self::new(shape)
    }

    pub fn ones(shape: TensorShape) -> Self
    where
        T: Default + Clone,
    {
        let mut t = Self::new(shape);
        t.data.fill(T::default());
        t
    }

    pub fn reshape(&self, new_shape: TensorShape) -> Self {
        assert_eq!(
            self.data.len(),
            new_shape.len(),
            "Cannot reshape: size mismatch"
        );
        Self {
            data: self.data.clone(),
            shape: new_shape,
            dtype: self.dtype,
            device: self.device,
        }
    }

    pub fn as_slice(&self) -> &[T] {
        &self.data
    }

    pub fn as_mut_slice(&mut self) -> &mut [T] {
        &mut self.data
    }

    pub fn index(&self, c: usize, h: usize, w: usize) -> T {
        let idx = c * self.shape.height * self.shape.width + h * self.shape.width + w;
        self.data[idx]
    }

    pub fn index_mut(&mut self, c: usize, h: usize, w: usize) -> &mut T {
        let idx = c * self.shape.height * self.shape.width + h * self.shape.width + w;
        &mut self.data[idx]
    }
}

impl Tensor<f32> {
    pub fn from_image_gray(data: &[u8], width: usize, height: usize) -> Self {
        let mut float_data = Vec::with_capacity(width * height);
        for &pixel in data {
            float_data.push(pixel as f32 / 255.0);
        }
        Self::from_vec(float_data, TensorShape::new(1, height, width))
    }

    pub fn from_image_rgb(data: &[u8], width: usize, height: usize) -> Self {
        let mut float_data = Vec::with_capacity(3 * width * height);
        for chunk in data.chunks(3) {
            float_data.push(chunk[0] as f32 / 255.0);
            float_data.push(chunk[1] as f32 / 255.0);
            float_data.push(chunk[2] as f32 / 255.0);
        }
        Self::from_vec(float_data, TensorShape::new(3, height, width))
    }

    pub fn to_image_gray(&self) -> Vec<u8> {
        assert!(self.shape.is_2d(), "Tensor must be 2D for gray image");
        self.data
            .iter()
            .map(|&v| (v.clamp(0.0, 1.0) * 255.0) as u8)
            .collect()
    }
}

impl<T: Clone + Copy> fmt::Display for Tensor<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Tensor({}, {}, {}, {:?})",
            self.shape.channels, self.shape.height, self.shape.width, self.device
        )
    }
}

pub type Tensor3f = Tensor<f32>;
pub type Tensor4f = Tensor<f64>;

pub fn create_tensor_2d<T: Clone + Copy>(height: usize, width: usize) -> Tensor<T> {
    Tensor::new(TensorShape::new(1, height, width))
}

pub fn create_tensor_3d<T: Clone + Copy>(
    channels: usize,
    height: usize,
    width: usize,
) -> Tensor<T> {
    Tensor::new(TensorShape::new(channels, height, width))
}
